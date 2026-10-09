use std::io;
use std::ptr::{self, NonNull, slice_from_raw_parts, slice_from_raw_parts_mut};

use super::file::MemFile;

/// A virtually-contiguous ring buffer using double memory mapping.
///
/// The backing memory file is mapped twice consecutively in virtual
/// address space, so data that wraps around the end appears contiguous.
pub struct MirroredBuf {
    _file: MemFile,
    ptr: NonNull<u8>,
    capacity: usize,
    limit: Option<usize>,
    producer: usize,
    consumer: usize,
}

impl MirroredBuf {
    /// Create a ring buffer of at least the given capacity and an optional
    /// maximum capacity.
    pub fn new(capacity: usize, limit: Option<usize>) -> io::Result<Self> {
        let capacity = super::checked_capacity(capacity, limit)?;
        let file = MemFile::new(capacity)?;
        let base = Self::map_double(&file, capacity)?;

        Ok(Self {
            _file: file,
            ptr: NonNull::new(base).unwrap(),
            capacity,
            limit,
            producer: 0,
            consumer: 0,
        })
    }

    #[cfg(unix)]
    fn map_double(file: &MemFile, capacity: usize) -> io::Result<*mut u8> {
        use std::os::fd::AsRawFd;

        let fd = file.fd.as_raw_fd();

        // Reserve virtual address space for 2x capacity.
        let base = unsafe {
            libc::mmap(
                ptr::null_mut(),
                capacity * 2,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        // Map the file into the first half.
        let p = unsafe {
            libc::mmap(
                base,
                capacity,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            unsafe { libc::munmap(base, capacity * 2) };
            return Err(io::Error::last_os_error());
        }

        // Map the same file into the second half.
        let p = unsafe {
            libc::mmap(
                base.add(capacity),
                capacity,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            unsafe { libc::munmap(base, capacity * 2) };
            return Err(io::Error::last_os_error());
        }

        Ok(base as *mut u8)
    }

    #[cfg(windows)]
    fn map_double(file: &MemFile, capacity: usize) -> io::Result<*mut u8> {
        use std::os::windows::io::AsRawHandle;

        use windows_sys::Win32::System::Memory::{
            MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
            MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3, PAGE_NOACCESS,
            PAGE_READWRITE, UnmapViewOfFile, VirtualAlloc2, VirtualFree,
        };

        let handle = file.handle.as_raw_handle() as *mut core::ffi::c_void;

        // Reserve 2x capacity as a placeholder so we can split it and
        // map two views adjacently — the Windows analogue of
        // `mmap(NULL, 2*cap, PROT_NONE, MAP_ANONYMOUS)`. Requires
        // Windows 10 1803+.
        let placeholder = unsafe {
            VirtualAlloc2(
                ptr::null_mut(),
                ptr::null_mut(),
                capacity * 2,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                ptr::null_mut(),
                0,
            )
        };
        if placeholder.is_null() {
            return Err(io::Error::last_os_error());
        }

        // Split the placeholder into two halves so each can be
        // independently replaced by a file view. The first call frees
        // the lower half from the placeholder while preserving its
        // identity as a placeholder; the upper half remains a single
        // placeholder of `capacity` bytes.
        if unsafe {
            VirtualFree(
                placeholder,
                capacity,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            )
        } == 0
        {
            let err = io::Error::last_os_error();
            unsafe { VirtualFree(placeholder, 0, MEM_RELEASE) };
            return Err(err);
        }

        // Replace the lower placeholder with a view of the file.
        let view1 = unsafe {
            MapViewOfFile3(
                handle,
                ptr::null_mut(),
                placeholder,
                0,
                capacity,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                ptr::null_mut(),
                0,
            )
        };
        if view1.Value.is_null() {
            let err = io::Error::last_os_error();
            unsafe {
                VirtualFree(placeholder, 0, MEM_RELEASE);
                VirtualFree((placeholder as *mut u8).add(capacity) as _, 0, MEM_RELEASE);
            }
            return Err(err);
        }

        // Replace the upper placeholder with a second view of the same file.
        let upper = unsafe { (placeholder as *mut u8).add(capacity) } as *mut core::ffi::c_void;
        let view2 = unsafe {
            MapViewOfFile3(
                handle,
                ptr::null_mut(),
                upper,
                0,
                capacity,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                ptr::null_mut(),
                0,
            )
        };
        if view2.Value.is_null() {
            let err = io::Error::last_os_error();
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view1.Value });
                VirtualFree(upper, 0, MEM_RELEASE);
            }
            return Err(err);
        }

        Ok(placeholder as *mut u8)
    }

    /// Total capacity of the ring buffer.
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of bytes available to read.
    pub fn readable(&self) -> usize {
        self.producer.wrapping_sub(self.consumer)
    }

    /// Number of bytes available to write.
    pub fn writable(&self) -> usize {
        self.capacity - self.readable()
    }

    /// Returns the readable region as an immutable slice.
    pub fn data(&self) -> &[u8] {
        let len = self.readable();
        let offset = self.consumer & (self.capacity - 1);
        unsafe { &*slice_from_raw_parts(self.ptr.as_ptr().add(offset), len) }
    }

    /// Returns the readable region as a mutable slice.
    pub fn data_mut(&mut self) -> &mut [u8] {
        let len = self.readable();
        let offset = self.consumer & (self.capacity - 1);
        unsafe { &mut *slice_from_raw_parts_mut(self.ptr.as_ptr().add(offset), len) }
    }

    /// Returns the writable region (space after producer) as an immutable slice.
    #[allow(dead_code)]
    pub fn unused(&mut self) -> &[u8] {
        let len = self.writable();
        let offset = self.producer & (self.capacity - 1);
        unsafe { &*slice_from_raw_parts(self.ptr.as_ptr().add(offset), len) }
    }

    /// Returns the writable region (space after producer) as a mutable slice.
    pub fn unused_mut(&mut self) -> &mut [u8] {
        let len = self.writable();
        let offset = self.producer & (self.capacity - 1);
        unsafe { &mut *slice_from_raw_parts_mut(self.ptr.as_ptr().add(offset), len) }
    }

    /// Mark `n` bytes as produced (written).
    pub fn produce(&mut self, n: usize) {
        debug_assert!(n <= self.writable());
        self.producer = self.producer.wrapping_add(n);
    }

    /// Mark `n` bytes as consumed (read).
    pub fn consume(&mut self, n: usize) {
        debug_assert!(n <= self.readable());
        self.consumer = self.consumer.wrapping_add(n);
    }

    /// Reset the buffer to empty, discarding all data.
    pub fn reset(&mut self) {
        self.producer = 0;
        self.consumer = 0;
    }

    /// Double the buffer capacity, preserving existing data.
    ///
    /// Errors if doubling would exceed the configured `limit`.
    pub fn grow(&mut self) -> io::Result<()> {
        let new_capacity = super::grown_capacity(self.capacity, self.limit)?;
        let mut new = Self::new(new_capacity, self.limit)?;
        let readable = self.readable();
        if readable > 0 {
            new.unused_mut()[..readable].copy_from_slice(self.data());
            new.produce(readable);
        }
        *self = new;
        Ok(())
    }
}

#[cfg(unix)]
impl Drop for MirroredBuf {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.capacity * 2);
        }
    }
}

#[cfg(windows)]
impl Drop for MirroredBuf {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.ptr.as_ptr() as _,
            });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.ptr.as_ptr().add(self.capacity) as _,
            });
        }
    }
}
