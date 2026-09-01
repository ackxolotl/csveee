use std::io;
use std::ptr::{self, NonNull, slice_from_raw_parts, slice_from_raw_parts_mut};

use super::file::MemFile;
use crate::sys::allocation_granularity;

/// A virtually-contiguous ring buffer using double memory mapping.
///
/// The backing memory file is mapped twice consecutively in virtual
/// address space, so data that wraps around the end appears contiguous.
pub struct RingBuf {
    _file: MemFile,
    ptr: NonNull<u8>,
    capacity: usize,
    limit: Option<usize>,
    producer: usize,
    consumer: usize,
}

impl RingBuf {
    /// Create a ring buffer of at least the given capacity and an optional
    /// maximum capacity.
    pub fn new(capacity: usize, limit: Option<usize>) -> io::Result<Self> {
        let granularity = allocation_granularity();
        let capacity = capacity.next_power_of_two().max(granularity);

        if let Some(limit) = limit
            && capacity > limit
        {
            return Err(io::Error::other(format!(
                "ring buffer capacity ({capacity} bytes, rounded up from the requested size to a \
                 power of two and at least the {granularity}-byte allocation granularity) would \
                 exceed limit ({limit} bytes)",
            )));
        }

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
        let new_capacity = self.capacity * 2;
        if let Some(limit) = self.limit
            && new_capacity > limit
        {
            return Err(io::Error::other(format!(
                "ring buffer growth ({new_capacity} bytes) would exceed limit ({limit} bytes)",
            )));
        }
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
impl Drop for RingBuf {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.capacity * 2);
        }
    }
}

#[cfg(windows)]
impl Drop for RingBuf {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ringbuf_basic() {
        let mut rb = RingBuf::new(allocation_granularity(), None).unwrap();
        let cap = rb.capacity();

        assert_eq!(rb.readable(), 0);
        assert_eq!(rb.writable(), cap);

        rb.unused_mut()[..5].copy_from_slice(b"hello");
        rb.produce(5);

        assert_eq!(rb.readable(), 5);
        assert_eq!(rb.data(), b"hello");

        rb.consume(5);
        assert_eq!(rb.readable(), 0);
        assert_eq!(rb.writable(), cap);
    }

    #[test]
    fn ringbuf_wrap_around() {
        let mut rb = RingBuf::new(allocation_granularity(), None).unwrap();
        let cap = rb.capacity();

        // Advance consumer/producer near the end.
        rb.unused_mut()[..cap - 2].copy_from_slice(&vec![0u8; cap - 2]);
        rb.produce(cap - 2);
        rb.consume(cap - 2);

        // Write across the wrap boundary.
        rb.unused_mut()[..5].copy_from_slice(b"abcde");
        rb.produce(5);

        // Data should appear contiguous despite wrapping.
        assert_eq!(rb.data(), b"abcde");
    }

    #[test]
    fn ringbuf_grow() {
        let mut rb = RingBuf::new(allocation_granularity(), None).unwrap();
        let old_cap = rb.capacity();

        rb.unused_mut()[..3].copy_from_slice(b"abc");
        rb.produce(3);

        rb.grow().unwrap();
        assert_eq!(rb.capacity(), old_cap * 2);
        assert_eq!(rb.readable(), 3);
        assert_eq!(&rb.data()[..3], b"abc");
    }

    #[test]
    fn ringbuf_new_rejects_limit_below_granularity() {
        let Err(err) = RingBuf::new(50, Some(200)) else {
            panic!("expected a ring buffer larger than its limit to be rejected");
        };
        assert!(
            err.to_string().contains("would exceed limit"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ringbuf_new_accepts_limit_at_granularity() {
        let granularity = allocation_granularity();
        let rb = RingBuf::new(1, Some(granularity)).unwrap();
        assert_eq!(rb.capacity(), granularity);
    }
}
