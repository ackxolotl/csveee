use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;

use super::reader::MmapChunkReader;

/// A memory-mapped file with MAP_PRIVATE (copy-on-write).
pub struct MmapFile {
    ptr: *mut u8,
    len: usize,
    /// Only [`Self::reset`]'s re-map needs the descriptor.
    #[cfg(not(target_os = "linux"))]
    file: Option<File>,
}

impl MmapFile {
    /// Memory-map the given file with MAP_PRIVATE (copy-on-write).
    pub fn open(path: &Path, len: usize) -> io::Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                #[cfg(not(target_os = "linux"))]
                file: None,
            });
        }

        let file = File::open(path)?;

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            ptr: ptr as *mut u8,
            len,
            #[cfg(not(target_os = "linux"))]
            file: Some(file),
        })
    }

    /// Create a chunk reader for the byte range [chunk_start, chunk_end).
    pub fn chunk_reader(&mut self, chunk_start: usize, chunk_end: usize) -> MmapChunkReader<'_> {
        assert!(chunk_start <= chunk_end && chunk_end <= self.len);
        MmapChunkReader::new(
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) },
            chunk_start,
            chunk_end,
        )
    }

    /// Restore the mapping to the file's contents, releasing its pages.
    ///
    /// Resets everything, not just the chunk about to be read: a pass also
    /// rewrites bytes past its chunk end while finishing a record.
    ///
    /// `MADV_DONTNEED` discards private modifications only on Linux, so
    /// elsewhere the mapping has to be re-established instead.
    pub fn reset(&mut self) -> io::Result<()> {
        if self.ptr.is_null() {
            return Ok(());
        }
        let addr = self.ptr as *mut libc::c_void;

        #[cfg(target_os = "linux")]
        {
            if unsafe { libc::madvise(addr, self.len, libc::MADV_DONTNEED) } != 0 {
                return Err(io::Error::last_os_error());
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            let Some(file) = self.file.as_ref() else {
                return Ok(());
            };
            let ptr = unsafe {
                libc::mmap(
                    addr,
                    self.len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_FIXED,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
        }

        Ok(())
    }
}

impl Drop for MmapFile {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }
}
