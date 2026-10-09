use std::io;
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
#[cfg(windows)]
use std::os::windows::io::{FromRawHandle, OwnedHandle, RawHandle};

/// An anonymous, in-memory backing object used for the ring buffer's
/// double mapping.
pub(super) struct MemFile {
    #[cfg(unix)]
    pub fd: OwnedFd,
    #[cfg(windows)]
    pub handle: OwnedHandle,
}

impl MemFile {
    pub fn new(size: usize) -> io::Result<Self> {
        #[cfg(target_os = "linux")]
        {
            let fd = unsafe { libc::memfd_create(c"ringbuf".as_ptr(), libc::MFD_CLOEXEC) };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            if unsafe { libc::ftruncate(fd.as_raw_fd(), size as libc::off_t) } != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Self { fd })
        }

        #[cfg(target_os = "macos")]
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);

            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let name = std::ffi::CString::new(format!("/csveee-{n}"))
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

            let fd = unsafe {
                libc::shm_open(
                    name.as_ptr(),
                    libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                    0o600,
                )
            };
            // Unlink immediately — fd keeps it alive.
            unsafe { libc::shm_unlink(name.as_ptr()) };

            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            if unsafe { libc::ftruncate(fd.as_raw_fd(), size as libc::off_t) } != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Self { fd })
        }

        #[cfg(windows)]
        {
            use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
            use windows_sys::Win32::System::Memory::{CreateFileMappingW, PAGE_READWRITE};

            let size_high = ((size as u64) >> 32) as u32;
            let size_low = (size as u64 & 0xFFFF_FFFF) as u32;
            let handle = unsafe {
                CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    size_high,
                    size_low,
                    std::ptr::null(),
                )
            };
            if handle.is_null() {
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                handle: unsafe { OwnedHandle::from_raw_handle(handle as RawHandle) },
            })
        }
    }
}
