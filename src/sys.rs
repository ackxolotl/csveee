use std::sync::LazyLock;

/// CPU page size in bytes (typically 4 KiB, 16 KiB on Apple Silicon).
pub fn page_size() -> usize {
    static PAGE_SIZE: LazyLock<usize> = LazyLock::new(imp::page_size);
    *PAGE_SIZE
}

/// Alignment unit required for memory mappings.
///
/// On Unix this equals the page size. On Windows this is
/// `dwAllocationGranularity` (typically 64 KiB).
pub fn allocation_granularity() -> usize {
    static GRANULARITY: LazyLock<usize> = LazyLock::new(imp::allocation_granularity);
    *GRANULARITY
}

/// Number of hardware threads available to the process.
///
/// Falls back to 1 if the OS cannot determine the count.
pub fn available_parallelism() -> usize {
    static PARALLELISM: LazyLock<usize> = LazyLock::new(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    *PARALLELISM
}

/// Total physical RAM in bytes, as reported by the kernel.
///
/// Does not account for cgroup / container memory limits, which can be
/// tighter than physical RAM on shared hosts. Returns `None` if the OS
/// cannot determine the value.
pub fn total_ram() -> Option<usize> {
    static TOTAL_RAM: LazyLock<Option<usize>> = LazyLock::new(imp::total_ram);
    *TOTAL_RAM
}

/// Raw platform queries, cached by the public wrappers above.
#[cfg(unix)]
mod imp {
    pub(super) fn page_size() -> usize {
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }

    pub(super) fn allocation_granularity() -> usize {
        super::page_size()
    }

    pub(super) fn total_ram() -> Option<usize> {
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        if pages <= 0 {
            return None;
        }
        Some((pages as usize).saturating_mul(super::page_size()))
    }
}

/// Raw platform queries, cached by the public wrappers above.
#[cfg(windows)]
mod imp {
    use windows_sys::Win32::System::SystemInformation::{
        GetSystemInfo, GlobalMemoryStatusEx, MEMORYSTATUSEX, SYSTEM_INFO,
    };

    fn system_info() -> SYSTEM_INFO {
        let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
        unsafe { GetSystemInfo(&mut info) };
        info
    }

    pub(super) fn page_size() -> usize {
        system_info().dwPageSize as usize
    }

    pub(super) fn allocation_granularity() -> usize {
        system_info().dwAllocationGranularity as usize
    }

    pub(super) fn total_ram() -> Option<usize> {
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
            return None;
        }
        Some(status.ullTotalPhys as usize)
    }
}
