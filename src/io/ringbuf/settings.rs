//! User-facing tuning for the ring-buffer I/O backend.

/// Per-thread ring-buffer I/O settings.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RingBufSettings {
    pub(crate) buffer_size: usize,
    pub(crate) buffer_limit: Option<usize>,
}

impl Default for RingBufSettings {
    fn default() -> Self {
        Self {
            buffer_size: 16 * 1024,
            buffer_limit: Some(0),
        }
    }
}

impl RingBufSettings {
    /// Initial capacity of one thread's buffer (default: 16 KiB).
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Maximum size one thread's buffer may grow to.
    ///
    /// `Some(0)` (the default) is the auto sentinel and will get resolved
    /// by the scheduler, `None` means unbounded.
    pub fn buffer_limit(mut self, limit: Option<usize>) -> Self {
        self.buffer_limit = limit;
        self
    }
}
