use std::cell::UnsafeCell;

/// A slot that is written exactly once by one thread and read after
/// all threads have joined. No synchronization needed beyond the
/// guarantee that each slot is accessed by a single writer.
pub(super) struct WriteOnceSlot<T>(UnsafeCell<Option<T>>);

// Safety: each slot is written by exactly one thread (guaranteed by
// the atomic chunk counter) and only read after `std::thread::scope`
// returns, ensuring all writes are visible.
unsafe impl<T: Send> Sync for WriteOnceSlot<T> {}

impl<T> WriteOnceSlot<T> {
    pub fn new() -> Self {
        Self(UnsafeCell::new(None))
    }

    /// Store a value. Must be called at most once, with no concurrent access.
    pub unsafe fn set(&self, value: T) {
        unsafe { *self.0.get() = Some(value) }
    }

    pub fn into_inner(self) -> Option<T> {
        self.0.into_inner()
    }
}
