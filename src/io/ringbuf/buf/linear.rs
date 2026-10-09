use std::io;

/// A heap buffer that keeps its readable data contiguous by copying.
///
/// The fallback for targets without double mapping. Readable data lives in
/// `buf[head..tail]`. Before handing out space to write, the leftover bytes
/// move to the front, so the free space is always one slice at the end.
/// The leftover is usually a partial record, which keeps the copy small.
pub struct LinearBuf {
    buf: Vec<u8>,
    head: usize,
    tail: usize,
    limit: Option<usize>,
}

impl LinearBuf {
    /// Create a buffer of at least the given capacity and an optional
    /// maximum capacity.
    pub fn new(capacity: usize, limit: Option<usize>) -> io::Result<Self> {
        let capacity = super::checked_capacity(capacity, limit)?;
        Ok(Self {
            buf: vec![0; capacity],
            head: 0,
            tail: 0,
            limit,
        })
    }

    /// Total capacity of the buffer.
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Number of bytes available to read.
    pub fn readable(&self) -> usize {
        self.tail - self.head
    }

    /// Number of bytes available to write.
    pub fn writable(&self) -> usize {
        self.buf.len() - self.readable()
    }

    /// Returns the readable region as an immutable slice.
    pub fn data(&self) -> &[u8] {
        &self.buf[self.head..self.tail]
    }

    /// Returns the readable region as a mutable slice.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.buf[self.head..self.tail]
    }

    /// Returns the writable region as a mutable slice, all
    /// [`writable`](Self::writable) bytes of it.
    ///
    /// Moves the readable data to the front first if it is not there yet.
    pub fn unused_mut(&mut self) -> &mut [u8] {
        if self.head != 0 {
            self.buf.copy_within(self.head..self.tail, 0);
            self.tail -= self.head;
            self.head = 0;
        }
        &mut self.buf[self.tail..]
    }

    /// Mark `n` bytes as produced (written).
    pub fn produce(&mut self, n: usize) {
        debug_assert!(n <= self.buf.len() - self.tail);
        self.tail += n;
    }

    /// Mark `n` bytes as consumed (read).
    pub fn consume(&mut self, n: usize) {
        debug_assert!(n <= self.readable());
        self.head += n;
        if self.head == self.tail {
            // Nothing left to move: start over at the front for free.
            self.reset();
        }
    }

    /// Reset the buffer to empty, discarding all data.
    pub fn reset(&mut self) {
        self.head = 0;
        self.tail = 0;
    }

    /// Double the buffer capacity, preserving existing data.
    ///
    /// Errors if doubling would exceed the configured `limit`.
    pub fn grow(&mut self) -> io::Result<()> {
        let new_capacity = super::grown_capacity(self.buf.len(), self.limit)?;
        self.buf.resize(new_capacity, 0);
        Ok(())
    }
}
