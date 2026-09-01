pub mod memory;
#[cfg(unix)]
pub mod mmap;
pub mod ringbuf;
pub mod slice;

/// A per-thread I/O context that creates chunk readers.
pub(crate) trait IoContext {
    type Reader<'a>: ChunkReader
    where
        Self: 'a;

    /// Create a chunk reader for the byte range [chunk_start, chunk_end).
    fn chunk_reader(
        &mut self,
        chunk_start: usize,
        chunk_end: usize,
    ) -> std::io::Result<Self::Reader<'_>>;
}

/// A reader that provides sequential access to a chunk's data.
///
/// Implementations must allow reading past the nominal chunk boundary
/// so the parser can finish the last owned record.
pub trait ChunkReader {
    /// Returns all data available from the current position.
    ///
    /// For mmap, this is the remainder of the mapped region.
    /// For ring buffers, this is the consumable buffer contents.
    fn buffer(&self) -> &[u8];

    /// Returns a mutable view of the available data.
    ///
    /// Needed for in-place escape character removal.
    fn buffer_mut(&mut self) -> &mut [u8];

    /// Fills the buffer so that `buffer()` holds at least `n` bytes.
    ///
    /// Implementations must keep reading until the request is met, growing
    /// their storage if needed; more may be left available. Returning short
    /// is permitted **only** at EOF — that is how the parser detects EOF.
    fn fill(&mut self, n: usize) -> std::io::Result<()>;

    /// Advance the read position by `n` bytes, releasing processed data.
    ///
    /// For mmap, this may issue MADV_DONTNEED on completed pages.
    /// For ring buffers, this advances the consumer pointer.
    fn consume(&mut self, n: usize);

    /// Bytes remaining before the nominal chunk boundary.
    ///
    /// Returns 0 when the reader has advanced past the chunk boundary.
    /// The parser stops starting new records once this reaches 0.
    fn remaining_in_chunk(&self) -> usize;
}
