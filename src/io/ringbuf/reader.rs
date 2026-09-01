use std::io::{self, Read};

use super::buf::RingBuf;
use crate::io::ChunkReader;

/// A chunk reader that borrows a ring buffer and a byte source.
///
/// Generic over `R: Read` so the same struct serves both the parallel
/// file-backed path (a `std::fs::File` held by `RingBufIo`) and the
/// single-threaded `Parser::parse_stream` path that reads from a
/// user-supplied `R: Read`. The fill/grow/consume logic is identical for
/// both; only the source of bytes differs.
///
/// `file_remaining = None` signals an unbounded source — the reader will
/// keep calling `src.read` until it returns 0. For the fd path, the file
/// length is known up front and is supplied as `Some`.
pub struct RingBufChunkReader<'a, R: Read> {
    buf: &'a mut RingBuf,
    src: &'a mut R,
    /// Bytes remaining in the source from the current read position,
    /// `None` for unbounded sources (streams).
    file_remaining: Option<usize>,
    /// Bytes remaining until the nominal chunk boundary.
    chunk_remaining: usize,
    /// Maximum bytes per `read()` call.
    read_cap: usize,
}

impl<'a, R: Read> RingBufChunkReader<'a, R> {
    /// Create a chunk reader borrowing the given buffer and source.
    pub(crate) fn new(
        buf: &'a mut RingBuf,
        src: &'a mut R,
        file_remaining: Option<usize>,
        chunk_remaining: usize,
        read_cap: usize,
    ) -> Self {
        Self {
            buf,
            src,
            file_remaining,
            chunk_remaining,
            read_cap,
        }
    }

    /// Read from the source into the ring buffer's unused space.
    fn read_into_buf(&mut self) -> io::Result<usize> {
        let writable = self.buf.unused_mut();
        let mut to_read = writable.len().min(self.read_cap);
        if let Some(rem) = self.file_remaining {
            to_read = to_read.min(rem);
        }
        if to_read == 0 {
            return Ok(0);
        }

        let n = self.src.read(&mut writable[..to_read])?;

        self.buf.produce(n);
        if let Some(rem) = &mut self.file_remaining {
            *rem -= n;
        }
        Ok(n)
    }
}

impl<R: Read> ChunkReader for RingBufChunkReader<'_, R> {
    fn buffer(&self) -> &[u8] {
        self.buf.data()
    }

    fn buffer_mut(&mut self) -> &mut [u8] {
        self.buf.data_mut()
    }

    fn fill(&mut self, n: usize) -> io::Result<()> {
        while self.buf.readable() < n && self.file_remaining.is_none_or(|r| r > 0) {
            if self.buf.writable() == 0 {
                #[cfg(feature = "trace")]
                let old_capacity = self.buf.capacity();
                self.buf.grow()?;
                #[cfg(feature = "trace")]
                {
                    let new_capacity = self.buf.capacity();
                    crate::trace::debug!(old_capacity, new_capacity, "needed to grow buffer",);
                }
            }
            let read = self.read_into_buf()?;
            if read == 0 {
                break;
            }
        }
        Ok(())
    }

    fn consume(&mut self, n: usize) {
        self.chunk_remaining = self.chunk_remaining.saturating_sub(n);
        self.buf.consume(n);
    }

    fn remaining_in_chunk(&self) -> usize {
        self.chunk_remaining
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use crate::io::ringbuf::{RingBufIo, RingBufSettings};
    use crate::io::{ChunkReader, IoContext};

    fn io_for(bytes: &[u8]) -> (tempfile::NamedTempFile, RingBufIo) {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(bytes).unwrap();
        let settings = RingBufSettings::default().buffer_limit(None);
        let io = RingBufIo::new(tmp.path(), bytes.len(), settings).unwrap();
        (tmp, io)
    }

    #[test]
    fn chunk_reader_single_chunk() {
        let (_tmp, mut io) = io_for(b"a,b\n1,2\n3,4\n");
        let mut reader = io.chunk_reader(0, 12).unwrap();

        assert_eq!(reader.remaining_in_chunk(), 12);

        reader.fill(12).unwrap();
        assert_eq!(reader.buffer(), b"a,b\n1,2\n3,4\n");

        reader.consume(4);
        assert_eq!(reader.remaining_in_chunk(), 8);
        assert_eq!(reader.buffer(), b"1,2\n3,4\n");
    }

    #[test]
    fn chunk_reader_epilogue() {
        let (_tmp, mut io) = io_for(b"a,b\nc,d\ne,f\n");
        // Chunk is [0, 5), but file is 12 bytes — reader can read past.
        let mut reader = io.chunk_reader(0, 5).unwrap();

        assert_eq!(reader.remaining_in_chunk(), 5);

        reader.fill(12).unwrap();
        assert_eq!(reader.buffer(), b"a,b\nc,d\ne,f\n");

        reader.consume(8);
        assert_eq!(reader.remaining_in_chunk(), 0);
        assert_eq!(reader.buffer(), b"e,f\n");
    }

    #[test]
    fn chunk_reader_second_chunk() {
        let (_tmp, mut io) = io_for(b"a,b\nc,d\ne,f\n");
        // Second chunk [4, 8) of a 12-byte file.
        let mut reader = io.chunk_reader(4, 8).unwrap();

        assert_eq!(reader.remaining_in_chunk(), 4);

        reader.fill(8).unwrap();
        assert_eq!(reader.buffer(), b"c,d\ne,f\n");
    }
}
