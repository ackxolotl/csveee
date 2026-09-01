use std::io::{self, Cursor};

use super::IoContext;
use super::ringbuf::{RingBuf, RingBufChunkReader, RingBufSettings};

/// Per-thread I/O context over a caller-supplied byte slice.
///
/// This copies per *chunk*, into the same ring buffer the file-backed path
/// uses, with a `Cursor` standing in for the file.
pub(crate) struct SliceIo<'a> {
    buf: RingBuf,
    src: Cursor<&'a [u8]>,
    len: usize,
    read_cap: usize,
}

impl<'a> SliceIo<'a> {
    pub fn new(data: &'a [u8], settings: RingBufSettings) -> io::Result<Self> {
        let buf = RingBuf::new(settings.buffer_size, settings.buffer_limit)?;
        Ok(Self {
            buf,
            src: Cursor::new(data),
            len: data.len(),
            read_cap: settings.buffer_size,
        })
    }
}

impl<'a> IoContext for SliceIo<'a> {
    type Reader<'b>
        = RingBufChunkReader<'b, Cursor<&'a [u8]>>
    where
        Self: 'b;

    fn chunk_reader(
        &mut self,
        chunk_start: usize,
        chunk_end: usize,
    ) -> io::Result<RingBufChunkReader<'_, Cursor<&'a [u8]>>> {
        self.buf.reset();
        self.src.set_position(chunk_start as u64);
        Ok(RingBufChunkReader::new(
            &mut self.buf,
            &mut self.src,
            Some(self.len - chunk_start),
            chunk_end - chunk_start,
            self.read_cap,
        ))
    }
}
