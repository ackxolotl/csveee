mod buf;
mod file;
mod reader;
mod settings;

use std::fs::File;
use std::io::{self, Seek, SeekFrom};
use std::path::Path;

pub use self::buf::RingBuf;
pub use self::reader::RingBufChunkReader;
pub use self::settings::RingBufSettings;
use super::IoContext;

/// Per-thread ring buffer I/O context.
///
/// Owns a ring buffer and an open file handle that are reused across
/// all chunks processed by this thread. Each call to `chunk_reader`
/// resets the buffer and seeks the file to the new chunk start, then
/// hands out a borrowed `RingBufChunkReader`.
pub(crate) struct RingBufIo {
    buf: RingBuf,
    file: File,
    file_size: usize,
    read_cap: usize,
}

impl RingBufIo {
    pub fn new(path: &Path, file_size: usize, settings: RingBufSettings) -> io::Result<Self> {
        let file = File::open(path)?;
        let buf = RingBuf::new(settings.buffer_size, settings.buffer_limit)?;
        Ok(Self {
            buf,
            file,
            file_size,
            read_cap: settings.buffer_size,
        })
    }
}

impl IoContext for RingBufIo {
    type Reader<'a> = RingBufChunkReader<'a, File>;

    fn chunk_reader(
        &mut self,
        chunk_start: usize,
        chunk_end: usize,
    ) -> io::Result<RingBufChunkReader<'_, File>> {
        debug_assert!(chunk_start <= chunk_end);
        self.buf.reset();
        self.file.seek(SeekFrom::Start(chunk_start as u64))?;
        Ok(RingBufChunkReader::new(
            &mut self.buf,
            &mut self.file,
            Some(self.file_size - chunk_start),
            chunk_end - chunk_start,
            self.read_cap,
        ))
    }
}
