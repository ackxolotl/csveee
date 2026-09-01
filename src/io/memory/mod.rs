mod reader;

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use self::reader::InMemoryChunkReader;
use super::IoContext;

/// Per-thread in-memory I/O context.
///
/// Files are read from chunk start to file end into a single buffer.
/// Good for small files, bad for big ones.
pub(crate) struct InMemoryIo {
    file: File,
    file_size: usize,
    data: Vec<u8>,
}

impl InMemoryIo {
    pub fn new(path: &Path, file_size: usize) -> io::Result<Self> {
        Ok(Self {
            file: File::open(path)?,
            file_size,
            data: Vec::new(),
        })
    }
}

impl IoContext for InMemoryIo {
    type Reader<'a> = InMemoryChunkReader<'a>;

    fn chunk_reader(
        &mut self,
        chunk_start: usize,
        chunk_end: usize,
    ) -> io::Result<InMemoryChunkReader<'_>> {
        debug_assert!(chunk_start <= chunk_end);
        self.data.clear();
        self.data.reserve(self.file_size - chunk_start);
        self.file.seek(SeekFrom::Start(chunk_start as u64))?;
        let tail = (self.file_size - chunk_start) as u64;
        (&self.file).take(tail).read_to_end(&mut self.data)?;
        let chunk_len = (chunk_end - chunk_start).min(self.data.len());
        Ok(InMemoryChunkReader::new(&mut self.data, chunk_len))
    }
}
