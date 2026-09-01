mod file;
mod reader;

use std::io;
use std::path::Path;

pub use self::file::MmapFile;
use self::reader::MmapChunkReader;
use super::IoContext;

/// Per-thread mmap I/O context.
///
/// Each thread creates its own MAP_PRIVATE mapping so COW pages
/// are isolated.
pub(crate) struct MmapIo {
    mmap: MmapFile,
}

impl MmapIo {
    pub fn new(path: &Path, file_size: usize) -> io::Result<Self> {
        Ok(Self {
            mmap: MmapFile::open(path, file_size)?,
        })
    }
}

impl IoContext for MmapIo {
    type Reader<'a> = MmapChunkReader<'a>;

    fn chunk_reader(
        &mut self,
        chunk_start: usize,
        chunk_end: usize,
    ) -> io::Result<MmapChunkReader<'_>> {
        debug_assert!(chunk_start <= chunk_end);
        self.mmap.reset()?;
        Ok(self.mmap.chunk_reader(chunk_start, chunk_end))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::io::ChunkReader;

    /// The shape `try_assumptions` produces: a pass rewrites mapped bytes in
    /// place, then the next assumption reads the same chunk and must not see
    /// them — including past `chunk_end`, which a pass reaches while
    /// finishing a straddling record.
    #[test]
    fn reset_restores_what_the_previous_reader_dirtied() {
        let page = crate::sys::page_size();
        let len = page * 4;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&vec![b'A'; len]).unwrap();
        tmp.flush().unwrap();

        let mut io = MmapIo::new(tmp.path(), len).unwrap();
        {
            let mut reader = io.chunk_reader(0, page).unwrap();
            let buf = reader.buffer_mut();
            buf[0] = b'Z';
            buf[page + 10] = b'Z';
        }

        let reader = io.chunk_reader(0, page).unwrap();
        let buf = reader.buffer();
        assert_eq!(buf[0], b'A', "in-range write survived");
        assert_eq!(buf[page + 10], b'A', "past-chunk-end write survived");
    }
}
