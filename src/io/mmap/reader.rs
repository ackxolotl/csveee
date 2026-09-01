use std::io;

use crate::io::ChunkReader;

/// A chunk reader backed by a memory-mapped region.
pub struct MmapChunkReader<'a> {
    data: &'a mut [u8],
    pos: usize,
    chunk_end: usize,
}

impl<'a> MmapChunkReader<'a> {
    pub(super) fn new(data: &'a mut [u8], chunk_start: usize, chunk_end: usize) -> Self {
        Self {
            data,
            pos: chunk_start,
            chunk_end,
        }
    }
}

impl ChunkReader for MmapChunkReader<'_> {
    fn buffer(&self) -> &[u8] {
        &self.data[self.pos..]
    }

    fn buffer_mut(&mut self) -> &mut [u8] {
        &mut self.data[self.pos..]
    }

    fn fill(&mut self, _n: usize) -> io::Result<()> {
        Ok(())
    }

    fn consume(&mut self, n: usize) {
        self.pos += n;
    }

    fn remaining_in_chunk(&self) -> usize {
        self.chunk_end.saturating_sub(self.pos)
    }
}
