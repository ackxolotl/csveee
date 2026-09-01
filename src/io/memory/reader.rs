use std::io;

use crate::io::ChunkReader;

/// A chunk reader backed by an in-memory byte buffer.
pub struct InMemoryChunkReader<'a> {
    data: &'a mut [u8],
    pos: usize,
    chunk_len: usize,
}

impl<'a> InMemoryChunkReader<'a> {
    pub(super) fn new(data: &'a mut [u8], chunk_len: usize) -> Self {
        Self {
            data,
            pos: 0,
            chunk_len,
        }
    }
}

impl ChunkReader for InMemoryChunkReader<'_> {
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
        self.chunk_len.saturating_sub(self.pos)
    }
}
