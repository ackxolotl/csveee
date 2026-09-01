use super::output::Output;
use crate::config::Config;
use crate::io::ChunkReader;

/// CSV parse state at a chunk boundary.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Assumption {
    /// The chunk starts outside a quoted field.
    OutOfQuotes,
    /// The chunk starts inside a quoted field.
    InQuotes,
    /// The chunk starts inside a quoted field after an escape character.
    InQuotesAfterEscape,
}

/// Return the assumptions a speculative parser must consider, in priority
/// order, for the given configuration.
pub fn assumptions_for_config(quote: Option<u8>, escape: Option<u8>) -> &'static [Assumption] {
    match (quote, escape) {
        (Some(q), Some(e)) if q == e => &[Assumption::OutOfQuotes, Assumption::InQuotes],
        (Some(_), Some(_)) => &[
            Assumption::OutOfQuotes,
            Assumption::InQuotes,
            Assumption::InQuotesAfterEscape,
        ],
        (Some(_), None) => &[Assumption::OutOfQuotes, Assumption::InQuotes],
        (None, Some(_)) => unreachable!("no escape char when quote char is set"),
        (None, None) => &[Assumption::OutOfQuotes],
    }
}

/// Result of parsing a chunk under one assumption.
pub struct PassResult<S> {
    /// Byte offset of the terminator the first record starts after.
    pub record_start: Option<usize>,
    /// Byte offset of the terminator the last record ends on.
    pub record_end: Option<usize>,
    /// The accumulated state if successful, or the parse error.
    pub result: crate::Result<S>,
}

impl<S> PassResult<S> {
    /// No record starts in this chunk.
    pub(crate) fn no_record(state: S) -> Self {
        Self {
            record_start: None,
            record_end: None,
            result: Ok(state),
        }
    }

    /// The chunk failed before a record start was found.
    pub(crate) fn failed(e: crate::Error) -> Self {
        Self {
            record_start: None,
            record_end: None,
            result: Err(e),
        }
    }

    /// A chunk whose first record starts at `start`; `outcome` carries the
    /// last record end, or the failure that ended the pass.
    pub(crate) fn started(start: usize, outcome: crate::Result<Option<usize>>, state: S) -> Self {
        match outcome {
            Ok(record_end) => Self {
                record_start: Some(start),
                record_end,
                result: Ok(state),
            },
            Err(e) => Self {
                record_start: Some(start),
                record_end: None,
                result: Err(e),
            },
        }
    }
}

/// How to find the first record boundary at the start of a chunk.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum FindRecordStart {
    /// Use strict parsing. Quickly rejects wrong assumptions, the default.
    Strict,
    /// Use lenient parsing. Fallback when all assumptions fail with strict.
    Lenient,
    /// Skip the search entirely — parsing starts from the first byte (chunk 0).
    No,
    /// Skip header rows (chunk 0).
    SkipHeaders,
}

/// A chunk parser processes one chunk under a given assumption.
pub trait ChunkParser {
    fn config(&self) -> &Config;

    /// Check whether this parser implementation supports the given config.
    fn supports(config: &Config) -> Result<(), &'static str>
    where
        Self: Sized;

    /// Consume up to the chunk's first record boundary, returning its
    /// terminator's offset.
    fn scan_record_start<R: ChunkReader>(
        &self,
        reader: &mut R,
        assumption: Assumption,
        strict_probe: bool,
    ) -> crate::Result<Option<usize>>;

    /// As [`ChunkParser::scan_record_start`], but past the header record.
    /// Chunk 0's state is known exactly, so nothing is probed.
    fn scan_header_end<R: ChunkReader>(&self, reader: &mut R) -> crate::Result<Option<usize>>;

    /// Parse records from `base` — where [`skip_empty_lines`] left this
    /// same `reader` — calling `acc` for each, returning the last record end.
    fn parse_from<S, A, R: ChunkReader, O: Output + ?Sized>(
        &self,
        reader: &mut R,
        state: &mut S,
        base: usize,
        acc: &A,
    ) -> crate::Result<Option<usize>>
    where
        A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Parse a chunk, calling `acc` for each complete record.
    #[cfg_attr(feature = "trace", tracing::instrument(skip(self, reader, state, acc)))]
    fn parse<S, A, R: ChunkReader, O: Output + ?Sized>(
        &self,
        reader: &mut R,
        mut state: S,
        assumption: Assumption,
        acc: &A,
        find: FindRecordStart,
    ) -> PassResult<S>
    where
        A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        let scanned = match find {
            FindRecordStart::No => Ok(Some(0)),
            FindRecordStart::SkipHeaders => self.scan_header_end(reader),
            FindRecordStart::Strict => self.scan_record_start(reader, assumption, true),
            FindRecordStart::Lenient => self.scan_record_start(reader, assumption, false),
        };
        let record_start = match scanned {
            Ok(Some(offset)) => offset,
            Ok(None) => return PassResult::no_record(state),
            Err(e) => return PassResult::failed(e),
        };

        // The boundary is known from here on, so every outcome reports it.
        let outcome = match skip_empty_lines(reader, self.config()) {
            // The run outlasts the chunk: a boundary, but no body.
            Ok((skipped, true)) => Ok(Some(record_start + skipped)),
            Ok((skipped, false)) => {
                self.parse_from::<S, A, R, O>(reader, &mut state, record_start + skipped, acc)
            }
            Err(e) => Err(e),
        };
        PassResult::started(record_start, outcome, state)
    }
}

/// Consume the run of bare terminator bytes at the reader's position, so
/// every backend's body starts on real content. The `bool` says the run
/// was still going at the chunk boundary, so the chunk holds no record.
pub(super) fn skip_empty_lines<R: ChunkReader>(
    reader: &mut R,
    config: &Config,
) -> crate::Result<(usize, bool)> {
    let (term_a, term_b) = config.terminator.bytes();
    let is_term = |c: u8| c == term_a || term_b == Some(c);
    let mut consumed = 0usize;
    loop {
        reader.fill(1)?;
        let buf = reader.buffer();
        if buf.is_empty() || !is_term(buf[0]) {
            return Ok((consumed, false));
        }
        let rem = reader.remaining_in_chunk();
        if rem == 0 {
            return Ok((consumed, true));
        }
        // `buf[0]` is a terminator and `rem >= 1`, so this always advances.
        let cap = buf.len().min(rem);
        let mut i = 0;
        while i < cap && is_term(buf[i]) {
            i += 1;
        }
        reader.consume(i);
        consumed += i;
    }
}
