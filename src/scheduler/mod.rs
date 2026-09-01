mod merge;
pub(crate) mod resolve;
mod slot;

use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use self::merge::{ChunkOutput, ResolvedPass};
use self::resolve::{ResolvedConfig, resolve_parser_backend};
use self::slot::WriteOnceSlot;
use crate::config::{Config, IoBackend, ParserBackend};
use crate::io::IoContext;
use crate::io::memory::InMemoryIo;
#[cfg(unix)]
use crate::io::mmap::MmapIo;
use crate::io::ringbuf::{RingBuf, RingBufChunkReader, RingBufIo};
use crate::io::slice::SliceIo;
use crate::parser::dfa::DfaChunkParser;
#[cfg(feature = "simd")]
use crate::parser::simd::SimdChunkParser;
use crate::parser::{Assumption, ChunkParser, FindRecordStart, Output, assumptions_for_config};

/// The scheduler orchestrates parallel chunk processing.
///
/// It implements the speculative parsing strategy:
/// 1. Split the file into chunks.
/// 2. Assign chunks to threads (work-stealing via atomic counter).
/// 3. Each thread parses its chunk under each assumption until one succeeds.
/// 4. Merge phase: resolve the correct assumption per chunk by aligning
///    record boundaries, then call the user's merge function.
#[derive(Debug)]
pub struct Scheduler {
    config: Config,
}

/// Single `parse` invocation against a streaming reader.
fn run_stream_inner<R, S, A, M, Out, O, P>(
    parser: &P,
    config: &Config,
    mut src: R,
    init: impl FnOnce() -> S,
    acc: A,
    merge: M,
) -> crate::Result<Out>
where
    R: Read,
    P: ChunkParser,
    O: Output + ?Sized,
    A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    M: FnOnce(&mut [S]) -> Out,
{
    let settings = resolve::stream_ring_settings(config);
    let mut buf =
        RingBuf::new(settings.buffer_size, settings.buffer_limit).map_err(crate::Error::IO)?;
    let mut reader =
        RingBufChunkReader::new(&mut buf, &mut src, None, usize::MAX, settings.buffer_size);
    let find = if config.has_headers {
        FindRecordStart::SkipHeaders
    } else {
        FindRecordStart::No
    };
    let state = init();
    let pr = parser.parse::<S, A, _, O>(&mut reader, state, Assumption::OutOfQuotes, &acc, find);
    let mut state = pr.result?;
    Ok(merge(std::slice::from_mut(&mut state)))
}

impl Scheduler {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Run the full parse pipeline: open file, split into chunks,
    /// parse in parallel, merge results.
    #[cfg_attr(feature = "trace", tracing::instrument(skip(self, init, acc, merge)))]
    pub fn run<S, I, A, M, R, O: Output + ?Sized>(
        &self,
        file: &Path,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Sync,
        M: FnOnce(&mut [S]) -> R,
    {
        let file_size = std::fs::metadata(file)?.len() as usize;
        let resolved = ResolvedConfig::for_file(&self.config, file_size)?;
        if resolved.chunk_count == 0 {
            return Ok(merge(&mut []));
        }
        self.dispatch(&resolved, file, init, acc, merge)
    }

    /// Pick the parser implementation from the resolved config and forward
    /// to [`Self::dispatch_io`], which does the same for I/O.
    fn dispatch<S, I, A, M, R, O: Output + ?Sized>(
        &self,
        resolved: &ResolvedConfig,
        file: &Path,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Sync,
        M: FnOnce(&mut [S]) -> R,
    {
        match resolved.parser_backend {
            ParserBackend::Dfa => {
                let parser = DfaChunkParser::new(resolved.config.clone());
                self.dispatch_io::<S, I, A, M, R, _, O>(resolved, file, &parser, init, acc, merge)
            }
            #[cfg(feature = "simd")]
            ParserBackend::Simd => {
                let parser = SimdChunkParser::new(resolved.config.clone());
                self.dispatch_io::<S, I, A, M, R, _, O>(resolved, file, &parser, init, acc, merge)
            }
            #[cfg(not(feature = "simd"))]
            ParserBackend::Simd => unreachable!("SIMD backend not compiled in"),
            ParserBackend::Auto => unreachable!("resolved by Scheduler::resolve"),
        }
    }

    /// Pick the I/O backend now that the parser is concrete.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_io<S, I, A, M, R, P, O>(
        &self,
        resolved: &ResolvedConfig,
        file: &Path,
        parser: &P,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Sync,
        M: FnOnce(&mut [S]) -> R,
        P: ChunkParser + Sync,
        O: Output + ?Sized,
    {
        match resolved.io_backend {
            #[cfg(unix)]
            IoBackend::Mmap => {
                let make_ctx = || MmapIo::new(file, resolved.file_size);
                self.run_with::<S, I, A, M, R, _, _, O>(
                    resolved,
                    Some(file),
                    parser,
                    make_ctx,
                    init,
                    acc,
                    merge,
                )
            }
            #[cfg(not(unix))]
            IoBackend::Mmap => Err(crate::Error::IO(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "IoBackend::Mmap is not supported on this platform",
            ))),
            IoBackend::RingBuf(settings) => {
                let make_ctx = || RingBufIo::new(file, resolved.file_size, settings);
                self.run_with::<S, I, A, M, R, _, _, O>(
                    resolved,
                    Some(file),
                    parser,
                    make_ctx,
                    init,
                    acc,
                    merge,
                )
            }
            IoBackend::InMemory => {
                let make_ctx = || InMemoryIo::new(file, resolved.file_size);
                self.run_with::<S, I, A, M, R, _, _, O>(
                    resolved,
                    Some(file),
                    parser,
                    make_ctx,
                    init,
                    acc,
                    merge,
                )
            }
            IoBackend::Auto => unreachable!("resolved by Scheduler::resolve"),
        }
    }

    /// Parallel parse of a caller-supplied byte slice.
    pub fn run_slice<S, I, A, M, R, O: Output + ?Sized>(
        &self,
        data: &[u8],
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Sync,
        M: FnOnce(&mut [S]) -> R,
    {
        let resolved = ResolvedConfig::for_slice(&self.config, data.len())?;
        if resolved.chunk_count == 0 {
            return Ok(merge(&mut []));
        }
        let IoBackend::RingBuf(settings) = resolved.io_backend else {
            unreachable!("for_slice resolves io_backend to RingBuf")
        };
        let make_ctx = || SliceIo::new(data, settings);
        match resolved.parser_backend {
            ParserBackend::Dfa => {
                let parser = DfaChunkParser::new(resolved.config.clone());
                self.run_with::<S, I, A, M, R, _, _, O>(
                    &resolved, None, &parser, make_ctx, init, acc, merge,
                )
            }
            #[cfg(feature = "simd")]
            ParserBackend::Simd => {
                let parser = SimdChunkParser::new(resolved.config.clone());
                self.run_with::<S, I, A, M, R, _, _, O>(
                    &resolved, None, &parser, make_ctx, init, acc, merge,
                )
            }
            #[cfg(not(feature = "simd"))]
            ParserBackend::Simd => unreachable!("SIMD backend not compiled in"),
            ParserBackend::Auto => unreachable!("resolved by Scheduler::resolve_slice"),
        }
    }

    /// Sequential single-threaded parse from a user-supplied `Read`.
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(skip(self, src, init, acc, merge))
    )]
    pub fn run_stream<R, S, A, M, Out, O>(
        &self,
        src: R,
        init: impl FnOnce() -> S,
        acc: A,
        merge: M,
    ) -> crate::Result<Out>
    where
        R: Read,
        O: Output + ?Sized,
        A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
        M: FnOnce(&mut [S]) -> Out,
    {
        let parser_backend = resolve_parser_backend(&self.config)?;
        match parser_backend {
            ParserBackend::Dfa => {
                let parser = DfaChunkParser::new(self.config.clone());
                run_stream_inner(&parser, &self.config, src, init, acc, merge)
            }
            #[cfg(feature = "simd")]
            ParserBackend::Simd => {
                let parser = SimdChunkParser::new(self.config.clone());
                run_stream_inner(&parser, &self.config, src, init, acc, merge)
            }
            #[cfg(not(feature = "simd"))]
            ParserBackend::Simd => unreachable!("SIMD backend not compiled in"),
            ParserBackend::Auto => unreachable!("resolved by resolve_parser_backend"),
        }
    }

    /// Run the parse pipeline with a specific I/O backend and chunk parser.
    #[allow(clippy::too_many_arguments)]
    fn run_with<S, I, A, M, R, P: ChunkParser + Sync, C: IoContext, O: Output + ?Sized>(
        &self,
        resolved: &ResolvedConfig,
        file_path: Option<&Path>,
        parser: &P,
        make_ctx: impl Fn() -> std::io::Result<C> + Sync,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Sync,
        M: FnOnce(&mut [S]) -> R,
    {
        // The speculation strategy — which assumptions to try and in what
        // order — lives here in the scheduler. Chunk parsers consume one
        // assumption at a time; they don't decide the policy.
        let assumptions = assumptions_for_config(resolved.config.quote, resolved.config.escape);
        let chunk_count = resolved.chunk_count;
        let thread_count = resolved.thread_count;
        let file_size = resolved.file_size;

        let results: Vec<WriteOnceSlot<ChunkOutput<S>>> =
            (0..chunk_count).map(|_| WriteOnceSlot::new()).collect();

        let work_loop = |initial_chunk: usize, next_chunk: &AtomicUsize| -> crate::Result<()> {
            let mut ctx = make_ctx().map_err(crate::Error::IO)?;

            let mut idx = initial_chunk;
            loop {
                let output = self.process_chunk::<S, I, A, _, _, O>(
                    resolved,
                    idx,
                    file_path,
                    &mut ctx,
                    parser,
                    assumptions,
                    &init,
                    &acc,
                );
                unsafe { results[idx].set(output) };

                idx = next_chunk.fetch_add(1, Ordering::Relaxed);
                if idx >= chunk_count {
                    break;
                }
            }
            Ok(())
        };

        let next_chunk = AtomicUsize::new(thread_count);

        if thread_count == 1 {
            work_loop(0, &next_chunk)?;
        } else {
            let next_chunk = &next_chunk;
            let errors: Vec<crate::Error> = std::thread::scope(|scope| {
                let handles: Vec<_> = (1..thread_count)
                    .map(|t| scope.spawn(move || work_loop(t, next_chunk)))
                    .collect();

                let mut errors = Vec::new();
                let mut panic_payload = None;
                if let Err(e) = work_loop(0, next_chunk) {
                    errors.push(e);
                }
                for handle in handles {
                    match handle.join() {
                        Ok(Err(e)) => errors.push(e),
                        Err(payload) if panic_payload.is_none() => {
                            panic_payload = Some(payload);
                        }
                        _ => {}
                    }
                }
                if let Some(payload) = panic_payload {
                    std::panic::resume_unwind(payload);
                }
                errors
            });
            if let Some(e) = errors.into_iter().next() {
                return Err(e);
            }
        }

        // Build a reparse callback for chunks where speculation fails.
        let reparse = |record_start: usize, chunk_end: usize| {
            reparse_chunk::<S, I, A, P, C, _, O>(
                record_start,
                chunk_end,
                file_size,
                &make_ctx,
                parser,
                &init,
                &acc,
            )
        };

        merge::collect_and_merge(&resolved.config, results, merge, file_path, reparse)
    }

    /// Process a single chunk: try assumptions in priority order, stop at first success.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(feature = "trace"), allow(unused_variables))]
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(skip(self, ctx, parser, init, acc))
    )]
    fn process_chunk<S, I, A, P: ChunkParser + Sync, C: IoContext, O: Output + ?Sized>(
        &self,
        resolved: &ResolvedConfig,
        chunk_idx: usize,
        file_path: Option<&Path>,
        ctx: &mut C,
        parser: &P,
        assumptions: &[Assumption],
        init: &I,
        acc: &A,
    ) -> ChunkOutput<S>
    where
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Sync,
    {
        let chunk_size = resolved.config.chunk_size;
        let chunk_start = chunk_idx * chunk_size;
        let chunk_end = (chunk_start + chunk_size).min(resolved.file_size);
        let find = resolved.find_mode(chunk_idx);

        let empty = || ChunkOutput {
            passes: vec![],
            record_end: None,
        };

        match self.try_assumptions::<S, I, A, _, _, O>(
            resolved,
            chunk_start,
            chunk_end,
            ctx,
            parser,
            assumptions,
            init,
            acc,
            find,
        ) {
            TryOutcome::Resolved { passes, record_end } => ChunkOutput { passes, record_end },
            TryOutcome::CleanNoBoundary => empty(),
            TryOutcome::AllFailed if find == FindRecordStart::Strict => {
                // Every strict assumption errored during find_record_start.
                // The strict DFA may be rejecting bytes that the user's
                // (lenient) dialect accepts — retry under lenient.
                #[cfg(feature = "trace")]
                if crate::trace::enabled!(crate::trace::Level::DEBUG)
                    && let Some((escaped, marker)) =
                        file_path.and_then(|p| merge::dump_offset_context(p, chunk_start, None))
                {
                    crate::trace::debug!(
                        chunk_idx,
                        chunk_start,
                        "strict failed context:\n{}\n{}",
                        escaped,
                        marker,
                    );
                }
                crate::trace::debug!(
                    chunk_idx,
                    chunk_start,
                    "strict failed, retrying with lenient"
                );
                match self.try_assumptions::<S, I, A, _, _, O>(
                    resolved,
                    chunk_start,
                    chunk_end,
                    ctx,
                    parser,
                    assumptions,
                    init,
                    acc,
                    FindRecordStart::Lenient,
                ) {
                    TryOutcome::Resolved { passes, record_end } => {
                        ChunkOutput { passes, record_end }
                    }
                    // `CleanNoBoundary`: no record starts here, and a
                    // later chunk's mismatch reparses across the gap.
                    // `AllFailed`: unreachable — neither find path errors
                    // with the probe off — and merge would skip the chunk.
                    TryOutcome::CleanNoBoundary | TryOutcome::AllFailed => empty(),
                }
            }
            // find == No / SkipHeaders / Lenient: no retry path, just
            // forward the empty outcome.
            TryOutcome::AllFailed => empty(),
        }
    }

    /// Try each assumption in priority order with the given find mode.
    ///
    /// Returns:
    /// - `Resolved` when at least one assumption produced a pass (successful
    ///   or errored-after-boundary).
    /// - `CleanNoBoundary` when no assumption produced a pass but at least one
    ///   scanned to chunk_end cleanly (`Ok(None)` from `find_record_start`):
    ///   no boundary starts in this chunk. A lenient retry can't help — the
    ///   scan is identical, only the strict probe differs. (We must try *all*
    ///   assumptions first: a chunk opening mid-quoted-field scans cleanly
    ///   under OutOfQuotes yet InQuotes finds the real boundary.)
    /// - `AllFailed` when every assumption errored during `find_record_start`.
    ///   Only under strict is this recoverable via a lenient retry.
    #[allow(clippy::too_many_arguments)]
    // `resolved` is read only to derive `chunk_idx` for instrumentation.
    #[cfg_attr(not(feature = "trace"), allow(unused_variables))]
    fn try_assumptions<S, I, A, P: ChunkParser + Sync, C: IoContext, O: Output + ?Sized>(
        &self,
        resolved: &ResolvedConfig,
        chunk_start: usize,
        chunk_end: usize,
        ctx: &mut C,
        parser: &P,
        assumptions: &[Assumption],
        init: &I,
        acc: &A,
        find: FindRecordStart,
    ) -> TryOutcome<S>
    where
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Sync,
    {
        let mut passes = Vec::new();
        let mut record_end = None;
        let mut saw_clean = false;
        #[cfg(feature = "trace")]
        let chunk_idx = chunk_start / resolved.config.chunk_size;

        for &assumption in assumptions {
            let state = init();
            let mut reader = ctx
                .chunk_reader(chunk_start, chunk_end)
                .expect("failed to create chunk reader");

            let pr = parser.parse(&mut reader, state, assumption, acc, find);

            #[cfg(feature = "trace")]
            {
                let abs_record_start = pr.record_start.map(|o| chunk_start + o);
                let abs_record_end = pr.record_end.map(|o| chunk_start + o);
                match &pr.result {
                    Ok(_) => crate::trace::debug!(
                        chunk_idx,
                        ?assumption,
                        ?find,
                        record_start = ?abs_record_start,
                        record_end = ?abs_record_end,
                        "assumption ok",
                    ),
                    Err(e) => crate::trace::debug!(
                        chunk_idx,
                        ?assumption,
                        ?find,
                        record_start = ?abs_record_start,
                        error = %e,
                        "assumption failed",
                    ),
                }
            }

            match (pr.record_start, pr.result) {
                (Some(start), result) => {
                    let success = result.is_ok();
                    if success {
                        record_end = pr.record_end.map(|o| chunk_start + o);
                    }
                    // Chunk-local error offsets become absolute here.
                    // No-op for Ok and for variants without a position.
                    let result = result.map_err(|e| e.with_base(chunk_start));
                    passes.push(ResolvedPass {
                        record_start: chunk_start + start,
                        result,
                    });
                    if success {
                        break;
                    }
                }
                (None, Ok(_)) => {
                    // A clean scan under *this* assumption doesn't prove
                    // the chunk has no boundary — see `CleanNoBoundary`
                    // above — so keep trying the others.
                    saw_clean = true;
                    crate::trace::debug!(chunk_idx, ?assumption, ?find, "clean scan, no boundary");
                }
                (None, Err(_)) => {
                    // find_record_start errored under this assumption — try
                    // the next one.
                }
            }
        }

        if !passes.is_empty() {
            TryOutcome::Resolved { passes, record_end }
        } else if saw_clean {
            // Every assumption scanned to chunk_end without a boundary (some
            // may have errored first). A boundary genuinely doesn't start in
            // this chunk; a lenient retry can't help — the scan is the same,
            // only the strict probe differs.
            TryOutcome::CleanNoBoundary
        } else {
            TryOutcome::AllFailed
        }
    }
}

/// Reparse one chunk from a known-correct record boundary.
#[cfg_attr(
    feature = "trace",
    tracing::instrument(skip(make_ctx, parser, init, acc))
)]
fn reparse_chunk<S, I, A, P, C, MC, O>(
    record_start: usize,
    chunk_end: usize,
    file_size: usize,
    make_ctx: &MC,
    parser: &P,
    init: &I,
    acc: &A,
) -> crate::Result<(S, Option<usize>)>
where
    P: ChunkParser + Sync,
    C: IoContext,
    MC: Fn() -> std::io::Result<C>,
    O: Output + ?Sized,
    I: Fn() -> S,
    A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
{
    let chunk_end = chunk_end.min(file_size);
    let mut ctx = make_ctx().map_err(crate::Error::IO)?;
    let mut reader = ctx
        .chunk_reader(record_start, chunk_end)
        .map_err(crate::Error::IO)?;
    let state = init();
    let pr = parser.parse::<S, A, _, O>(
        &mut reader,
        state,
        Assumption::OutOfQuotes,
        acc,
        FindRecordStart::No,
    );
    let result = pr.result.map_err(|e| e.with_base(record_start));
    Ok((result?, pr.record_end))
}

/// Outcome of a single `try_assumptions` call.
enum TryOutcome<S> {
    /// At least one assumption produced a pass.
    Resolved {
        passes: Vec<ResolvedPass<S>>,
        record_end: Option<usize>,
    },
    /// No boundary was found.
    CleanNoBoundary,
    /// Every assumption errored during `find_record_start`.
    AllFailed,
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    #[cfg(feature = "simd")]
    use crate::RecordTerminator;
    use crate::config::QuoteHandling;
    use crate::io::ringbuf::RingBufSettings;
    use crate::test_support::run_matrix;

    fn no_headers_config() -> Config {
        let mut config = Config::default();
        config.has_headers = false;
        config
    }

    /// Write CSV to a temp file and parse via the scheduler.
    fn scheduler_parse(config: Config, csv_data: &[u8]) -> crate::Result<Vec<Vec<String>>> {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(csv_data).unwrap();
        let path = tmp.path().to_path_buf();

        let scheduler = Scheduler::new(config);
        scheduler.run(
            &path,
            Vec::<Vec<String>>::new,
            |state: &mut Vec<Vec<String>>, fields: &mut [&mut str]| {
                state.push(fields.iter().map(|s| s.to_string()).collect());
                Ok(())
            },
            |states| {
                let mut all = Vec::new();
                for s in states {
                    all.extend(std::mem::take(s));
                }
                all
            },
        )
    }

    /// Parse the same bytes via the slice path.
    fn scheduler_parse_slice(config: Config, csv_data: &[u8]) -> crate::Result<Vec<Vec<String>>> {
        let scheduler = Scheduler::new(config);
        scheduler.run_slice(
            csv_data,
            Vec::<Vec<String>>::new,
            |state: &mut Vec<Vec<String>>, fields: &mut [&mut str]| {
                state.push(fields.iter().map(|s| s.to_string()).collect());
                Ok(())
            },
            |states| {
                let mut all = Vec::new();
                for s in states {
                    all.extend(std::mem::take(s));
                }
                all
            },
        )
    }

    /// The slice path must agree with the file path byte for byte. Run both
    /// over the same input and compare, so any divergence in chunking,
    /// speculation or the epilogue shows up as a mismatch rather than
    /// needing an expectation hand-written per case.
    fn assert_slice_matches_file(config: Config, csv_data: &[u8]) {
        let from_file = scheduler_parse(config.clone(), csv_data);
        let from_slice = scheduler_parse_slice(config.clone(), csv_data);
        match (from_file, from_slice) {
            (Ok(f), Ok(s)) => assert_eq!(
                f,
                s,
                "slice/file divergence on {:?} (chunk_size={})",
                String::from_utf8_lossy(csv_data),
                config.chunk_size,
            ),
            (Err(f), Err(s)) => assert_eq!(
                f.to_string(),
                s.to_string(),
                "slice/file error divergence on {:?}",
                String::from_utf8_lossy(csv_data),
            ),
            (f, s) => panic!(
                "slice/file disagree on success for {:?}: file={:?} slice={:?}",
                String::from_utf8_lossy(csv_data),
                f.is_ok(),
                s.is_ok(),
            ),
        }
    }

    /// Inputs picked to hit the seams: chunk-boundary-aligned records,
    /// records longer than a chunk, quotes spanning boundaries (speculation,
    /// and the reparse when it guesses wrong), and the degenerate
    /// empty/no-trailing-newline cases.
    const SEAM_CASES: &[&[u8]] = &[
        b"",
        b"a,b\n",
        b"a,b\nc,d\ne,f\ng,h\n",
        b"a,b\nc,d\ne,f\ng,h",
        b"\"aa\",\"bb\"\n\"cc\",\"dd\"\n\"ee\",\"ff\"\n",
        b"\"a,b\",\"c\nd\"\n\"e\",\"f\"\n",
        b"abcdefghij,klmnopqrst\nx,y\n",
        b"\"\",\"\"\n\"\",\"\"\n",
        b"a,\"b\"\"c\"\nd,\"e\nf\"\ng,h\n",
        // Terminator runs: at small chunk sizes a boundary lands inside
        // one, and a run longer than a chunk leaves a chunk record-less.
        b"a,b\n\n\n\nc,d\ne,f\n",
        b"\n\n\n\n\n\na,b\nc,d\n",
        b"a,b\nc,d\n\n\n\n\n",
        b"a,b\r\r\r\nc,d\r\ne,f\r\n",
        b"a,b\n\r\n\rc,d\r\n\ne,f\r\n",
        b"\n\n\n\n\n\n\n\n",
    ];

    #[test]
    fn slice_matches_file_across_chunk_sizes() {
        // Small chunk sizes force many chunks over tiny inputs, so every
        // record boundary lands at a different offset relative to a chunk.
        for &chunk_size in &[1, 2, 3, 5, 8, 10, 16, 64] {
            for &data in SEAM_CASES {
                let mut config = no_headers_config();
                config.chunk_size = chunk_size;
                config.concurrency = 4;
                assert_slice_matches_file(config, data);
            }
        }
    }

    /// `run_matrix` only varies the DFA, and SIMD needs a fixed field
    /// count, so the run cases get their own sweep.
    #[cfg(feature = "simd")]
    #[test]
    fn terminator_runs_match_across_backends_and_chunk_sizes() {
        const RUN_CASES: &[&[u8]] = &[
            b"a,b\n\n\n\nc,d\ne,f\n",
            b"\n\n\n\n\n\na,b\nc,d\n",
            b"a,b\nc,d\n\n\n\n\n",
            b"a,b\r\r\r\nc,d\r\ne,f\r\n",
            b"a,b\n\r\n\rc,d\r\n\ne,f\r\n",
        ];
        for &data in RUN_CASES {
            let baseline = {
                let mut c = no_headers_config();
                c.field_count = Some(2);
                c.chunk_size = 1 << 20;
                scheduler_parse_slice(c, data).unwrap()
            };
            for &chunk_size in &[1, 2, 3, 5, 8, 10, 16, 64] {
                for backend in [ParserBackend::Dfa, ParserBackend::Simd] {
                    let mut c = no_headers_config();
                    c.field_count = Some(2);
                    c.parser_backend = backend;
                    c.chunk_size = chunk_size;
                    c.concurrency = 4;
                    let got = scheduler_parse_slice(c, data).unwrap_or_else(|e| {
                        panic!("{backend:?} chunk_size={chunk_size} on {data:?}: {e}")
                    });
                    assert_eq!(
                        got, baseline,
                        "{backend:?} chunk_size={chunk_size} on {data:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn slice_matches_file_with_headers() {
        for &chunk_size in &[1, 4, 16] {
            let mut config = Config::default(); // has_headers = true
            config.chunk_size = chunk_size;
            config.concurrency = 4;
            assert_slice_matches_file(config, b"name,value\na,1\nb,2\nc,3\n");
        }
    }

    /// Inputs where a chunk can open at a field start whose field is
    /// quoted — the case `Literal` and `Strict` read differently from
    /// mid-field, and `find_record_start` sidesteps by scanning Toggle.
    const QUOTE_SEAM_CASES: &[&[u8]] = &[
        b"p,\"m\nn\"\nt,u\n",
        b"\"x\",\"y\nz\"\n\"w\",\"v\"\n",
        b"a,\"b\"\"c\"\nd,e\n",
        // Toggle and Literal disagree on where this record ends, so
        // merge's reparse has to catch it.
        b"q\"w,\"e\nf\"\nt,u\n",
        // `Strict` must still reject, at the single-chunk offset.
        b"pp,x\"y\nt,u\n",
    ];

    /// Chunking must not change what a dialect yields, at any boundary
    /// offset. Broad guard; the regression it grew out of is
    /// [`last_chunk_opening_on_a_quoted_field_start`].
    #[test]
    fn quote_handling_survives_every_chunk_boundary() {
        for qh in [
            QuoteHandling::Toggle,
            QuoteHandling::Literal,
            QuoteHandling::Strict,
        ] {
            for &data in QUOTE_SEAM_CASES {
                let parse = |chunk_size| {
                    let mut c = no_headers_config();
                    c.quote_handling = qh;
                    c.chunk_size = chunk_size;
                    c.concurrency = 4;
                    scheduler_parse_slice(c, data).map_err(|e| e.to_string())
                };
                let baseline = parse(1 << 20);
                for &chunk_size in &[1, 2, 3, 5, 8, 10, 16, 64] {
                    assert_eq!(
                        baseline,
                        parse(chunk_size),
                        "{qh:?} chunk_size={chunk_size} on {:?}",
                        String::from_utf8_lossy(data),
                    );
                }
            }
        }
    }

    /// `z,zz…z\n` + `tail`, sized so `tail` begins exactly at the last
    /// chunk boundary — no successor chunk to force merge's reparse, so a
    /// chunk that fails speculation loses its records outright.
    fn tail_at_last_boundary(chunk_size: usize, tail: &[u8]) -> Vec<u8> {
        assert!(chunk_size >= 3 && tail.len() <= chunk_size);
        let mut data = vec![b'z'; chunk_size];
        data[1] = b',';
        data[chunk_size - 1] = b'\n';
        data.extend_from_slice(tail);
        data
    }

    /// Regression: `Strict` made `InField` + quote a hard `Error`, so a
    /// last chunk opening on a quoted field start failed every assumption
    /// at both strict and lenient, and was skipped — losing `t,u`.
    #[test]
    fn last_chunk_opening_on_a_quoted_field_start() {
        const TAILS: &[&[u8]] = &[
            b"\"m\nn\"\nt,u\n",
            b"\"m\nn\"\n",
            b"\"m\",\"n\"\nt,u\n",
            b"q\"w,\"e\nf\"\n",
            b"x\"y\nt,u\n",
        ];
        for qh in [
            QuoteHandling::Toggle,
            QuoteHandling::Literal,
            QuoteHandling::Strict,
        ] {
            for &tail in TAILS {
                for &chunk_size in &[16, 24, 32, 64, 128] {
                    let data = tail_at_last_boundary(chunk_size, tail);
                    let parse = |cs| {
                        let mut c = no_headers_config();
                        c.quote_handling = qh;
                        c.chunk_size = cs;
                        c.concurrency = 4;
                        scheduler_parse_slice(c, &data).map_err(|e| e.to_string())
                    };
                    assert_eq!(
                        parse(1 << 20),
                        parse(chunk_size),
                        "{qh:?} chunk_size={chunk_size} on {:?}",
                        String::from_utf8_lossy(tail),
                    );
                }
            }
        }
    }

    /// With a distinct escape byte, quote parity alone would miscount
    /// `\\"`; the Toggle table keeps `InEscapedQuote`.
    #[test]
    fn distinct_escape_survives_every_chunk_boundary() {
        const TAIL: &[u8] = b"\"m\\\"n\nq\",r\nt,u\n";
        for qh in [
            QuoteHandling::Toggle,
            QuoteHandling::Literal,
            QuoteHandling::Strict,
        ] {
            for &chunk_size in &[24, 32, 64, 128] {
                let data = tail_at_last_boundary(chunk_size, TAIL);
                let parse = |cs| {
                    let mut c = no_headers_config();
                    c.quote_handling = qh;
                    c.escape = Some(b'\\');
                    c.chunk_size = cs;
                    c.concurrency = 4;
                    scheduler_parse_slice(c, &data).map_err(|e| e.to_string())
                };
                assert_eq!(
                    parse(1 << 20),
                    parse(chunk_size),
                    "{qh:?} chunk_size={chunk_size}"
                );
            }
        }
    }

    /// Only `Assumption::InQuotesAfterEscape` models a chunk opening on
    /// the byte after an escape: `OutOfQuotes` rejects, `InQuotes` scans
    /// clean off the end — which merge reads as "no records" and skips.
    #[test]
    fn escaped_escape_split_across_a_boundary() {
        for &chunk_size in &[16, 24, 32, 64, 128] {
            // `z,"ww…\` fills the first chunk exactly, so the last chunk
            // opens on the second byte of a `\\` pair inside the quote.
            let mut data = Vec::from(*b"z,\"");
            data.resize(chunk_size - 1, b'w');
            data.push(b'\\');
            assert_eq!(data.len(), chunk_size);
            data.extend_from_slice(b"\\\"\nt,u\n");
            let parse = |cs| {
                let mut c = no_headers_config();
                c.escape = Some(b'\\');
                c.chunk_size = cs;
                c.concurrency = 4;
                scheduler_parse_slice(c, &data).map_err(|e| e.to_string())
            };
            assert_eq!(parse(1 << 20), parse(chunk_size), "chunk_size={chunk_size}");
        }
    }

    #[test]
    fn slice_ignores_the_io_backend_setting() {
        // Every `IoBackend` names a way of reaching a file, so none of them
        // applies to a slice: all must fall through to `SliceIo` and parse
        // identically rather than erroring or diverging.
        //
        // Compared slice-against-slice, *not* against the file path. That
        // path honours the setting and rejects backends its platform lacks
        // — `Mmap` off unix — so it reports an error
        // exactly where the slice path is correct to succeed. Using it as
        // the oracle here would fail on Windows for `Mmap`, which is the
        // divergence this test exists to assert, not a defect.
        let run = |io_backend, data: &[u8]| {
            let mut config = no_headers_config();
            config.io_backend = io_backend;
            config.chunk_size = 4;
            config.concurrency = 4;
            scheduler_parse_slice(config, data).map_err(|e| e.to_string())
        };
        for &data in SEAM_CASES {
            // Auto is the baseline every explicit setting must reproduce.
            let baseline = run(IoBackend::Auto, data);
            for io_backend in [
                IoBackend::Mmap,
                IoBackend::RingBuf(RingBufSettings::default()),
                IoBackend::InMemory,
            ] {
                assert_eq!(
                    baseline,
                    run(io_backend, data),
                    "slice result changed under io_backend {io_backend:?} for {:?}",
                    String::from_utf8_lossy(data),
                );
            }
        }
    }

    #[test]
    fn slice_leaves_caller_data_untouched() {
        // Escape compaction rewrites the buffer it parses; the caller's
        // slice must not be that buffer.
        let data = b"a,\"b\"\"c\",d\ne,\"f\ng\",h\n".to_vec();
        let before = data.clone();
        let mut config = no_headers_config();
        config.chunk_size = 4;
        config.concurrency = 4;
        let records = scheduler_parse_slice(config, &data).unwrap();
        assert_eq!(records[0], vec!["a", "b\"c", "d"]);
        assert_eq!(data, before, "parse_slice mutated the caller's bytes");
    }

    #[test]
    fn slice_multi_chunk_long_record() {
        // One record far longer than a chunk: the reader must grow past the
        // nominal boundary to finish it.
        let long = "x".repeat(10_000);
        let data = format!("{long},{long}\nshort,row\n");
        let mut config = no_headers_config();
        config.chunk_size = 64;
        config.concurrency = 4;
        let records = scheduler_parse_slice(config, data.as_bytes()).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0], vec![long.clone(), long]);
        assert_eq!(records[1], vec!["short", "row"]);
    }

    #[test]
    fn scheduler_simple_csv() {
        run_matrix(no_headers_config(), |config| {
            let records = scheduler_parse(config, b"a,b,c\n1,2,3\n4,5,6\n").unwrap();
            assert_eq!(
                records,
                vec![
                    vec!["a", "b", "c"],
                    vec!["1", "2", "3"],
                    vec!["4", "5", "6"],
                ]
            );
        });
    }

    #[test]
    fn scheduler_empty_file() {
        let records = scheduler_parse(no_headers_config(), b"").unwrap();
        assert!(records.is_empty());
    }

    /// Regression: a chunk that opens mid-quoted-field and contains a
    /// balanced empty quoted field `""` scans cleanly under the
    /// `OutOfQuotes` assumption (no out-of-quotes terminator, and the strict
    /// probe stays quiet because `""`'s adjacent quotes look like valid
    /// boundaries). `try_assumptions` used to short-circuit on that
    /// `CleanNoBoundary` before trying `InQuotes`, which is the assumption
    /// that actually finds the record — so the record was dropped.
    ///
    /// `simd_diff` can't catch this: at concurrency 1 a small file is a
    /// single chunk, so chunk-boundary speculation never runs. Force a tiny
    /// `chunk_size` so the boundary lands inside the first record's quoted
    /// field (byte 0 of chunk 1 is the previous field's closing quote).
    #[cfg(feature = "simd")]
    #[test]
    fn simd_clean_no_boundary_does_not_drop_empty_quoted_record() {
        for (term, data) in [
            (
                RecordTerminator::CRLF,
                &b"aaaaaaaa,\"qqqqq\"\r\naa,\"\"\r\n"[..],
            ),
            (RecordTerminator::LF, &b"aaaaaaaa,\"qqqqq\"\naa,\"\"\n"[..]),
        ] {
            let mut config = no_headers_config();
            config.parser_backend = ParserBackend::Simd;
            config.field_count = Some(2);
            config.chunk_size = 15;
            config.terminator = term;
            let records = scheduler_parse(config, data).unwrap();
            assert_eq!(
                records,
                vec![vec!["aaaaaaaa", "qqqqq"], vec!["aa", ""]],
                "terminator {term:?}",
            );
        }
    }

    #[test]
    fn scheduler_single_record() {
        let records = scheduler_parse(no_headers_config(), b"hello,world\n").unwrap();
        assert_eq!(records, vec![vec!["hello", "world"]]);
    }

    #[test]
    fn scheduler_no_trailing_newline() {
        let records = scheduler_parse(no_headers_config(), b"a,b\n1,2").unwrap();
        assert_eq!(records, vec![vec!["a", "b"], vec!["1", "2"]]);
    }

    #[test]
    fn scheduler_quoted_fields() {
        let records = scheduler_parse(
            no_headers_config(),
            b"\"hello\",\"world\"\n\"foo\",\"bar\"\n",
        )
        .unwrap();
        assert_eq!(records, vec![vec!["hello", "world"], vec!["foo", "bar"]]);
    }

    #[test]
    fn scheduler_multi_chunk() {
        let mut base = no_headers_config();
        base.chunk_size = 8;
        base.concurrency = 2;

        run_matrix(base, |config| {
            let records = scheduler_parse(config, b"a,b\nc,d\ne,f\ng,h\n").unwrap();
            assert_eq!(
                records,
                vec![
                    vec!["a", "b"],
                    vec!["c", "d"],
                    vec!["e", "f"],
                    vec!["g", "h"],
                ]
            );
        });
    }

    #[test]
    fn scheduler_multi_chunk_quoted() {
        let mut base = no_headers_config();
        base.chunk_size = 10;
        base.concurrency = 4;

        run_matrix(base, |config| {
            let records =
                scheduler_parse(config, b"\"aa\",\"bb\"\n\"cc\",\"dd\"\n\"ee\",\"ff\"\n").unwrap();
            assert_eq!(
                records,
                vec![vec!["aa", "bb"], vec!["cc", "dd"], vec!["ee", "ff"],]
            );
        });
    }

    #[test]
    fn scheduler_record_spans_chunks() {
        let mut config = no_headers_config();
        config.chunk_size = 5;
        config.concurrency = 4;

        let records = scheduler_parse(config, b"abcdefghij,klmnopqrst\nx,y\n").unwrap();
        assert_eq!(
            records,
            vec![vec!["abcdefghij", "klmnopqrst"], vec!["x", "y"],]
        );
    }

    #[test]
    fn scheduler_headers_skipped() {
        let records = scheduler_parse(
            Config::default(), // has_headers = true
            b"name,value\na,1\nb,2\n",
        )
        .unwrap();
        assert_eq!(records, vec![vec!["a", "1"], vec!["b", "2"]]);
    }

    #[test]
    fn scheduler_headers_skipped_with_comments() {
        let mut config = Config::default(); // has_headers = true
        config.comment = Some(b'#');
        let records = scheduler_parse(
            config,
            b"# comment\n# another\nname,value\n# mid\na,1\nb,2\n",
        )
        .unwrap();
        assert_eq!(records, vec![vec!["a", "1"], vec!["b", "2"]]);
    }

    #[test]
    fn scheduler_ringbuf_backend() {
        let mut config = no_headers_config();
        config.io_backend = IoBackend::RingBuf(RingBufSettings::default());
        config.chunk_size = 8;
        config.concurrency = 2;

        let records = scheduler_parse(config, b"a,b\nc,d\ne,f\ng,h\n").unwrap();
        assert_eq!(
            records,
            vec![
                vec!["a", "b"],
                vec!["c", "d"],
                vec!["e", "f"],
                vec!["g", "h"],
            ]
        );
    }

    #[test]
    fn scheduler_in_memory_backend() {
        let mut config = no_headers_config();
        config.io_backend = IoBackend::InMemory;
        config.chunk_size = 8;
        config.concurrency = 2;

        let records = scheduler_parse(config, b"a,b\nc,d\ne,f\ng,h\n").unwrap();
        assert_eq!(
            records,
            vec![
                vec!["a", "b"],
                vec!["c", "d"],
                vec!["e", "f"],
                vec!["g", "h"],
            ]
        );
    }

    #[test]
    fn the_in_memory_backend_does_not_enforce_the_memory_limit() {
        let mut config = no_headers_config();
        config.io_backend = IoBackend::InMemory;
        config.io_buffer_limit = Some(8);

        let records = scheduler_parse(config, b"a,b\nc,d\ne,f\ng,h\n").unwrap();
        assert_eq!(records.len(), 4);
        assert_eq!(records[0], vec!["a", "b"]);
    }

    #[test]
    fn a_tight_limit_still_parses_under_auto() {
        let granularity = crate::sys::allocation_granularity();
        let mut data = Vec::new();
        while data.len() < granularity * 2 {
            data.extend_from_slice(b"aaaa,bbbb\n");
        }
        let expected_rows = data.len() / 10;

        let mut config = no_headers_config();
        config.io_backend = IoBackend::Auto;
        config.io_buffer_limit = Some(granularity);
        assert!(data.len() > granularity);

        let records = scheduler_parse(config, &data).unwrap();
        assert_eq!(records.len(), expected_rows);
        assert_eq!(records[0], vec!["aaaa", "bbbb"]);
    }

    #[test]
    fn scheduler_trim_fields() {
        let mut config = no_headers_config();
        config.trim = true;

        let records = scheduler_parse(config, b"  hello , world \n\tfoo\t,  bar  \n").unwrap();
        assert_eq!(records, vec![vec!["hello", "world"], vec!["foo", "bar"],]);
    }

    #[test]
    fn scheduler_error_position_is_absolute() {
        let mut base = no_headers_config();
        base.chunk_size = 8;
        base.concurrency = 2;

        run_matrix(base, |config| {
            let err = scheduler_parse(config, b"aa,bb\ncc,dd\n\"unclosed\n").unwrap_err();
            match err {
                crate::Error::UnclosedQuote { position } => {
                    assert_eq!(position.byte_offset, 12);
                }
                e => panic!("expected UnclosedQuote, got {e:?}"),
            }
        });
    }

    /// Run the streaming entry point against an in-memory byte slice.
    fn stream_parse(config: Config, csv_data: &[u8]) -> crate::Result<Vec<Vec<String>>> {
        let scheduler = Scheduler::new(config);
        scheduler.run_stream(
            std::io::Cursor::new(csv_data.to_vec()),
            Vec::<Vec<String>>::new,
            |state: &mut Vec<Vec<String>>, fields: &mut [&mut str]| {
                state.push(fields.iter().map(|s| s.to_string()).collect());
                Ok(())
            },
            |states| {
                let mut all = Vec::new();
                for s in states {
                    all.extend(std::mem::take(s));
                }
                all
            },
        )
    }

    #[test]
    fn stream_simple_csv() {
        let records = stream_parse(no_headers_config(), b"a,b,c\n1,2,3\n4,5,6\n").unwrap();
        assert_eq!(
            records,
            vec![
                vec!["a", "b", "c"],
                vec!["1", "2", "3"],
                vec!["4", "5", "6"],
            ]
        );
    }

    #[test]
    fn stream_skips_headers() {
        let records = stream_parse(Config::default(), b"name,value\na,1\nb,2\n").unwrap();
        assert_eq!(records, vec![vec!["a", "1"], vec!["b", "2"]]);
    }

    #[test]
    fn stream_quoted_with_embedded_newline() {
        let records = stream_parse(no_headers_config(), b"\"a\nb\",\"c\"\n\"d\",\"e\"\n").unwrap();
        assert_eq!(records, vec![vec!["a\nb", "c"], vec!["d", "e"]]);
    }

    #[test]
    fn stream_no_trailing_newline() {
        let records = stream_parse(no_headers_config(), b"a,b\n1,2").unwrap();
        assert_eq!(records, vec![vec!["a", "b"], vec!["1", "2"]]);
    }

    #[test]
    fn stream_empty_input() {
        let records = stream_parse(no_headers_config(), b"").unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn stream_growing_record() {
        // A single record longer than the default io_buffer_size (16 KiB)
        // forces RingBuf::grow during streaming.
        let mut field = String::new();
        for i in 0..10000 {
            if i > 0 {
                field.push(',');
            }
            field.push_str(&format!("v{}", i));
        }
        let mut csv = field.clone();
        csv.push('\n');
        let records = stream_parse(no_headers_config(), csv.as_bytes()).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].len(), 10000);
    }

    #[test]
    fn stream_unclosed_quote_errors() {
        let err = stream_parse(no_headers_config(), b"a,b\n\"unclosed\n").unwrap_err();
        assert!(matches!(err, crate::Error::UnclosedQuote { .. }));
    }
}
