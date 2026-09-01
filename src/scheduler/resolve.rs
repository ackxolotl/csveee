//! Turning a user [`Config`] into a [`ResolvedConfig`].
//!
//! Everything here is policy: given the user's settings plus what we can
//! observe about the input (size, host parallelism), decide which backends
//! to use and how big the work units are. Resolution order matters a lot
//! here.

use crate::config::{Config, IoBackend, ParserBackend};
use crate::io::ringbuf::RingBufSettings;
use crate::parser::dfa::DfaChunkParser;
#[cfg(feature = "simd")]
use crate::parser::simd::SimdChunkParser;
use crate::parser::{ChunkParser, FindRecordStart};

/// Fully-resolved scheduler configuration.
#[derive(Debug)]
pub(super) struct ResolvedConfig {
    pub config: Config,
    pub parser_backend: ParserBackend,
    pub io_backend: IoBackend,
    pub thread_count: usize,
    pub chunk_count: usize,
    pub file_size: usize,
}

impl ResolvedConfig {
    /// Resolve auto sentinels into concrete values for a file input.
    pub(super) fn for_file(config: &Config, file_size: usize) -> crate::Result<Self> {
        let parser_backend = resolve_parser_backend(config)?;

        let chunk_size = resolve_chunk_size(config, parser_backend);
        let chunk_count = file_size.div_ceil(chunk_size);
        let io_buffer_limit = resolve_io_buffer_limit(config);
        let thread_count = resolve_thread_count(config, chunk_count, io_buffer_limit);
        let share = per_thread_share(io_buffer_limit, thread_count);
        let io_backend = resolve_io_backend(config, file_size, share);

        let mut config = config.clone();
        config.chunk_size = chunk_size;
        config.io_buffer_limit = io_buffer_limit;
        Ok(ResolvedConfig {
            config,
            parser_backend,
            io_backend,
            thread_count,
            chunk_count,
            file_size,
        })
    }

    /// [`ResolvedConfig::for_file`] for a slice input.
    pub(super) fn for_slice(config: &Config, len: usize) -> crate::Result<Self> {
        let parser_backend = resolve_parser_backend(config)?;

        let chunk_size = resolve_chunk_size(config, parser_backend);
        let chunk_count = len.div_ceil(chunk_size);
        let io_buffer_limit = resolve_io_buffer_limit(config);
        let thread_count = resolve_thread_count(config, chunk_count, io_buffer_limit);
        let share = per_thread_share(io_buffer_limit, thread_count);

        let mut config = config.clone();
        config.chunk_size = chunk_size;
        config.io_buffer_limit = io_buffer_limit;
        Ok(ResolvedConfig {
            config,
            parser_backend,
            io_backend: IoBackend::RingBuf(unbacked_ring_settings(share)),
            thread_count,
            chunk_count,
            file_size: len,
        })
    }

    /// Determine how `find_record_start` should behave for a given chunk.
    pub(super) fn find_mode(&self, chunk_idx: usize) -> FindRecordStart {
        if chunk_idx == 0 {
            if self.config.has_headers {
                FindRecordStart::SkipHeaders
            } else {
                FindRecordStart::No
            }
        } else {
            FindRecordStart::Strict
        }
    }
}

/// Size below which `Auto` prefers the in-memory backend over mmap.
const IN_MEMORY_FILE_SIZE_THRESHOLD: usize = 1024 * 1024;

/// Determine the number of threads to use.
fn resolve_thread_count(
    config: &Config,
    chunk_count: usize,
    io_buffer_limit: Option<usize>,
) -> usize {
    // concurrency == 0 is the sentinel for "auto-detect now".
    let h = if config.concurrency == 0 {
        crate::sys::available_parallelism()
    } else {
        config.concurrency
    };
    let c = chunk_count;
    let sqrt_c = (c as f64).sqrt().ceil() as usize;
    let b = affordable_thread_count(io_buffer_limit);
    h.min(c).min(6 * sqrt_c).min(b)
}

/// How many workers the I/O buffer budget can pay for.
fn affordable_thread_count(io_buffer_limit: Option<usize>) -> usize {
    match io_buffer_limit {
        None => usize::MAX,
        Some(budget) => (budget / min_thread_buffer()).max(1),
    }
}

/// Smallest buffer a worker can be given and still do useful work.
fn min_thread_buffer() -> usize {
    crate::sys::allocation_granularity()
}

/// Select the I/O backend, resolving `Auto` from file size.
fn resolve_io_backend(config: &Config, file_size: usize, share: Option<usize>) -> IoBackend {
    let selected = match config.io_backend {
        IoBackend::Auto => {
            if file_size <= IN_MEMORY_FILE_SIZE_THRESHOLD {
                IoBackend::InMemory
            } else {
                IoBackend::RingBuf(RingBufSettings::default())
            }
        }
        other => other,
    };
    match selected {
        IoBackend::RingBuf(ring) => IoBackend::RingBuf(resolve_ringbuf_settings(ring, share)),
        other => other,
    }
}

/// Turn caller-stated [`RingBufSettings`] into the concrete values the
/// ring buffer is built from.
fn resolve_ringbuf_settings(ring: RingBufSettings, share: Option<usize>) -> RingBufSettings {
    RingBufSettings {
        buffer_size: ring.buffer_size,
        buffer_limit: resolve_buffer_limit(ring.buffer_limit, share),
    }
}

/// The cap one worker's buffer actually gets.
fn resolve_buffer_limit(own: Option<usize>, share: Option<usize>) -> Option<usize> {
    match own {
        None | Some(0) => share,
        Some(own) => match share {
            Some(share) => Some(own.min(share)),
            None => Some(own),
        },
    }
}

/// Ring-buffer settings for the paths that buffer through a ring buffer
/// without selecting a backend: `parse_slice` and `parse_stream`.
fn unbacked_ring_settings(share: Option<usize>) -> RingBufSettings {
    resolve_ringbuf_settings(RingBufSettings::default(), share)
}

/// Resolve the chunk size, honoring the auto-detect sentinel.
fn resolve_chunk_size(config: &Config, parser_backend: ParserBackend) -> usize {
    if config.chunk_size != 0 {
        return config.chunk_size;
    }
    match parser_backend {
        ParserBackend::Dfa => 256 * 1024,
        #[cfg(feature = "simd")]
        ParserBackend::Simd => 1024 * 1024,
        #[cfg(not(feature = "simd"))]
        ParserBackend::Simd => unreachable!("SIMD backend not compiled in"),
        ParserBackend::Auto => unreachable!("called after resolve_parser_backend"),
    }
}

/// Resolve the *total* I/O buffer budget, honoring the auto sentinel.
fn resolve_io_buffer_limit(config: &Config) -> Option<usize> {
    match config.io_buffer_limit {
        Some(0) => Some(crate::sys::total_ram()? / 2),
        other => other,
    }
}

/// Ring-buffer settings for the streaming path.
pub(super) fn stream_ring_settings(config: &Config) -> RingBufSettings {
    let io_buffer_limit = resolve_io_buffer_limit(config);
    let share = per_thread_share(io_buffer_limit, 1);
    unbacked_ring_settings(share)
}

/// Divide the total limit into one worker's share.
fn per_thread_share(io_buffer_limit: Option<usize>, thread_count: usize) -> Option<usize> {
    let total = io_buffer_limit?;
    if thread_count == 0 {
        return None;
    }
    Some(total / thread_count)
}

/// Select the chunk parser backend, resolving `Auto`.
pub(super) fn resolve_parser_backend(config: &Config) -> crate::Result<ParserBackend> {
    match config.parser_backend {
        ParserBackend::Auto => {
            #[cfg(feature = "simd")]
            if SimdChunkParser::supports(config).is_ok() {
                return Ok(ParserBackend::Simd);
            }
            DfaChunkParser::supports(config).map_err(crate::Error::InvalidConfig)?;
            Ok(ParserBackend::Dfa)
        }
        ParserBackend::Dfa => {
            DfaChunkParser::supports(config).map_err(crate::Error::InvalidConfig)?;
            Ok(ParserBackend::Dfa)
        }
        ParserBackend::Simd => {
            #[cfg(feature = "simd")]
            {
                SimdChunkParser::supports(config).map_err(crate::Error::InvalidConfig)?;
                Ok(ParserBackend::Simd)
            }
            #[cfg(not(feature = "simd"))]
            {
                Err(crate::Error::InvalidConfig(
                    "SIMD parser not compiled in (build with `--features simd` on nightly)",
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_count_capped_by_chunks() {
        let config = Config::default();
        assert!(resolve_thread_count(&config, 2, None) <= 2);
    }

    #[test]
    fn thread_count_capped_by_the_buffer_limit() {
        let mut config = Config::default();
        config.concurrency = 64;
        let floor = min_thread_buffer();

        // A limit covering four worker buffers funds at most four
        // workers, however many chunks and cores are on offer.
        let threads = resolve_thread_count(&config, 1024, Some(floor * 4));
        assert_eq!(threads, 4);
        // ...and each of those four gets its full share, not less.
        assert_eq!(per_thread_share(Some(floor * 4), threads), Some(floor));

        // An unbounded budget leaves the other terms in charge.
        assert_eq!(resolve_thread_count(&config, 1024, None), 64);
    }

    #[test]
    fn a_limit_below_one_buffer_still_runs_one_thread() {
        let mut config = Config::default();
        config.concurrency = 8;
        let floor = min_thread_buffer();

        // Too small for even a single worker: we do not round the share
        // up to the floor (that would overshoot the caller's total), and
        // we do not resolve to zero threads. The backend rejects it with
        // a precise message instead.
        let threads = resolve_thread_count(&config, 1024, Some(floor / 4));
        assert_eq!(threads, 1);
        assert_eq!(per_thread_share(Some(floor / 4), threads), Some(floor / 4));
    }

    #[test]
    fn buffer_limit_is_split_across_workers() {
        // The contract users see: state a total, each worker gets 1/T.
        assert_eq!(per_thread_share(Some(64 * 1024), 4), Some(16 * 1024));
        assert_eq!(per_thread_share(Some(64 * 1024), 1), Some(64 * 1024));
        // Unbounded stays unbounded.
        assert_eq!(per_thread_share(None, 8), None);
    }

    #[test]
    fn the_thread_cap_ignores_the_ring_buffers_working_size() {
        // The floor is backend-independent on purpose: it has to be
        // computable before `Auto` has picked a backend, so no backend's
        // settings may feed into it. A huge ring-buffer working size is
        // that backend's own problem — `RingBuf::new` rejects it by name
        // — and must not silently change how many threads we run.
        let mut config = Config::default();
        config.concurrency = 8;
        let limit = Some(min_thread_buffer() * 8);

        let baseline = resolve_thread_count(&config, 1024, limit);
        config.io_backend =
            IoBackend::RingBuf(RingBufSettings::default().buffer_size(64 * 1024 * 1024));
        assert_eq!(resolve_thread_count(&config, 1024, limit), baseline);
    }

    #[test]
    fn auto_limit_is_half_of_ram() {
        let mut config = Config::default();
        config.io_buffer_limit = Some(0);
        let budget = resolve_io_buffer_limit(&config);

        match crate::sys::total_ram() {
            // Hosts that do not report RAM fall back to unbounded.
            None => assert_eq!(budget, None),
            Some(total_ram) => assert_eq!(budget, Some(total_ram / 2)),
        }

        // Explicit totals and `None` pass through as stated.
        config.io_buffer_limit = Some(4096);
        assert_eq!(resolve_io_buffer_limit(&config), Some(4096));
        config.io_buffer_limit = None;
        assert_eq!(resolve_io_buffer_limit(&config), None);
    }

    #[test]
    fn chunk_count_via_resolve() {
        // for_file picks chunk_size = 256 KiB for DFA, so an empty file
        // yields zero chunks regardless of input size 0.
        let r = ResolvedConfig::for_file(&Config::default(), 0).unwrap();
        assert_eq!(r.chunk_count, 0);
        assert_ne!(r.config.chunk_size, 0, "auto sentinel must be resolved");

        let mut config = Config::default();
        config.chunk_size = 100;
        assert_eq!(
            ResolvedConfig::for_file(&config, 200).unwrap().chunk_count,
            2
        );
        assert_eq!(
            ResolvedConfig::for_file(&config, 250).unwrap().chunk_count,
            3
        );
    }

    #[test]
    fn auto_backend_by_file_size() {
        let config = Config::default();
        let threshold = IN_MEMORY_FILE_SIZE_THRESHOLD;
        let backend = |fsize| resolve_io_backend(&config, fsize, None);
        // Resolution consumes the `Some(0)` sentinel: no parser limit to
        // share out leaves the ring buffer unbounded.
        let ring = IoBackend::RingBuf(resolve_ringbuf_settings(RingBufSettings::default(), None));

        // Small files → InMemory.
        assert_eq!(backend(1), IoBackend::InMemory);
        assert_eq!(backend(threshold), IoBackend::InMemory);
        // Larger files → RingBuf: it matches or beats mmap across
        // dialects, and avoids mmap's COW-fault penalty when escape
        // compaction writes to the buffer (see resolve_io_backend).
        assert_eq!(backend(threshold + 1), ring);
    }

    #[test]
    fn auto_backend_ignores_the_memory_limit() {
        // Selection is a function of file size alone. `Auto` only
        // reaches for InMemory below the threshold, and thread_count is
        // capped by chunk_count, so the exposure it can create is a few
        // MiB — not worth letting a memory setting steer the choice.
        let config = Config::default();
        let tiny = Some(16 * 1024);
        assert_eq!(
            resolve_io_backend(&config, 64 * 1024, tiny),
            IoBackend::InMemory
        );
    }

    #[test]
    fn the_tighter_cap_wins() {
        // Whichever side is tighter, in either direction.
        assert_eq!(resolve_buffer_limit(Some(4096), Some(8192)), Some(4096));
        assert_eq!(resolve_buffer_limit(Some(8192), Some(4096)), Some(4096));
        // Only the backend has an opinion: it stands alone.
        assert_eq!(resolve_buffer_limit(Some(4096), None), Some(4096));
        // Neither side bounds anything: unbounded.
        assert_eq!(resolve_buffer_limit(Some(0), None), None);
    }

    #[test]
    fn no_opinion_takes_the_share_whichever_way_it_is_spelled() {
        // `Some(0)` is the auto sentinel and `None` is treated the same,
        // so neither can widen past the parser-level limit — a
        // per-thread setting only ever tightens.
        for no_opinion in [Some(0), None] {
            assert_eq!(resolve_buffer_limit(no_opinion, Some(1024)), Some(1024));
        }
    }

    #[test]
    fn resolution_consumes_the_sentinel_and_leaves_the_working_size() {
        let stated = RingBufSettings::default().buffer_size(4096);
        assert_eq!(stated.buffer_limit, Some(0), "default is the sentinel");

        let resolved = resolve_ringbuf_settings(stated, Some(1024));
        assert_eq!(resolved.buffer_size, 4096, "working size passes through");
        assert_eq!(resolved.buffer_limit, Some(1024));
    }

    #[test]
    fn the_share_reaches_the_selected_ring_buffer() {
        // The point of the payload: what comes out of resolution is a
        // backend carrying an effective per-thread cap, not a total that
        // something downstream has to remember to divide.
        let config = Config::default();
        let resolved = resolve_io_backend(&config, IN_MEMORY_FILE_SIZE_THRESHOLD + 1, Some(4096));
        assert_eq!(
            resolved,
            IoBackend::RingBuf(RingBufSettings::default().buffer_limit(Some(4096)))
        );

        // A caller's own per-thread bound composes with it as a min.
        let mut config = Config::default();
        config.io_backend = IoBackend::RingBuf(RingBufSettings::default().buffer_limit(Some(1024)));
        let resolved = resolve_io_backend(&config, 1, Some(4096));
        assert_eq!(
            resolved,
            IoBackend::RingBuf(RingBufSettings::default().buffer_limit(Some(1024)))
        );
    }
}
