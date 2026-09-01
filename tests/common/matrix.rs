//! Backend-matrix and thread budget for the test harnesses.
//!
//! Set `CSVEEE_TEST_MATRIX=full` to run each test entry across the
//! cartesian product of I/O backends and parser backends. Failures
//! are labelled with the offending case so the matrix doesn't hide
//! which configuration broke.

use csveee::{IoBackend, ParserBackend, ParserBuilder, RingBufSettings};

/// Worker threads each test's parser is pinned to.
pub const PARSER_THREADS: usize = 4;

/// Deliberate overcommit of trial concurrency.
const OVERCOMMIT: usize = 2;

/// Trials to run at once: the machine divided by what one trial averages.
pub fn test_threads() -> usize {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    (cores * OVERCOMMIT / PARSER_THREADS).max(4)
}

#[derive(Clone, Copy, Debug)]
pub struct BackendCase {
    pub io: IoBackend,
    pub parser: ParserBackend,
}

impl BackendCase {
    pub const fn auto() -> Self {
        Self {
            io: IoBackend::Auto,
            parser: ParserBackend::Auto,
        }
    }

    pub fn label(self) -> String {
        format!("io={:?} parser={:?}", self.io, self.parser)
    }

    /// Configure a parser for this case. Every parser the integration
    /// harness builds goes through here, so this is also where the thread
    /// budget is applied. Benchmarks deliberately don't — they want the
    /// machine to themselves.
    pub fn apply<M, E: ?Sized>(self, builder: ParserBuilder<M, E>) -> ParserBuilder<M, E> {
        builder
            .io_backend(self.io)
            .parser_backend(self.parser)
            .concurrency(PARSER_THREADS)
    }
}

fn full_matrix() -> bool {
    std::env::var("CSVEEE_TEST_MATRIX").as_deref() == Ok("full")
}

/// Cases to run for the active matrix mode.
///
/// Default mode: a single `Auto`/`Auto` case — equivalent to the
/// pre-matrix behaviour, keeping the suite fast.
///
/// Full mode: the cartesian product of concrete backends.
pub fn matrix_cases() -> Vec<BackendCase> {
    if !full_matrix() {
        return vec![BackendCase::auto()];
    }
    let ios = [
        IoBackend::Mmap,
        IoBackend::RingBuf(RingBufSettings::default()),
    ];

    let mut cases = Vec::new();
    for io in ios {
        #[allow(clippy::single_element_loop)]
        for parser in [ParserBackend::Dfa] {
            cases.push(BackendCase { io, parser });
        }
    }
    cases
}
