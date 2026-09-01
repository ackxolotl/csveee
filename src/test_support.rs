//! Helpers for unit tests that exercise parser logic across multiple
//! backend configurations.
//!
//! Set `CSVEEE_TEST_MATRIX=full` to expand each `run_matrix` call into
//! the cartesian product of I/O backends and parser backends, catching
//! backend-specific divergence (e.g. at chunk/page boundaries) that the
//! `Auto` heuristic otherwise hides.

use crate::config::{Config, IoBackend, ParserBackend};
use crate::io::ringbuf::RingBufSettings;

/// Corpus-path resolution.
#[path = "../tests/common/data.rs"]
pub(crate) mod data;

fn full_matrix() -> bool {
    std::env::var("CSVEEE_TEST_MATRIX").as_deref() == Ok("full")
}

/// The I/O backends the full matrix compares against each other.
fn io_backends() -> Vec<IoBackend> {
    vec![
        IoBackend::Mmap,
        IoBackend::RingBuf(RingBufSettings::default()),
        IoBackend::InMemory,
    ]
}

fn matrix_cases(base: Config) -> Vec<Config> {
    if !full_matrix() {
        return vec![base];
    }
    let mut cases = Vec::new();
    for io in io_backends() {
        #[allow(clippy::single_element_loop)]
        for parser in [ParserBackend::Dfa] {
            let mut cfg = base.clone();
            cfg.io_backend = io;
            cfg.parser_backend = parser;
            cases.push(cfg);
        }
    }
    cases
}

/// Run `body` once per configuration in the active matrix.
pub(crate) fn run_matrix<F>(base: Config, body: F)
where
    F: Fn(Config),
{
    for cfg in matrix_cases(base) {
        let label = format!(
            "io={:?} parser={:?} chunk_size={}",
            cfg.io_backend, cfg.parser_backend, cfg.chunk_size,
        );
        let cfg_for_call = cfg.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(cfg_for_call)));
        if let Err(payload) = result {
            eprintln!("matrix case failed: {label}");
            std::panic::resume_unwind(payload);
        }
    }
}
