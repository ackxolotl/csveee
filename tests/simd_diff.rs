//! Differential test: SIMD parser vs DFA parser, byte-equivalent rows.
//!
//! For every dialect in the test corpus that the SIMD backend supports,
//! parse the file twice with `QuoteHandling::Toggle` (the only style SIMD
//! supports), once via DFA and once via SIMD, and assert identical row
//! digests. The DFA backend is treated as oracle here — its compatibility
//! with rust-csv is enforced separately by `tests/integration.rs`, which
//! uses `QuoteHandling::Literal` (rust-csv's semantics).

mod common;

use csveee::{ParserBackend, ParserBuilder, QuoteHandling, RecordTerminator};
use libtest_mimic::{Arguments, Failed, Trial};

use self::common::{
    Dialect, PARSER_THREADS, RowDigest, TestEntry, compare_digests, load_suite, suites,
    test_threads, too_big_to_test,
};

const SUITES: &[&str] = &["duckdb", "fixtures", "kaggle", "postgres", "rust-csv"];

/// Return `Some(reason)` describing why the SIMD backend cannot run
/// this dialect, or `None` if it can.
///
/// Test-level checks (Arity column-count cap, flexible field count)
/// stay here because they aren't visible to the chunk parser's
/// `supports`. Everything else delegates to
/// `ParserBuilder::supports_backend(Simd)`, which lets the bench and
/// this test share the same predicate without re-implementing it.
fn simd_skip_reason(entry: &TestEntry) -> Option<&'static str> {
    if entry.flexible {
        return Some("SIMD does not support flexible field count");
    }
    // Two different ceilings meet here: Arity dispatch covers 1..=64 columns,
    // but the SIMD chunk parser handles at most 63 fields per record, and
    // `supports_backend` cannot see a column count to check it against. The
    // tighter one wins — a 64-column file is a build error, not a divergence.
    if entry.columns == 0 || entry.columns > 63 {
        return Some("SIMD covers 1..=63 fields per record");
    }
    build_parser(&entry.dialect, ParserBackend::Simd)
        .supports_backend(ParserBackend::Simd)
        .err()
}

/// Build a CSV-eee parser configured per `dialect` with the given
/// backend. Always uses `QuoteHandling::Toggle` so DFA and SIMD
/// produce semantically identical output.
fn build_parser(dialect: &Dialect, backend: ParserBackend) -> ParserBuilder {
    let mut b = ParserBuilder::new()
        .quote_handling(QuoteHandling::Toggle)
        .parser_backend(backend)
        .concurrency(PARSER_THREADS);

    if let Some(d) = dialect.delimiter {
        b = b.delimiter(d);
    }
    match dialect.terminator.as_deref() {
        Some("LF") => b = b.terminator(RecordTerminator::LF),
        Some("CR") => b = b.terminator(RecordTerminator::CR),
        // `simd_skip_reason` filters CRLF via `supports_backend`, but
        // `simd_skip_reason` itself calls this helper to perform the
        // probe — so we accept CRLF here and let the SIMD support
        // check do the rejecting.
        Some("CRLF") | None => b = b.terminator(RecordTerminator::CRLF),
        Some(other) => panic!("unsupported terminator: {other}"),
    }
    match dialect.quote.as_deref() {
        Some("") => b = b.quote(None),
        Some(s) if s.len() == 1 => b = b.quote(Some(s.as_bytes()[0])),
        None => {}
        Some(other) => panic!("unsupported quote: {other:?}"),
    }
    match dialect.escape.as_deref() {
        Some("") => b = b.escape(None),
        Some(s) if s.len() == 1 => b = b.escape(Some(s.as_bytes()[0])),
        None => {}
        Some(other) => panic!("unsupported escape: {other:?}"),
    }
    if let Some(h) = dialect.has_headers {
        b = b.has_headers(h);
    }
    // Apply the comment char so commented files are recognised as such:
    // `supports()` rejects `comment.is_some()`, so `simd_skip_reason`
    // routes them to the DFA and they drop out of the SIMD differential
    // entirely. Without this they reach the SIMD parser as ragged plain
    // CSV (`#` lines treated as data) — which it must never handle.
    if let Some(c) = dialect.comment {
        b = b.comment(Some(c));
    }
    b
}

/// Streaming Arity-Text digest for `path` parsed under `backend`.
fn digest_arity(
    dialect: &Dialect,
    backend: ParserBackend,
    path: &str,
    columns: usize,
) -> csveee::Result<RowDigest> {
    macro_rules! dispatch {
        ($($n:literal),+) => {
            match columns {
                $($n => {
                    let mut parser = build_parser(dialect, backend).build();
                    parser.parse::<_, _, _, _, _, _, $n>(
                        path,
                        RowDigest::new,
                        |d: &mut RowDigest, fields: [&mut str; $n]| {
                            d.add_text_row(fields.as_slice());
                            Ok(())
                        },
                        merge_digests,
                    )
                },)+
                _ => panic!("Arity not supported for {columns} columns"),
            }
        };
    }
    dispatch!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
    )
}

/// Streaming Arity-Bytes digest for `path` parsed under `backend`.
/// Used for non-UTF-8 files where `Text` would fail validation.
fn digest_bytes_arity(
    dialect: &Dialect,
    backend: ParserBackend,
    path: &str,
    columns: usize,
) -> csveee::Result<RowDigest> {
    macro_rules! dispatch {
        ($($n:literal),+) => {
            match columns {
                $($n => {
                    let mut parser = build_parser(dialect, backend).bytes().build();
                    parser.parse::<_, _, _, _, _, _, $n>(
                        path,
                        RowDigest::new,
                        |d: &mut RowDigest, fields: [&mut [u8]; $n]| {
                            d.add_byte_row(fields.as_slice());
                            Ok(())
                        },
                        merge_digests,
                    )
                },)+
                _ => panic!("Arity not supported for {columns} columns"),
            }
        };
    }
    dispatch!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
    )
}

fn merge_digests(digests: &mut [RowDigest]) -> RowDigest {
    let mut result = RowDigest::new();
    for d in digests {
        result.count += d.count;
        result.hash = result.hash.wrapping_add(d.hash);
    }
    result
}

fn run_diff(entry: &TestEntry) -> Result<(), String> {
    if !entry.abs_path.exists() {
        return Err(format!("CSV file not found: {}", entry.abs_path.display()));
    }
    if entry.expect_error.is_some() {
        // Expected-error cases are dialect-driven and not the focus of
        // this differential — skip rather than risk false positives
        // when DFA and SIMD signal errors at slightly different points.
        return Ok(());
    }
    if entry.simd_skip {
        // SIMD intentionally diverges (emits N=1 blank lines, errors on
        // ragged/empty rows) rather than skipping them like the DFA.
        return Ok(());
    }

    let path = entry.abs_path.to_str().unwrap();
    let is_utf8 = entry.encoding == "utf-8";

    if is_utf8 {
        let dfa = digest_arity(&entry.dialect, ParserBackend::Dfa, path, entry.columns);
        let simd = digest_arity(&entry.dialect, ParserBackend::Simd, path, entry.columns);
        compare_results("text", &simd, &dfa)?;
    }

    let dfa_b = digest_bytes_arity(&entry.dialect, ParserBackend::Dfa, path, entry.columns);
    let simd_b = digest_bytes_arity(&entry.dialect, ParserBackend::Simd, path, entry.columns);
    compare_results("bytes", &simd_b, &dfa_b)?;

    Ok(())
}

/// Compare DFA and SIMD outcomes. A backend may legitimately reject
/// malformed input under `QuoteHandling::Toggle` (e.g. unterminated quote
/// at EOF) — the differential passes as long as both backends agree on
/// success-or-rejection. When both succeed, row digests must match;
/// when both fail, the error kinds must match.
fn compare_results(
    label: &str,
    simd: &csveee::Result<RowDigest>,
    dfa: &csveee::Result<RowDigest>,
) -> Result<(), String> {
    match (simd, dfa) {
        (Ok(s), Ok(d)) => compare_digests(label, s, d),
        (Err(s), Err(d)) => {
            let same_kind = std::mem::discriminant(s) == std::mem::discriminant(d);
            if same_kind {
                Ok(())
            } else {
                Err(format!(
                    "{label}: error kind mismatch — SIMD: {s}, DFA: {d}"
                ))
            }
        }
        (Ok(_), Err(d)) => Err(format!("{label}: DFA rejected ({d}) but SIMD accepted")),
        (Err(s), Ok(_)) => Err(format!("{label}: SIMD rejected ({s}) but DFA accepted")),
    }
}

fn main() {
    let mut args = Arguments::from_args();

    // Each parse is pinned to `PARSER_THREADS` rather than expanding to
    // `available_parallelism()`, so files no longer have to run one at a
    // time to avoid oversubscribing the machine — `cores / PARSER_THREADS`
    // of them fit at once. A CLI --test-threads override is respected.
    if args.test_threads.is_none() {
        args.test_threads = Some(test_threads());
    }

    let tests: Vec<Trial> = suites(SUITES, args.filter.as_deref())
        .into_iter()
        .flat_map(|suite| {
            load_suite(&suite).into_iter().map(move |entry| {
                let name = format!("simd-vs-dfa::{}::{}", suite.name, entry.rel_path);
                let skip_reason = if entry.skip {
                    entry.skip_reason.clone()
                } else if !entry.abs_path.exists() {
                    Some("file missing".to_string())
                } else if too_big_to_test(&entry) {
                    Some("large file; set CSVEEE_TEST_BIG=1".to_string())
                } else {
                    simd_skip_reason(&entry).map(|s| s.to_string())
                };
                let trial = Trial::test(name, move || run_diff(&entry).map_err(Failed::from));
                if skip_reason.is_some() {
                    trial.with_ignored_flag(true)
                } else {
                    trial
                }
            })
        })
        .collect();

    libtest_mimic::run(&args, tests).exit();
}
