//! Shared test-suite infrastructure used by both integration tests and
//! benchmarks. Keep this file free of test-only assertion code so it stays
//! usable from `benches/*`.

#![allow(dead_code)]

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use csv::Terminator;
use serde::Deserialize;

#[path = "data.rs"]
pub mod data;
#[path = "matrix.rs"]
pub mod matrix;
use self::matrix::BackendCase;

/// One corpus: committed metadata in `suites/<name>/`, CSV bytes wherever
/// [`data::data_dir`] resolves to.
#[derive(Clone, Copy, Debug)]
pub struct Suite {
    pub name: &'static str,
    /// `true` when the bytes are committed next to the metadata rather than
    /// provisioned into the data root. `fixtures` alone: those files are
    /// written by hand for this repository and are what makes a fresh clone
    /// able to run `cargo test` against something.
    pub committed: bool,
    /// Whether a bare `cargo test` includes this suite.
    pub default: bool,
}

impl Suite {
    /// Where the manifest and overrides live.
    pub fn suite_dir(&self) -> PathBuf {
        data::suite_dir(self.name)
    }

    /// Where this suite's CSV files live.
    pub fn data_dir(&self) -> PathBuf {
        if self.committed {
            self.suite_dir()
        } else {
            data::data_dir(self.name)
        }
    }
}

/// Every corpus this repository knows about. The test and bench binaries pick
/// subsets of this list rather than repeating the names and paths themselves.
pub const SUITES: &[Suite] = &[
    Suite {
        name: "duckdb",
        committed: false,
        default: true,
    },
    Suite {
        name: "fixtures",
        committed: true,
        default: true,
    },
    Suite {
        name: "kaggle",
        committed: false,
        default: false,
    },
    Suite {
        name: "postgres",
        committed: false,
        default: true,
    },
    Suite {
        name: "rust-csv",
        committed: false,
        default: true,
    },
    Suite {
        name: "tpch",
        committed: false,
        default: true,
    },
];

/// Look a suite up by name; panics on an unknown name, which can only be a typo
/// in a test or bench binary.
pub fn suite(name: &str) -> Suite {
    *SUITES
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("unknown suite {name:?}"))
}

/// The suites a test binary should run, in the order given.
///
/// Two things drop a suite: no `manifest.toml` (an unfetched or never-scraped
/// corpus has none), and not being selected. A bare `cargo test` selects the
/// suites marked [`Suite::default`], which is every one but `kaggle`. It comes
/// back in three ways:
///
/// ```text
/// CSVEEE_TEST_SUITES=all             every suite
/// CSVEEE_TEST_SUITES=kaggle,duckdb   exactly these
/// cargo test kaggle::tunguz          any suite the name filter mentions
/// ```
///
/// The third is why `filter` is a parameter: libtest-mimic hands us the name
/// filter before the trial list is built, so naming a suite in it *is* the
/// opt-in. Without that, `cargo test kaggle::tunguz` would match nothing —
/// the suite would have been dropped before the filter ever ran.
///
/// A filter that names no suite (`cargo test tunguz`) selects the defaults, so
/// it searches the same tests a bare run would.
/// Every named suite whose `manifest.toml` exists, in the order given, with no
/// test selection applied — an unfetched or never-scraped corpus has no
/// manifest and drops out here.
///
/// This is what the benches want: `benches/throughput.rs` names the suites it
/// benchmarks explicitly, and `kaggle`'s large files are precisely why it names
/// them. Test binaries want [`suites`] instead.
pub fn available_suites(names: &[&str]) -> Vec<Suite> {
    names
        .iter()
        .map(|n| suite(n))
        .filter(|s| s.suite_dir().join("manifest.toml").exists())
        .collect()
}

/// See [`available_suites`]; this one additionally applies test selection.
pub fn suites(names: &[&str], filter: Option<&str>) -> Vec<Suite> {
    let env = std::env::var("CSVEEE_TEST_SUITES").ok();
    let asked: Option<Vec<&str>> = env.as_deref().map(|v| {
        v.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect()
    });

    let wanted = |s: &Suite| -> bool {
        match asked.as_deref() {
            Some(["all"]) => true,
            Some(list) => list.contains(&s.name),
            None => s.default || filter.is_some_and(|f| f.contains(s.name)),
        }
    };

    let (selected, skipped): (Vec<Suite>, Vec<Suite>) =
        available_suites(names).into_iter().partition(wanted);

    // Say what was left out, once. Listing the entries as ignored instead
    // would bury the run under hundreds of lines that mean "not asked for".
    if !skipped.is_empty() {
        let names: Vec<&str> = skipped.iter().map(|s| s.name).collect();
        eprintln!(
            "note: skipping suite(s) {} — run with CSVEEE_TEST_SUITES=all, \
             or name one in the filter (cargo test {}::)",
            names.join(", "),
            names[0],
        );
    }

    selected
}

/// A single file's dialect + test expectations, after merging manifest + overrides.
#[derive(Clone, Debug)]
pub struct TestEntry {
    pub rel_path: String,
    pub abs_path: PathBuf,
    pub dialect: Dialect,
    pub columns: usize,
    pub flexible: bool,
    pub encoding: String,
    pub skip: bool,
    pub skip_reason: Option<String>,
    pub expect_error: Option<String>,
    /// Skip only the SIMD differential (SIMD diverges on empty/ragged rows).
    pub simd_skip: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Dialect {
    pub delimiter: Option<u8>,
    pub terminator: Option<String>,
    pub quote: Option<String>,
    pub escape: Option<String>,
    pub has_headers: Option<bool>,
    pub comment: Option<u8>,
}

#[derive(Deserialize)]
struct Manifest {
    files: HashMap<String, FileEntry>,
}

#[derive(Deserialize, Default)]
struct FileEntry {
    delimiter: Option<String>,
    terminator: Option<String>,
    quote: Option<String>,
    escape: Option<String>,
    has_headers: Option<bool>,
    comment: Option<String>,
    columns: Option<usize>,
    flexible: Option<bool>,
    encoding: Option<String>,
    skip: Option<bool>,
    skip_reason: Option<String>,
    expect_error: Option<String>,
    simd_skip: Option<bool>,
}

/// Size above which a corpus file is benchmark-only by default.
///
/// The harnesses parse every file twice: once with csveee and once with
/// rust-csv, whose single-threaded output is the oracle the comparison rests
/// on. That second parse is the cost — the kaggle sample reaches 61 GB in one
/// file, which is minutes of oracle for a dialect the small files already
/// cover. `benches/throughput.rs` deliberately does not apply this: large
/// files are exactly what it wants.
pub const BIG_FILE_BYTES: u64 = 100 * 1024 * 1024;

/// Whether `entry` is skipped for being large. `CSVEEE_TEST_BIG=1` runs them.
///
/// This is a policy about how long a test run may take, so it lives here
/// rather than as a per-file manifest field: `skip` would also hide the file
/// from the bench, which is backwards for precisely these files.
pub fn too_big_to_test(entry: &TestEntry) -> bool {
    if std::env::var("CSVEEE_TEST_BIG").as_deref() == Ok("1") {
        return false;
    }
    fs::metadata(&entry.abs_path).is_ok_and(|m| m.len() > BIG_FILE_BYTES)
}

/// Read `suites/<name>/manifest.toml` (plus `overrides.toml`) and resolve every
/// entry against the suite's data directory.
///
/// A file the manifest lists but the data root does not have yields an entry
/// whose `abs_path` does not exist; callers report those as ignored rather than
/// failed, so an unprovisioned corpus is quiet rather than noisy.
pub fn load_suite(suite: &Suite) -> Vec<TestEntry> {
    let dir_path = suite.suite_dir();
    let data_path = suite.data_dir();

    let manifest_path = dir_path.join("manifest.toml");
    assert!(
        manifest_path.exists(),
        "manifest.toml not found in {}. Run: .venv/bin/python scripts/sniff_dialects.py {}",
        dir_path.display(),
        suite.name,
    );

    let manifest: Manifest =
        toml::from_str(&fs::read_to_string(&manifest_path).expect("failed to read manifest.toml"))
            .expect("failed to parse manifest.toml");

    let overrides_path = dir_path.join("overrides.toml");
    let overrides: Option<Manifest> = if overrides_path.exists() {
        Some(
            toml::from_str(
                &fs::read_to_string(&overrides_path).expect("failed to read overrides.toml"),
            )
            .expect("failed to parse overrides.toml"),
        )
    } else {
        None
    };

    let mut entries: Vec<TestEntry> = manifest
        .files
        .into_iter()
        .map(|(rel_path, manifest_entry)| {
            let overrides_entry = overrides.as_ref().and_then(|o| o.files.get(&rel_path));
            let merged = merge_entries(&manifest_entry, overrides_entry);
            let abs_path = data_path.join(&rel_path);

            TestEntry {
                rel_path,
                abs_path,
                dialect: Dialect {
                    delimiter: parse_delimiter(merged.delimiter.as_deref()),
                    terminator: merged.terminator,
                    quote: merged.quote,
                    escape: merged.escape,
                    has_headers: merged.has_headers,
                    comment: parse_delimiter(merged.comment.as_deref()),
                },
                columns: merged.columns.unwrap_or(1),
                flexible: merged.flexible.unwrap_or(false),
                encoding: merged.encoding.unwrap_or_else(|| "utf-8".to_string()),
                skip: merged.skip.unwrap_or(false),
                skip_reason: merged.skip_reason,
                expect_error: merged.expect_error,
                simd_skip: merged.simd_skip.unwrap_or(false),
            }
        })
        .collect();

    if let Some(ref ov) = overrides {
        for (rel_path, ov_entry) in &ov.files {
            if entries.iter().any(|e| &e.rel_path == rel_path) {
                continue;
            }
            let abs_path = data_path.join(rel_path);
            entries.push(TestEntry {
                rel_path: rel_path.clone(),
                abs_path,
                dialect: Dialect {
                    delimiter: parse_delimiter(ov_entry.delimiter.as_deref()),
                    terminator: ov_entry.terminator.clone(),
                    quote: ov_entry.quote.clone(),
                    escape: ov_entry.escape.clone(),
                    has_headers: ov_entry.has_headers,
                    comment: parse_delimiter(ov_entry.comment.as_deref()),
                },
                columns: ov_entry.columns.unwrap_or(1),
                flexible: ov_entry.flexible.unwrap_or(false),
                encoding: ov_entry
                    .encoding
                    .clone()
                    .unwrap_or_else(|| "utf-8".to_string()),
                skip: ov_entry.skip.unwrap_or(false),
                skip_reason: ov_entry.skip_reason.clone(),
                expect_error: ov_entry.expect_error.clone(),
                simd_skip: ov_entry.simd_skip.unwrap_or(false),
            });
        }
    }

    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    entries
}

fn merge_entries(base: &FileEntry, overrides: Option<&FileEntry>) -> FileEntry {
    let Some(ov) = overrides else {
        return FileEntry {
            delimiter: base.delimiter.clone(),
            terminator: base.terminator.clone(),
            quote: base.quote.clone(),
            escape: base.escape.clone(),
            has_headers: base.has_headers,
            comment: base.comment.clone(),
            columns: base.columns,
            flexible: base.flexible,
            encoding: base.encoding.clone(),
            skip: base.skip,
            skip_reason: base.skip_reason.clone(),
            expect_error: base.expect_error.clone(),
            simd_skip: base.simd_skip,
        };
    };

    FileEntry {
        delimiter: ov.delimiter.clone().or_else(|| base.delimiter.clone()),
        terminator: ov.terminator.clone().or_else(|| base.terminator.clone()),
        quote: ov.quote.clone().or_else(|| base.quote.clone()),
        escape: ov.escape.clone().or_else(|| base.escape.clone()),
        has_headers: ov.has_headers.or(base.has_headers),
        comment: ov.comment.clone().or_else(|| base.comment.clone()),
        columns: ov.columns.or(base.columns),
        flexible: ov.flexible.or(base.flexible),
        encoding: ov.encoding.clone().or_else(|| base.encoding.clone()),
        skip: ov.skip.or(base.skip),
        skip_reason: ov.skip_reason.clone().or_else(|| base.skip_reason.clone()),
        expect_error: ov
            .expect_error
            .clone()
            .or_else(|| base.expect_error.clone()),
        simd_skip: ov.simd_skip.or(base.simd_skip),
    }
}

fn parse_delimiter(s: Option<&str>) -> Option<u8> {
    let s = s?;
    match s {
        "\\t" | "\t" => Some(b'\t'),
        s if s.len() == 1 => Some(s.as_bytes()[0]),
        _ => panic!("unsupported delimiter: {s:?}"),
    }
}

/// Build a csveee parser from a dialect.
///
/// Uses `QuoteHandling::Literal` to match rust-csv's semantics (the reference
/// implementation these tests compare against).
pub fn build_csveee_parser(dialect: &Dialect) -> csveee::ParserBuilder {
    let mut builder = csveee::ParserBuilder::new().quote_handling(csveee::QuoteHandling::Literal);

    if let Some(d) = dialect.delimiter {
        builder = builder.delimiter(d);
    }

    match dialect.terminator.as_deref() {
        Some("LF") => builder = builder.terminator(csveee::RecordTerminator::LF),
        Some("CR") => builder = builder.terminator(csveee::RecordTerminator::CR),
        Some("CRLF") | None => builder = builder.terminator(csveee::RecordTerminator::CRLF),
        Some(other) => panic!("unsupported terminator: {other}"),
    }

    match dialect.quote.as_deref() {
        Some("") => builder = builder.quote(None),
        Some(s) if s.len() == 1 => builder = builder.quote(Some(s.as_bytes()[0])),
        None => {}
        Some(other) => panic!("unsupported quote: {other:?}"),
    }

    match dialect.escape.as_deref() {
        Some("") => builder = builder.escape(None),
        Some(s) if s.len() == 1 => builder = builder.escape(Some(s.as_bytes()[0])),
        None => {}
        Some(other) => panic!("unsupported escape: {other:?}"),
    }

    if let Some(has_headers) = dialect.has_headers {
        builder = builder.has_headers(has_headers);
    }

    if let Some(comment) = dialect.comment {
        builder = builder.comment(Some(comment));
    }

    builder
}

/// Build a rust-csv reader from a dialect.
pub fn build_csv_reader(path: &Path, dialect: &Dialect) -> csv::Reader<std::fs::File> {
    let mut builder = csv::ReaderBuilder::new();

    if let Some(d) = dialect.delimiter {
        builder.delimiter(d);
    }

    match dialect.terminator.as_deref() {
        Some("LF") => {
            builder.terminator(Terminator::Any(b'\n'));
        }
        Some("CR") => {
            builder.terminator(Terminator::Any(b'\r'));
        }
        Some("CRLF") => {
            builder.terminator(Terminator::CRLF);
        }
        None => {
            builder.terminator(Terminator::CRLF);
        }
        Some(other) => panic!("unsupported terminator: {other}"),
    }

    match dialect.quote.as_deref() {
        Some("") => {
            builder.quoting(false);
        }
        Some(s) if s.len() == 1 => {
            builder.quote(s.as_bytes()[0]);
        }
        None => {}
        Some(other) => panic!("unsupported quote: {other:?}"),
    }

    match dialect.escape.as_deref() {
        Some("") => {}
        Some(s) if s.len() == 1 => {
            builder.escape(Some(s.as_bytes()[0]));
        }
        None => {}
        Some(other) => panic!("unsupported escape: {other:?}"),
    }

    if let Some(has_headers) = dialect.has_headers {
        builder.has_headers(has_headers);
    }

    if let Some(comment) = dialect.comment {
        builder.comment(Some(comment));
    }

    builder.flexible(true);
    builder
        .from_path(path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()))
}

pub type Rows = Vec<Vec<String>>;
pub type ByteRows = Vec<Vec<Vec<u8>>>;

pub fn merge_rows(thread_rows: &mut [Rows]) -> Rows {
    let mut all = Vec::new();
    for rows in thread_rows {
        all.append(rows);
    }
    all
}

pub fn merge_byte_rows(thread_rows: &mut [ByteRows]) -> ByteRows {
    let mut all = Vec::new();
    for rows in thread_rows {
        all.append(rows);
    }
    all
}

/// Parse with Flexible Text mode (variable field count, UTF-8 validated).
pub fn parse_flexible(dialect: &Dialect, path: &str) -> csveee::Result<Rows> {
    let mut parser = build_csveee_parser(dialect).flexible().build();
    parser.parse(
        path,
        Vec::new,
        |rows: &mut Rows, fields: &mut [&mut str]| {
            rows.push(fields.iter().map(|f| f.to_string()).collect());
            Ok(())
        },
        merge_rows,
    )
}

/// Parse with Flexible Bytes mode (variable field count, no UTF-8 validation).
pub fn parse_bytes_flexible(dialect: &Dialect, path: &str) -> csveee::Result<ByteRows> {
    let mut parser = build_csveee_parser(dialect).flexible().bytes().build();
    parser.parse(
        path,
        Vec::new,
        |rows: &mut ByteRows, fields: &mut [&mut [u8]]| {
            rows.push(fields.iter().map(|f| f.to_vec()).collect());
            Ok(())
        },
        merge_byte_rows,
    )
}

/// Parse with Arity Text mode, dispatching on column count at runtime.
pub fn parse_arity(dialect: &Dialect, path: &str, columns: usize) -> csveee::Result<Rows> {
    macro_rules! dispatch_fixed {
        ($($n:literal),+) => {
            match columns {
                $($n => {
                    let mut parser = build_csveee_parser(dialect).build();
                    parser.parse::<_, _, _, _, _, _, $n>(
                        path,
                        Vec::new,
                        |rows: &mut Rows, fields: [&mut str; $n]| {
                            rows.push(fields.iter().map(|f| f.to_string()).collect());
                            Ok(())
                        },
                        merge_rows,
                    )
                },)+
                _ => panic!("Arity mode not supported for {columns} columns (max 64)"),
            }
        };
    }

    dispatch_fixed!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
    )
}

// -- Count-only variants --
//
// Record-count workloads used by benchmarks: the per-record closure does
// nothing but increment a counter, so we measure parser throughput
// without per-row allocation dominating the signal.

fn sum_counts(states: &mut [usize]) -> usize {
    states.iter().sum()
}

/// Count records with Flexible Text mode.
pub fn count_flexible(dialect: &Dialect, path: &str) -> csveee::Result<usize> {
    let mut parser = build_csveee_parser(dialect).flexible().build();
    parser.parse(
        path,
        || 0usize,
        |n: &mut usize, _fields: &mut [&mut str]| {
            *n += 1;
            Ok(())
        },
        sum_counts,
    )
}

/// Count records with Flexible Bytes mode.
pub fn count_bytes_flexible(dialect: &Dialect, path: &str) -> csveee::Result<usize> {
    let mut parser = build_csveee_parser(dialect).flexible().bytes().build();
    parser.parse(
        path,
        || 0usize,
        |n: &mut usize, _fields: &mut [&mut [u8]]| {
            *n += 1;
            Ok(())
        },
        sum_counts,
    )
}

/// Count records with Arity Text mode (fixed column count, dispatched at runtime).
pub fn count_arity(dialect: &Dialect, path: &str, columns: usize) -> csveee::Result<usize> {
    macro_rules! dispatch_fixed {
        ($($n:literal),+) => {
            match columns {
                $($n => {
                    let mut parser = build_csveee_parser(dialect).build();
                    parser.parse::<_, _, _, _, _, _, $n>(
                        path,
                        || 0usize,
                        |n: &mut usize, _fields: [&mut str; $n]| {
                            *n += 1;
                            Ok(())
                        },
                        sum_counts,
                    )
                },)+
                _ => panic!("Arity mode not supported for {columns} columns (max 64)"),
            }
        };
    }

    dispatch_fixed!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
    )
}

/// Count records with Arity Bytes mode (fixed column count, dispatched at runtime).
pub fn count_bytes_arity(dialect: &Dialect, path: &str, columns: usize) -> csveee::Result<usize> {
    macro_rules! dispatch_fixed {
        ($($n:literal),+) => {
            match columns {
                $($n => {
                    let mut parser = build_csveee_parser(dialect).bytes().build();
                    parser.parse::<_, _, _, _, _, _, $n>(
                        path,
                        || 0usize,
                        |n: &mut usize, _fields: [&mut [u8]; $n]| {
                            *n += 1;
                            Ok(())
                        },
                        sum_counts,
                    )
                },)+
                _ => panic!("Arity mode not supported for {columns} columns (max 64)"),
            }
        };
    }

    dispatch_fixed!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
    )
}

/// Parse with Arity Bytes mode, dispatching on column count at runtime.
pub fn parse_bytes_arity(
    dialect: &Dialect,
    path: &str,
    columns: usize,
) -> csveee::Result<ByteRows> {
    macro_rules! dispatch_fixed {
        ($($n:literal),+) => {
            match columns {
                $($n => {
                    let mut parser = build_csveee_parser(dialect).bytes().build();
                    parser.parse::<_, _, _, _, _, _, $n>(
                        path,
                        Vec::new,
                        |rows: &mut ByteRows, fields: [&mut [u8]; $n]| {
                            rows.push(fields.iter().map(|f| f.to_vec()).collect());
                            Ok(())
                        },
                        merge_byte_rows,
                    )
                },)+
                _ => panic!("Arity mode not supported for {columns} columns (max 64)"),
            }
        };
    }

    dispatch_fixed!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
    )
}

// ── Streaming digest comparison (O(1) memory) ──────────────────────

/// A streaming digest of parsed rows. Tracks row count and a content
/// hash so that we can compare parser outputs without collecting all
/// rows into memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowDigest {
    pub count: usize,
    pub hash: u64,
}

impl RowDigest {
    pub fn new() -> Self {
        Self { count: 0, hash: 0 }
    }

    fn hash_fields(fields: impl Iterator<Item: AsRef<[u8]>>) -> u64 {
        let mut hasher = DefaultHasher::new();
        for field in fields {
            field.as_ref().hash(&mut hasher);
        }
        hasher.finish()
    }

    pub fn add_text_row(&mut self, fields: &[&mut str]) {
        self.count += 1;
        self.hash = self
            .hash
            .wrapping_add(Self::hash_fields(fields.iter().map(|f| f.as_bytes())));
    }

    pub fn add_byte_row(&mut self, fields: &[&mut [u8]]) {
        self.count += 1;
        self.hash = self
            .hash
            .wrapping_add(Self::hash_fields(fields.iter().map(|f| &**f)));
    }

    pub fn add_csv_text_record(&mut self, record: &csv::StringRecord) {
        self.count += 1;
        self.hash = self
            .hash
            .wrapping_add(Self::hash_fields(record.iter().map(|f| f.as_bytes())));
    }

    pub fn add_csv_byte_record(&mut self, record: &csv::ByteRecord) {
        self.count += 1;
        self.hash = self.hash.wrapping_add(Self::hash_fields(record.iter()));
    }
}

fn merge_digests(digests: &mut [RowDigest]) -> RowDigest {
    let mut result = RowDigest::new();
    for d in digests {
        result.count += d.count;
        result.hash = result.hash.wrapping_add(d.hash);
    }
    result
}

pub fn compare_digests(
    label: &str,
    csveee: &RowDigest,
    reference: &RowDigest,
) -> Result<(), String> {
    if csveee.count != reference.count {
        return Err(format!(
            "{label}: row count mismatch — rust-csv: {}, csveee: {}",
            reference.count, csveee.count,
        ));
    }
    if csveee.hash != reference.hash {
        return Err(format!(
            "{label}: content mismatch (row count {}, hash csveee: {:016x}, rust-csv: {:016x})",
            csveee.count, csveee.hash, reference.hash,
        ));
    }
    Ok(())
}

// ── Digest-producing parse functions ────────────────────────────────

pub fn digest_flexible(
    dialect: &Dialect,
    case: BackendCase,
    path: &str,
) -> csveee::Result<RowDigest> {
    let mut parser = case.apply(build_csveee_parser(dialect).flexible()).build();
    parser.parse(
        path,
        RowDigest::new,
        |d: &mut RowDigest, fields: &mut [&mut str]| {
            d.add_text_row(fields);
            Ok(())
        },
        merge_digests,
    )
}

/// Digest of `parse_slice` over the file's bytes read into memory.
pub fn digest_slice_flexible(
    dialect: &Dialect,
    case: BackendCase,
    data: &[u8],
) -> csveee::Result<RowDigest> {
    let mut parser = case.apply(build_csveee_parser(dialect).flexible()).build();
    parser.parse_slice(
        data,
        RowDigest::new,
        |d: &mut RowDigest, fields: &mut [&mut str]| {
            d.add_text_row(fields);
            Ok(())
        },
        merge_digests,
    )
}

/// Bytes-encoding counterpart of [`digest_slice_flexible`].
pub fn digest_slice_bytes_flexible(
    dialect: &Dialect,
    case: BackendCase,
    data: &[u8],
) -> csveee::Result<RowDigest> {
    let mut parser = case
        .apply(build_csveee_parser(dialect).flexible().bytes())
        .build();
    parser.parse_slice(
        data,
        RowDigest::new,
        |d: &mut RowDigest, fields: &mut [&mut [u8]]| {
            d.add_byte_row(fields);
            Ok(())
        },
        merge_digests,
    )
}

pub fn digest_bytes_flexible(
    dialect: &Dialect,
    case: BackendCase,
    path: &str,
) -> csveee::Result<RowDigest> {
    let mut parser = case
        .apply(build_csveee_parser(dialect).flexible().bytes())
        .build();
    parser.parse(
        path,
        RowDigest::new,
        |d: &mut RowDigest, fields: &mut [&mut [u8]]| {
            d.add_byte_row(fields);
            Ok(())
        },
        merge_digests,
    )
}

pub fn digest_arity(
    dialect: &Dialect,
    case: BackendCase,
    path: &str,
    columns: usize,
) -> csveee::Result<RowDigest> {
    macro_rules! dispatch_fixed {
        ($($n:literal),+) => {
            match columns {
                $($n => {
                    let mut parser = case.apply(build_csveee_parser(dialect)).build();
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
                _ => panic!("Arity mode not supported for {columns} columns (max 64)"),
            }
        };
    }

    dispatch_fixed!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
    )
}

pub fn digest_bytes_arity(
    dialect: &Dialect,
    case: BackendCase,
    path: &str,
    columns: usize,
) -> csveee::Result<RowDigest> {
    macro_rules! dispatch_fixed {
        ($($n:literal),+) => {
            match columns {
                $($n => {
                    let mut parser = case.apply(build_csveee_parser(dialect).bytes()).build();
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
                _ => panic!("Arity mode not supported for {columns} columns (max 64)"),
            }
        };
    }

    dispatch_fixed!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
    )
}

// ── Rust-csv reference digests ──────────────────────────────────────

/// Build a rust-csv text digest, stripping trailing empty records.
///
/// rust-csv emits a phantom `[""]` record when a file ends with an
/// unterminated comment line (no trailing newline). csveee correctly
/// produces nothing, so we strip those trailing records to match.
pub fn csv_text_digest(path: &Path, dialect: &Dialect) -> RowDigest {
    let mut csv_reader = build_csv_reader(path, dialect);
    let mut digest = RowDigest::new();
    let mut trimmed = digest.clone();
    for r in csv_reader.records() {
        let record = r.unwrap_or_else(|e| panic!("rust-csv error: {e}"));
        digest.add_csv_text_record(&record);
        let is_empty = record.len() == 1 && record.get(0) == Some("");
        if !is_empty {
            trimmed = digest.clone();
        }
    }
    trimmed
}

/// Build a rust-csv byte digest, stripping trailing empty records.
pub fn csv_byte_digest(path: &Path, dialect: &Dialect) -> RowDigest {
    let mut csv_reader = build_csv_reader(path, dialect);
    let mut digest = RowDigest::new();
    let mut trimmed = digest.clone();
    for r in csv_reader.byte_records() {
        let record = r.unwrap_or_else(|e| panic!("rust-csv error: {e}"));
        digest.add_csv_byte_record(&record);
        let is_empty = record.len() == 1 && record.get(0) == Some(b"" as &[u8]);
        if !is_empty {
            trimmed = digest.clone();
        }
    }
    trimmed
}
