//! Throughput benchmark comparing csveee against rust-csv and DuckDB.

#[path = "../tests/common/shared.rs"]
mod shared;

use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use csveee::{ParserBackend, ParserBuilder, RecordTerminator};

use self::shared::{Dialect, TestEntry, available_suites, build_csv_reader, load_suite};

const SUITES: &[&str] = &["duckdb", "kaggle", "tpch"];

fn bench_all(c: &mut Criterion) {
    for suite in available_suites(SUITES) {
        for entry in load_suite(&suite) {
            if !is_benchable(&entry) {
                continue;
            }

            let group_name = format!("{}::{}", suite.name, entry.rel_path);
            if !group_matches(&group_name) {
                continue;
            }

            let Ok(size) = fs::metadata(&entry.abs_path).map(|m| m.len()) else {
                continue;
            };
            if size == 0 {
                continue;
            }

            let mut group = c.benchmark_group(&group_name);
            group
                .throughput(Throughput::BytesDecimal(size))
                .sample_size(10)
                .measurement_time(Duration::from_secs(3))
                .warm_up_time(Duration::from_millis(500));

            bench_csveee(
                &mut group,
                &group_name,
                &entry,
                ParserBackend::Dfa,
                "csveee-dfa",
            );
            if simd_supports(&entry) {
                bench_csveee(
                    &mut group,
                    &group_name,
                    &entry,
                    ParserBackend::Simd,
                    "csveee-simd",
                );
            }
            bench_rust_csv(&mut group, &group_name, &entry);
            bench_duckdb(&mut group, &group_name, &entry);

            group.finish();
        }
    }
}

/// Skip entries that aren't meaningful to benchmark.
fn is_benchable(entry: &TestEntry) -> bool {
    if entry.skip {
        return false;
    }
    if entry.expect_error.is_some() {
        return false;
    }
    if entry.encoding != "utf-8" {
        return false;
    }
    if !entry.abs_path.exists() {
        return false;
    }
    true
}

fn bench_csveee(
    group: &mut BenchmarkGroup<'_, WallTime>,
    group_name: &str,
    entry: &TestEntry,
    backend: ParserBackend,
    label: &'static str,
) {
    let bench_id = format!("{group_name}/{label}");
    if !bench_filter_matches(&bench_id) {
        return;
    }

    let path = entry.abs_path.to_str().unwrap().to_string();
    let dialect = entry.dialect.clone();
    let use_arity = !entry.flexible && entry.columns <= 64;
    let columns = entry.columns;

    let dry_run = if use_arity {
        csveee_count_arity(&dialect, backend, &path, columns)
    } else {
        csveee_count_flexible(&dialect, backend, &path)
    };
    if let Err(e) = dry_run {
        eprintln!(
            "[{label}] skip {}: parse failed: {}",
            entry.rel_path,
            sanitize(&e.to_string())
        );
        return;
    }

    group.bench_function(label, |b| {
        if use_arity {
            b.iter(|| {
                let n = csveee_count_arity(&dialect, backend, &path, columns)
                    .expect("csveee arity parse");
                black_box(n);
            });
        } else {
            b.iter(|| {
                let n =
                    csveee_count_flexible(&dialect, backend, &path).expect("csveee flexible parse");
                black_box(n);
            });
        }
    });
}

/// True if any benchmark in this group could be selected by the CLI filter.
fn group_matches(group_name: &str) -> bool {
    ["csveee-dfa", "csveee-simd", "rust-csv", "duckdb"]
        .iter()
        .any(|label| bench_filter_matches(&format!("{group_name}/{label}")))
}

fn bench_filter_matches(id: &str) -> bool {
    use std::sync::OnceLock;

    enum BenchFilter {
        AcceptAll,
        RejectAll,
        Exact(String),
        Regex(regex::Regex),
    }

    static FILTER: OnceLock<BenchFilter> = OnceLock::new();
    let filter = FILTER.get_or_init(|| {
        const VALUE_OPTS: &[&str] = &[
            "--color",
            "-c",
            "--save-baseline",
            "-s",
            "--baseline",
            "-b",
            "--baseline-lenient",
            "--format",
            "--profile-time",
            "--load-baseline",
            "--sample-size",
            "--warm-up-time",
            "--measurement-time",
            "--nresamples",
            "--noise-threshold",
            "--confidence-level",
            "--significance-level",
            "--plotting-backend",
            "--output-format",
        ];
        let mut exact = false;
        let mut ignored = false;
        let mut positional: Option<String> = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--exact" => exact = true,
                "--ignored" => ignored = true,
                a if VALUE_OPTS.contains(&a) => {
                    args.next();
                }
                a if a.starts_with('-') => {}
                _ if positional.is_none() => positional = Some(arg),
                _ => {}
            }
        }
        if ignored {
            BenchFilter::RejectAll
        } else if let Some(f) = positional {
            if exact {
                BenchFilter::Exact(f)
            } else {
                regex::Regex::new(&f).map_or(BenchFilter::AcceptAll, BenchFilter::Regex)
            }
        } else {
            BenchFilter::AcceptAll
        }
    });

    match filter {
        BenchFilter::AcceptAll => true,
        BenchFilter::RejectAll => false,
        BenchFilter::Exact(e) => id == e,
        BenchFilter::Regex(re) => re.is_match(id),
    }
}

fn simd_supports(entry: &TestEntry) -> bool {
    if entry.columns > 63 {
        return false;
    }
    let builder = bench_builder(&entry.dialect, ParserBackend::Simd);
    if entry.flexible {
        builder
            .flexible()
            .supports_backend(ParserBackend::Simd)
            .is_ok()
    } else {
        builder.supports_backend(ParserBackend::Simd).is_ok()
    }
}

fn bench_builder(dialect: &Dialect, backend: ParserBackend) -> ParserBuilder {
    let mut b = ParserBuilder::new().parser_backend(backend);
    if let Some(d) = dialect.delimiter {
        b = b.delimiter(d);
    }
    match dialect.terminator.as_deref() {
        Some("LF") => b = b.terminator(RecordTerminator::LF),
        Some("CR") => b = b.terminator(RecordTerminator::CR),
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
    if let Some(c) = dialect.comment {
        b = b.comment(Some(c));
    }
    b
}

fn csveee_count_flexible(
    dialect: &Dialect,
    backend: ParserBackend,
    path: &str,
) -> csveee::Result<usize> {
    let mut parser = bench_builder(dialect, backend).flexible().build();
    parser.parse(
        path,
        || 0usize,
        |n: &mut usize, _fields: &mut [&mut str]| {
            *n += 1;
            Ok(())
        },
        |states| states.iter().sum(),
    )
}

fn csveee_count_arity(
    dialect: &Dialect,
    backend: ParserBackend,
    path: &str,
    columns: usize,
) -> csveee::Result<usize> {
    macro_rules! dispatch {
        ($($n:literal),+) => {
            match columns {
                $($n => {
                    let mut parser = bench_builder(dialect, backend).build();
                    parser.parse::<_, _, _, _, _, _, $n>(
                        path,
                        || 0usize,
                        |n: &mut usize, _fields: [&mut str; $n]| {
                            *n += 1;
                            Ok(())
                        },
                        |states| states.iter().sum(),
                    )
                },)+
                _ => panic!("Arity mode not supported for {columns} columns (max 64)"),
            }
        };
    }
    dispatch!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
    )
}

fn bench_rust_csv(group: &mut BenchmarkGroup<'_, WallTime>, group_name: &str, entry: &TestEntry) {
    if !bench_filter_matches(&format!("{group_name}/rust-csv")) {
        return;
    }

    let path = entry.abs_path.clone();
    let dialect = entry.dialect.clone();

    let read_all = || -> Result<usize, csv::Error> {
        let mut reader = build_csv_reader(&path, &dialect);
        let mut record = csv::StringRecord::new();
        let mut n = 0usize;
        while reader.read_record(&mut record)? {
            n += 1;
        }
        Ok(n)
    };

    if let Err(e) = read_all() {
        eprintln!(
            "[rust-csv] skip {}: read failed: {}",
            entry.rel_path,
            sanitize(&e.to_string())
        );
        return;
    }

    group.bench_function("rust-csv", |b| {
        b.iter(|| {
            let n = read_all().expect("rust-csv read");
            black_box(n);
        });
    });
}

/// Escape control bytes in an error message before it reaches the terminal.
fn sanitize(msg: &str) -> String {
    msg.chars()
        .map(|c| {
            if c.is_control() && c != '\n' && c != '\t' {
                format!("\\u{{{:02x}}}", c as u32)
            } else {
                c.to_string()
            }
        })
        .collect()
}

#[cfg(feature = "bench-duckdb")]
fn bench_duckdb(group: &mut BenchmarkGroup<'_, WallTime>, group_name: &str, entry: &TestEntry) {
    use duckdb::Connection;

    if !bench_filter_matches(&format!("{group_name}/duckdb")) {
        return;
    }

    let conn = Connection::open_in_memory().expect("duckdb open");
    let fixed_columns = (!entry.flexible && entry.columns > 0).then_some(entry.columns);
    let sql = format!(
        "SELECT count(*) FROM ({})",
        duckdb_sql_for(&entry.abs_path, &entry.dialect, fixed_columns)
    );

    // prepare
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "[duckdb] skip {}: prepare failed: {}",
                entry.rel_path,
                sanitize(&e.to_string())
            );
            return;
        }
    };

    // dry-run to catch errors ...
    if let Err(e) = stmt.query_row([], |r| r.get::<_, i64>(0)) {
        eprintln!(
            "[duckdb] skip {}: query failed: {}",
            entry.rel_path,
            sanitize(&e.to_string())
        );
        return;
    }

    // ... and bench
    group.bench_function("duckdb", |b| {
        b.iter(|| {
            let n: i64 = stmt.query_row([], |r| r.get(0)).expect("duckdb query");
            black_box(n);
        });
    });
}

#[cfg(not(feature = "bench-duckdb"))]
fn bench_duckdb(_group: &mut BenchmarkGroup<'_, WallTime>, _group_name: &str, _entry: &TestEntry) {
    // Enable with: cargo bench --features bench-duckdb
}

/// Build a DuckDB `read_csv(...)` invocation from a Dialect.
#[cfg_attr(not(feature = "bench-duckdb"), allow(dead_code))]
fn duckdb_sql_for(path: &Path, dialect: &Dialect, fixed_columns: Option<usize>) -> String {
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("'{}'", path.display()));

    let delim = dialect.delimiter.unwrap_or(b',');
    parts.push(format!("delim='{}'", escape_sql_char(delim)));

    let quote = match dialect.quote.as_deref() {
        Some("") => None,
        Some(s) if s.len() == 1 => Some(s.as_bytes()[0]),
        _ => Some(b'"'),
    };
    match quote {
        Some(q) => parts.push(format!("quote='{}'", escape_sql_char(q))),
        None => parts.push("quote=''".into()),
    }

    match dialect.escape.as_deref() {
        Some("") => parts.push("escape=''".into()),
        Some(s) if s.len() == 1 => {
            parts.push(format!("escape='{}'", escape_sql_char(s.as_bytes()[0])))
        }
        _ => {
            if let Some(q) = quote {
                parts.push(format!("escape='{}'", escape_sql_char(q)));
            }
        }
    }
    if let Some(h) = dialect.has_headers {
        parts.push(format!("header={}", if h { "true" } else { "false" }));
    }
    match dialect.terminator.as_deref() {
        Some("LF") => parts.push("new_line='\\n'".into()),
        Some("CR") => parts.push("new_line='\\r'".into()),
        Some("CRLF") => parts.push("new_line='\\r\\n'".into()),
        _ => {}
    }
    if let Some(c) = dialect.comment {
        parts.push(format!("comment='{}'", escape_sql_char(c)));
    }
    match fixed_columns {
        Some(n) => {
            // let's pass in the number of columns we expect
            let cols: Vec<String> = (0..n).map(|i| format!("'c{i}': 'VARCHAR'")).collect();
            parts.push(format!("columns={{{}}}", cols.join(", ")));
            parts.push("auto_detect=false".into());
        }
        None => {
            parts.push("all_varchar=true".into());
        }
    }
    format!("SELECT * FROM read_csv({})", parts.join(", "))
}

fn escape_sql_char(b: u8) -> String {
    match b {
        b'\'' => "''".into(),
        b'\t' => "\\t".into(),
        c => (c as char).to_string(),
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
