//! `examples/simple.rs` with tracing wired up: parse a file and record a
//! Perfetto trace of the parse to `target/csveee.pftrace`.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example trace --features trace-example -- path/to/your.csv
//! ```
//!
//! The path is optional and defaults to a small bundled fixture. Point it at
//! the file that parses slower than you expect, then open the trace at
//! <https://ui.perfetto.dev>. Use `--release`: the timings from a debug build
//! are not representative.

use std::fs::File;
use std::sync::Mutex;

use csveee::ParserBuilder;
use tracing_perfetto::PerfettoLayer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let csv_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "suites/fixtures/standard.csv".to_string());

    let trace_path = "target/csveee.pftrace";
    let layer =
        PerfettoLayer::new(Mutex::new(File::create(trace_path)?)).with_debug_annotations(true);

    // `RUST_LOG` overrides this if set.
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,csveee=debug"));

    tracing_subscriber::registry()
        .with(filter)
        .with(layer)
        .init();

    let mut parser = ParserBuilder::new().flexible().build();

    let records = parser.parse(
        &csv_path,
        || 0usize,
        |count, _fields| {
            *count += 1;
            Ok(())
        },
        |counts| counts.iter().sum::<usize>(),
    )?;

    println!("parsed {records} records from {csv_path}");
    println!("trace:  {trace_path}  — open at https://ui.perfetto.dev");

    Ok(())
}
