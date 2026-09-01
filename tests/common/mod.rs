#![allow(dead_code, unused_imports)]

mod shared;

pub use self::shared::matrix::{BackendCase, PARSER_THREADS, matrix_cases, test_threads};
pub use self::shared::{
    Dialect, RowDigest, Suite, TestEntry, build_csv_reader, compare_digests, csv_byte_digest,
    csv_text_digest, digest_arity, digest_bytes_arity, digest_bytes_flexible, digest_flexible,
    digest_slice_bytes_flexible, digest_slice_flexible, load_suite, parse_arity, parse_bytes_arity,
    parse_bytes_flexible, parse_flexible, suites, too_big_to_test,
};

/// Run a single test entry: parse with csveee and rust-csv, compare digests.
///
/// Uses streaming digest comparison (O(1) memory) so that multi-GB files
/// don't require collecting all rows into memory.
///
/// **UTF-8 files** (default): Text parser (Flexible + optionally Arity) and
/// Bytes parser (Flexible + optionally Arity) are all tested against rust-csv.
///
/// **Non-UTF-8 files** (encoding != "utf-8"): only the Bytes parser is
/// tested, compared against rust-csv's ByteRecord output.
///
/// In `CSVEEE_TEST_MATRIX=full` mode, every case from [`matrix_cases`] is
/// exercised; failures are prefixed with the offending case label.
pub fn run_test_entry(entry: &TestEntry) -> Result<(), String> {
    if !entry.abs_path.exists() {
        return Err(format!("CSV file not found: {}", entry.abs_path.display()));
    }

    let path = entry.abs_path.to_str().unwrap();
    let is_utf8 = entry.encoding == "utf-8";

    if is_utf8 {
        run_utf8_entry(entry, path)
    } else {
        run_bytes_entry(entry, path)
    }
}

/// Test a UTF-8 file with Text (Flexible + Arity) and Bytes parsers.
fn run_utf8_entry(entry: &TestEntry, path: &str) -> Result<(), String> {
    let use_arity = !entry.flexible && entry.columns <= 64;

    if let Some(ref expected_error) = entry.expect_error {
        // Expected-error cases are dialect-driven and case-independent;
        // run a single flexible parse with the default case.
        let flexible_digest = digest_flexible(&entry.dialect, BackendCase::auto(), path);
        return match flexible_digest {
            Err(e) => {
                let err_str = format!("{e}");
                if err_str.contains(expected_error) {
                    Ok(())
                } else {
                    Err(format!(
                        "expected error containing {expected_error:?}, got: {err_str}"
                    ))
                }
            }
            Ok(_) => Err(format!(
                "expected error {expected_error:?}, but parsing succeeded"
            )),
        };
    }

    let csv_text_ref = csv_text_digest(&entry.abs_path, &entry.dialect);
    let csv_byte_ref = csv_byte_digest(&entry.abs_path, &entry.dialect);

    for case in matrix_cases() {
        let label = case.label();
        let prefix = |s: String| format!("[{label}] {s}");

        let flexible_digest = digest_flexible(&entry.dialect, case, path)
            .map_err(|e| prefix(format!("csveee flexible parse error: {e}")))?;
        compare_digests("flexible", &flexible_digest, &csv_text_ref).map_err(prefix)?;

        if use_arity {
            let arity_digest = digest_arity(&entry.dialect, case, path, entry.columns)
                .map_err(|e| prefix(format!("csveee arity parse error: {e}")))?;
            compare_digests("arity", &arity_digest, &csv_text_ref).map_err(prefix)?;
        }

        let bytes_digest = digest_bytes_flexible(&entry.dialect, case, path)
            .map_err(|e| prefix(format!("csveee bytes flexible parse error: {e}")))?;
        compare_digests("bytes-flexible", &bytes_digest, &csv_byte_ref).map_err(prefix)?;

        if use_arity {
            let bytes_arity_digest = digest_bytes_arity(&entry.dialect, case, path, entry.columns)
                .map_err(|e| prefix(format!("csveee bytes arity parse error: {e}")))?;
            compare_digests("bytes-arity", &bytes_arity_digest, &csv_byte_ref).map_err(prefix)?;
        }
    }

    for case in slice_cases() {
        let prefix = |s: String| format!("[slice {}] {s}", case.label());
        let data =
            std::fs::read(&entry.abs_path).map_err(|e| prefix(format!("read failed: {e}")))?;

        let slice_digest = digest_slice_flexible(&entry.dialect, case, &data)
            .map_err(|e| prefix(format!("csveee slice flexible parse error: {e}")))?;
        compare_digests("slice-flexible", &slice_digest, &csv_text_ref).map_err(prefix)?;

        let slice_bytes_digest = digest_slice_bytes_flexible(&entry.dialect, case, &data)
            .map_err(|e| prefix(format!("csveee slice bytes parse error: {e}")))?;
        compare_digests("slice-bytes", &slice_bytes_digest, &csv_byte_ref).map_err(prefix)?;
    }

    Ok(())
}

/// `parse_slice` reads every chunk through one context regardless of the
/// `IoBackend` setting, so one case covers it.
fn slice_cases() -> [BackendCase; 1] {
    [BackendCase::auto()]
}

/// Test a non-UTF-8 file with the Bytes parser only (Flexible + Arity).
fn run_bytes_entry(entry: &TestEntry, path: &str) -> Result<(), String> {
    let use_arity = !entry.flexible && entry.columns <= 64;

    if let Some(ref expected_error) = entry.expect_error {
        let bytes_digest = digest_bytes_flexible(&entry.dialect, BackendCase::auto(), path);
        return match bytes_digest {
            Err(e) => {
                let err_str = format!("{e}");
                if err_str.contains(expected_error) {
                    Ok(())
                } else {
                    Err(format!(
                        "expected error containing {expected_error:?}, got: {err_str}"
                    ))
                }
            }
            Ok(_) => Err(format!(
                "expected error {expected_error:?}, but parsing succeeded"
            )),
        };
    }

    let csv_byte_ref = csv_byte_digest(&entry.abs_path, &entry.dialect);

    for case in matrix_cases() {
        let label = case.label();
        let prefix = |s: String| format!("[{label}] {s}");

        let bytes_digest = digest_bytes_flexible(&entry.dialect, case, path)
            .map_err(|e| prefix(format!("csveee bytes flexible parse error: {e}")))?;
        compare_digests("bytes-flexible", &bytes_digest, &csv_byte_ref).map_err(prefix)?;

        if use_arity {
            let bytes_arity_digest = digest_bytes_arity(&entry.dialect, case, path, entry.columns)
                .map_err(|e| prefix(format!("csveee bytes arity parse error: {e}")))?;
            compare_digests("bytes-arity", &bytes_arity_digest, &csv_byte_ref).map_err(prefix)?;
        }
    }

    for case in slice_cases() {
        let prefix = |s: String| format!("[slice {}] {s}", case.label());
        let data =
            std::fs::read(&entry.abs_path).map_err(|e| prefix(format!("read failed: {e}")))?;
        let slice_bytes_digest = digest_slice_bytes_flexible(&entry.dialect, case, &data)
            .map_err(|e| prefix(format!("csveee slice bytes parse error: {e}")))?;
        compare_digests("slice-bytes", &slice_bytes_digest, &csv_byte_ref).map_err(prefix)?;
    }

    Ok(())
}
