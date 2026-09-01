//! Where the CSV bytes live.
//!
//! `suites/<name>/` holds the committed metadata; the bytes it describes are
//! provisioned separately by `scripts/provision_data.py` and need not sit
//! inside the repository at all. The corpora run from 40 MB to tens of GB —
//! `tpch` at sf100 alone is 79 GB — which is more than a home directory should
//! be asked to hold, so the data root is relocatable.
//!
//! Resolution order for suite `<name>`, first hit wins:
//!
//!   1. `$CSVEEE_DATA_<NAME>` — that one suite's directory (`-` becomes `_`,
//!      so `rust-csv` reads `CSVEEE_DATA_RUST_CSV`)
//!   2. `$CSVEEE_DATA` — parent directory for every suite
//!   3. `data.local.toml` at the repository root (untracked):
//!      `root = "..."` and/or `[suites]` `tpch = "..."`
//!   4. `<repo>/data` — the default, so a fresh clone needs no configuration
//!
//! Per-suite settings (1 and `[suites]` in 3) name the suite's own directory;
//! the blanket settings (2 and 4) name a parent that `<name>` is appended to.
//! The asymmetry is deliberate: the common case is one relocated tree, and the
//! interesting case is a single oversized suite sent somewhere else on its own.
//!
//! A config *file* exists alongside the environment variables because the
//! variables are easy to lose: an IDE run configuration does not apply to
//! `cargo test` in a terminal, and vice versa. The file applies to both.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::OnceLock;

/// The repository root, baked in at compile time so resolution does not
/// depend on the working directory the test binary happens to be run from.
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Committed metadata for `suite`: `suites/<name>/`.
pub fn suite_dir(suite: &str) -> PathBuf {
    repo_root().join("suites").join(suite)
}

/// The directory holding `suite`'s CSV bytes, per the order documented above.
pub fn data_dir(suite: &str) -> PathBuf {
    if let Some(dir) = env_var(&format!("CSVEEE_DATA_{}", env_suffix(suite))) {
        return PathBuf::from(dir);
    }
    if let Some(root) = env_var("CSVEEE_DATA") {
        return PathBuf::from(root).join(suite);
    }
    if let Some(dir) = config_lookup(&["suites", suite]) {
        return PathBuf::from(dir);
    }
    if let Some(root) = config_lookup(&["root"]) {
        return PathBuf::from(root).join(suite);
    }
    repo_root().join("data").join(suite)
}

/// `rust-csv` -> `RUST_CSV`. Suite names are ASCII, so this stays a plain
/// uppercase; anything exotic would need quoting rules an env var cannot have.
fn env_suffix(suite: &str) -> String {
    suite.replace('-', "_").to_uppercase()
}

fn env_var(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(v) if !v.trim().is_empty() => Some(v),
        _ => None,
    }
}

/// Parsed `data.local.toml`, or `None` when it does not exist.
fn config() -> Option<&'static toml::Value> {
    static CONFIG: OnceLock<Option<toml::Value>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let path = repo_root().join("data.local.toml");
            let text = std::fs::read_to_string(&path).ok()?;
            match toml::from_str::<toml::Value>(&text) {
                Ok(v) => Some(v),
                // A malformed config would otherwise degrade to "every suite
                // is ignored", which looks like a missing download rather than
                // a typo. Say which it is.
                Err(e) => panic!("failed to parse {}: {e}", path.display()),
            }
        })
        .as_ref()
}

fn config_lookup(path: &[&str]) -> Option<String> {
    let mut node = config()?;
    for key in path {
        node = node.get(key)?;
    }
    node.as_str().map(str::to_owned)
}
