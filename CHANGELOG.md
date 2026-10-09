# Changelog

## 0.2.1 - 2026-10-09

### Fixed

- Builds on targets other than Linux, macOS and Windows, such as
  WebAssembly and FreeBSD. There the ring-buffer backend copies instead
  of mapping its memory twice. On `wasm32-unknown-unknown`, use
  `parse_slice` or `parse_stream`, since there is no file system.
- A worker thread that cannot be spawned now makes the parse return an
  error instead of panicking. On targets without threads, keep the
  default `concurrency`.

## 0.2.0 - 2026-09-29

### Breaking

- The `simd` feature is gone: the SIMD parser is now always built.
  Remove `features = ["simd"]` from your `Cargo.toml`.

### Changed

- The crate builds on stable Rust; nightly is no longer needed.
- The SIMD parser picks the best instruction set for the CPU at runtime.
- Tuned the ring-buffer size, notably on Apple platforms.
- Faster field-count verification on CPUs without native PEXT.

## 0.1.1 - 2026-09-24

- Faster index construction in the SIMD parser.
- Documentation improvements.

## 0.1.0 - 2026-09-01

- Initial release.
