# Changelog

## Unreleased

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
