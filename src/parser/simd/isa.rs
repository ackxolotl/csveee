//! The SIMD parser's instruction-set seam: [`Isa`] bundles the primitives
//! whose best implementation depends on the CPU, and everything above
//! them is generic over it. A token is a zero-sized value, so calling a
//! primitive needs one in hand: tokens of ISAs beyond the compile-time
//! target come only from runtime detection, see [`Level`].
//!
//! Detection happens once per process. `CSVEEE_SIMD_LEVEL` pins a level
//! for testing and benchmarking (`compile-time`, `avx2`, `avx512`, `pmull`);
//! a level this CPU or build lacks falls back to detection.

use std::sync::OnceLock;

use super::bitmask::VECTOR_BYTES;
use super::lanes::CompileTimeLanes;

/// The per-ISA primitives under the bitmask pipeline.
pub(super) trait Isa: Copy {
    /// A 64-byte input vector held in registers.
    type Lanes: Copy;

    /// Load a 64-byte input vector.
    fn load(self, input: &[u8; VECTOR_BYTES]) -> Self::Lanes;

    /// Bit `i` set iff byte `i` of `v` equals `b`.
    fn eq_mask(self, v: Self::Lanes, b: u8) -> u64;

    /// XOR prefix sum of `q`: bit `i` is the XOR of input bits `0..=i`.
    fn xor_prefix_sum(self, q: u64) -> u64;

    /// Parallel bit extract: gather the bits of `src` selected by `mask`
    /// into the low bits of the result.
    fn pext(self, src: u64, mask: u64) -> u64;

    /// Whether [`Isa::pext`] is a single instruction rather than a loop
    /// over the mask's bits.
    const FAST_PEXT: bool;

    /// Run `f` compiled for this ISA. Target features only reach code
    /// inlined into the function that enables them, so `f` should be a
    /// thin closure around an `#[inline(always)]` body.
    fn vectorize<R>(self, f: impl FnOnce() -> R) -> R;
}

/// Whatever the build targets: each primitive takes the widest
/// implementation `cfg(target_feature)` enables, so the token is free to
/// construct.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CompileTime;

impl Isa for CompileTime {
    type Lanes = CompileTimeLanes;

    const FAST_PEXT: bool = cfg!(all(target_arch = "x86_64", target_feature = "bmi2"));

    #[inline(always)]
    fn load(self, input: &[u8; VECTOR_BYTES]) -> CompileTimeLanes {
        // SAFETY: `CompileTimeLanes` is a kernel the target enables.
        unsafe { CompileTimeLanes::load(input) }
    }

    #[inline(always)]
    fn eq_mask(self, v: CompileTimeLanes, b: u8) -> u64 {
        // SAFETY: as for `load`.
        unsafe { v.eq_mask(b) }
    }

    #[inline(always)]
    fn xor_prefix_sum(self, q: u64) -> u64 {
        super::bitmask::xor_prefix_sum(q)
    }

    #[inline(always)]
    fn pext(self, src: u64, mask: u64) -> u64 {
        super::bitops::pext(src, mask)
    }

    #[inline(always)]
    fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        f()
    }
}

/// Define a runtime-detected x86_64 token: the target features it enables,
/// the `Lanes` kernel it runs, and `$detect`, which hands it out once the
/// CPU has every feature. Where the build already enables them all,
/// [`CompileTime`] covers the level: the token is an alias of it and
/// `$detect` declines, so no second copy of the parser gets compiled.
macro_rules! x86_token {
    (
        $(#[$doc:meta])*
        $name:ident, $detect:ident, lanes: $lanes:ident,
        features: $features:literal, detect: [$($feature:tt),+ $(,)?],
    ) => {
        $(#[$doc])*
        #[cfg(all(target_arch = "x86_64", not(all($(target_feature = $feature),+))))]
        #[derive(Debug, Clone, Copy)]
        pub(super) struct $name(());

        $(#[$doc])*
        #[cfg(all(target_arch = "x86_64", all($(target_feature = $feature),+)))]
        pub(super) type $name = CompileTime;

        #[cfg(all(target_arch = "x86_64", not(all($(target_feature = $feature),+))))]
        fn $detect() -> Option<$name> {
            (true $(&& std::arch::is_x86_feature_detected!($feature))+).then_some($name(()))
        }

        #[cfg(all(target_arch = "x86_64", all($(target_feature = $feature),+)))]
        fn $detect() -> Option<$name> {
            None
        }

        #[cfg(all(target_arch = "x86_64", not(all($(target_feature = $feature),+))))]
        impl Isa for $name {
            type Lanes = super::lanes::$lanes;

            const FAST_PEXT: bool = true;

            #[inline(always)]
            fn load(self, input: &[u8; VECTOR_BYTES]) -> Self::Lanes {
                // SAFETY: the token exists only once detection saw its features.
                unsafe { super::lanes::$lanes::load(input) }
            }

            #[inline(always)]
            fn eq_mask(self, v: Self::Lanes, b: u8) -> u64 {
                // SAFETY: as for `load`.
                unsafe { v.eq_mask(b) }
            }

            #[inline(always)]
            fn xor_prefix_sum(self, q: u64) -> u64 {
                // SAFETY: as for `load`; every x86 token includes PCLMULQDQ.
                unsafe { super::bitmask::xor_prefix_sum_clmul(q) }
            }

            #[inline(always)]
            fn pext(self, src: u64, mask: u64) -> u64 {
                // SAFETY: as for `load`; every x86 token includes BMI2.
                unsafe { super::bitops::pext_bmi2(src, mask) }
            }

            #[inline(always)]
            fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
                #[target_feature(enable = $features)]
                fn run<R>(f: impl FnOnce() -> R) -> R {
                    f()
                }
                // SAFETY: as for `load`.
                unsafe { run(f) }
            }
        }

        // Named after `$detect` for a unique name; modules and functions
        // live in separate namespaces.
        #[cfg(all(test, target_arch = "x86_64", not(all($(target_feature = $feature),+))))]
        mod $detect {
            /// `run` must enable exactly the features `$detect` checks, or a
            /// token could run code on a CPU nobody checked.
            #[test]
            fn enables_exactly_what_it_detects() {
                let enabled: Vec<&str> = $features.split(',').collect();
                assert_eq!(enabled, [$($feature),+]);
            }
        }
    };
}

x86_token! {
    /// AVX2 with BMI1/2, LZCNT, POPCNT and PCLMULQDQ. Not x86-64-v3, which
    /// lacks PCLMULQDQ and adds FMA, MOVBE and F16C the parser does not use.
    Avx2, detect_avx2, lanes: Avx2Lanes,
    features: "avx2,bmi1,bmi2,lzcnt,popcnt,pclmulqdq",
    detect: ["avx2", "bmi1", "bmi2", "lzcnt", "popcnt", "pclmulqdq"],
}

x86_token! {
    /// [`Avx2`] plus AVX-512F/BW/VL.
    Avx512, detect_avx512, lanes: Avx512Lanes,
    features: "avx2,bmi1,bmi2,lzcnt,popcnt,pclmulqdq,avx512f,avx512bw,avx512vl",
    detect: [
        "avx2", "bmi1", "bmi2", "lzcnt", "popcnt", "pclmulqdq",
        "avx512f", "avx512bw", "avx512vl",
    ],
}

/// NEON plus `pmull`, which generic aarch64 Linux targets lack at compile time.
#[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
#[derive(Debug, Clone, Copy)]
pub(super) struct Pmull(());

/// NEON plus `pmull`; the target already has it, so [`CompileTime`] covers it.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub(super) type Pmull = CompileTime;

/// The token, if this CPU has the AES extension (and with it `pmull`).
#[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
fn detect_pmull() -> Option<Pmull> {
    std::arch::is_aarch64_feature_detected!("aes").then_some(Pmull(()))
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
fn detect_pmull() -> Option<Pmull> {
    None
}

#[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
impl Isa for Pmull {
    type Lanes = CompileTimeLanes;

    const FAST_PEXT: bool = false;

    #[inline(always)]
    fn load(self, input: &[u8; VECTOR_BYTES]) -> CompileTimeLanes {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { CompileTimeLanes::load(input) }
    }

    #[inline(always)]
    fn eq_mask(self, v: CompileTimeLanes, b: u8) -> u64 {
        // SAFETY: as for `load`.
        unsafe { v.eq_mask(b) }
    }

    #[inline(always)]
    fn xor_prefix_sum(self, q: u64) -> u64 {
        // SAFETY: the token exists only once detection saw `aes`.
        unsafe { super::bitmask::xor_prefix_sum_pmull(q) }
    }

    #[inline(always)]
    fn pext(self, src: u64, mask: u64) -> u64 {
        super::bitops::pext(src, mask)
    }

    #[inline(always)]
    fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        #[target_feature(enable = "aes")]
        fn run<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        // SAFETY: as for `xor_prefix_sum`.
        unsafe { run(f) }
    }
}

/// The ISA level the SIMD parser runs at, holding its token.
#[derive(Debug, Clone, Copy)]
pub(super) enum Level {
    CompileTime(CompileTime),
    #[cfg(target_arch = "x86_64")]
    Avx2(Avx2),
    #[cfg(target_arch = "x86_64")]
    Avx512(Avx512),
    #[cfg(target_arch = "aarch64")]
    Pmull(Pmull),
}

impl Level {
    /// This process's level: the best one the CPU supports, unless
    /// `CSVEEE_SIMD_LEVEL` pins an available one.
    pub(super) fn get() -> Self {
        static LEVEL: OnceLock<Level> = OnceLock::new();
        *LEVEL.get_or_init(|| {
            let pinned = std::env::var("CSVEEE_SIMD_LEVEL").ok();
            let available = Self::available();
            pinned
                .and_then(|name| available.iter().copied().find(|l| l.name() == name))
                .unwrap_or(available[0])
        })
    }

    /// Every level this CPU and build support beyond [`CompileTime`], best
    /// first, then [`Level::CompileTime`] itself.
    fn available() -> Vec<Self> {
        let mut levels = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            levels.extend(detect_avx512().map(Level::Avx512));
            levels.extend(detect_avx2().map(Level::Avx2));
        }
        #[cfg(target_arch = "aarch64")]
        levels.extend(detect_pmull().map(Level::Pmull));
        levels.push(Level::CompileTime(CompileTime));
        levels
    }

    /// The level's `CSVEEE_SIMD_LEVEL` name.
    pub(super) fn name(self) -> &'static str {
        match self {
            Level::CompileTime(_) => "compile-time",
            #[cfg(target_arch = "x86_64")]
            Level::Avx2(_) => "avx2",
            #[cfg(target_arch = "x86_64")]
            Level::Avx512(_) => "avx512",
            #[cfg(target_arch = "aarch64")]
            Level::Pmull(_) => "pmull",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_is_available_and_compile_time_is_last() {
        let available = Level::available();
        assert!(matches!(available.last(), Some(Level::CompileTime(_))));
        let level = Level::get();
        assert!(available.iter().any(|l| l.name() == level.name()));
        eprintln!(
            "SIMD level: {} (available: {:?})",
            level.name(),
            available.iter().map(|l| l.name()).collect::<Vec<_>>()
        );
    }
}
