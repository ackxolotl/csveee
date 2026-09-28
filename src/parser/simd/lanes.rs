//! Byte-equality bitmasks over a 64-byte vector: the one per-ISA kernel
//! under [`super::bitmask::match_structural`]. Each ISA is an inner
//! module with the same shape: `load` and `eq_mask`, both `unsafe` with
//! the sole contract that the CPU supports that ISA. The x86 kernels are
//! all compiled on x86_64 and carry `#[target_feature]`, so runtime
//! dispatch can pick one; [`CompileTimeLanes`] is the widest one the
//! target enables at compile time. SSE2 and NEON are baseline on their
//! targets, so only other architectures fall back to scalar code.

// An x86 kernel goes unused where the build's own features already cover
// the dispatch level that would run it, e.g. SSE2 in an AVX2 build.
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub(super) use self::avx2::Avx2Lanes;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub(super) use self::avx512::Avx512Lanes;
#[cfg(target_arch = "aarch64")]
pub(super) use self::neon::NeonLanes;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(super) use self::scalar::ScalarLanes;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub(super) use self::sse2::Sse2Lanes;
use super::bitmask::VECTOR_BYTES;

/// The widest kernel `cfg(target_feature)` enables.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub(super) type CompileTimeLanes = Avx512Lanes;
/// The widest kernel `cfg(target_feature)` enables.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512bw"),
))]
pub(super) type CompileTimeLanes = Avx2Lanes;
/// The widest kernel `cfg(target_feature)` enables.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
pub(super) type CompileTimeLanes = Sse2Lanes;
/// The widest kernel `cfg(target_feature)` enables.
#[cfg(target_arch = "aarch64")]
pub(super) type CompileTimeLanes = NeonLanes;
/// The widest kernel `cfg(target_feature)` enables.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(super) type CompileTimeLanes = ScalarLanes;

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // see the re-exports above
mod avx512 {
    use std::arch::x86_64::{
        __m512i, _mm512_cmpeq_epi8_mask, _mm512_loadu_si512, _mm512_set1_epi8,
    };

    use super::VECTOR_BYTES;

    /// A 64-byte input vector in one ZMM register.
    #[derive(Clone, Copy)]
    pub(in crate::parser::simd) struct Avx512Lanes(__m512i);

    impl Avx512Lanes {
        /// # Safety
        /// The CPU must support AVX-512F and AVX-512BW.
        #[target_feature(enable = "avx512f,avx512bw")]
        #[inline]
        pub unsafe fn load(input: &[u8; VECTOR_BYTES]) -> Self {
            // SAFETY: `input` is 64 readable bytes and the load is unaligned.
            Self(unsafe { _mm512_loadu_si512(input.as_ptr().cast()) })
        }

        /// Bit `i` set iff byte `i` equals `b`.
        ///
        /// # Safety
        /// The CPU must support AVX-512F and AVX-512BW.
        #[target_feature(enable = "avx512f,avx512bw")]
        #[inline]
        pub unsafe fn eq_mask(self, b: u8) -> u64 {
            _mm512_cmpeq_epi8_mask(self.0, _mm512_set1_epi8(b as i8))
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // see the re-exports above
mod avx2 {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_set1_epi8};

    use super::VECTOR_BYTES;

    /// `vpmovmskb` as opaque asm: LLVM otherwise sees through the
    /// intrinsic and rebuilds a vector of 64 booleans, which it can only
    /// scalarize bit by bit on AVX2 without AVX-512 mask registers.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn movemask(v: __m256i) -> u32 {
        let m: u32;
        // SAFETY: register-only operands; AVX2 is enabled for this fn.
        unsafe {
            std::arch::asm!(
                "vpmovmskb {m:e}, {v}",
                m = out(reg) m,
                v = in(ymm_reg) v,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        m
    }

    /// A 64-byte input vector in two YMM registers.
    #[derive(Clone, Copy)]
    pub(in crate::parser::simd) struct Avx2Lanes([__m256i; 2]);

    impl Avx2Lanes {
        /// # Safety
        /// The CPU must support AVX2.
        #[target_feature(enable = "avx2")]
        #[inline]
        pub unsafe fn load(input: &[u8; VECTOR_BYTES]) -> Self {
            let p = input.as_ptr().cast::<__m256i>();
            // SAFETY: `input` is 64 readable bytes, two unaligned 32-byte loads.
            unsafe { Self([_mm256_loadu_si256(p), _mm256_loadu_si256(p.add(1))]) }
        }

        /// Bit `i` set iff byte `i` equals `b`.
        ///
        /// # Safety
        /// The CPU must support AVX2.
        #[target_feature(enable = "avx2")]
        #[inline]
        pub unsafe fn eq_mask(self, b: u8) -> u64 {
            let s = _mm256_set1_epi8(b as i8);
            let lo = movemask(_mm256_cmpeq_epi8(self.0[0], s));
            let hi = movemask(_mm256_cmpeq_epi8(self.0[1], s));
            (lo as u64) | ((hi as u64) << 32)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // see the re-exports above
mod sse2 {
    use std::arch::x86_64::{
        __m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8,
    };

    use super::VECTOR_BYTES;

    /// A 64-byte input vector in four XMM registers.
    #[derive(Clone, Copy)]
    pub(in crate::parser::simd) struct Sse2Lanes([__m128i; 4]);

    impl Sse2Lanes {
        /// # Safety
        /// None beyond x86_64, where SSE2 is baseline; `unsafe` only to
        /// share the other kernels' signature.
        #[inline]
        pub unsafe fn load(input: &[u8; VECTOR_BYTES]) -> Self {
            let p = input.as_ptr().cast::<__m128i>();
            // SAFETY: `input` is 64 readable bytes, four unaligned 16-byte loads.
            unsafe {
                Self([
                    _mm_loadu_si128(p),
                    _mm_loadu_si128(p.add(1)),
                    _mm_loadu_si128(p.add(2)),
                    _mm_loadu_si128(p.add(3)),
                ])
            }
        }

        /// Bit `i` set iff byte `i` equals `b`.
        ///
        /// # Safety
        /// As for [`Self::load`].
        #[inline]
        pub unsafe fn eq_mask(self, b: u8) -> u64 {
            // SAFETY: SSE2 is baseline on x86_64.
            unsafe {
                let s = _mm_set1_epi8(b as i8);
                let mut mask = 0u64;
                for (i, &v) in self.0.iter().enumerate() {
                    // `movemask` fills the low 16 bits only.
                    let m = _mm_movemask_epi8(_mm_cmpeq_epi8(v, s)) as u32 as u64;
                    mask |= m << (16 * i);
                }
                mask
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::{
        uint8x16x4_t, vandq_u8, vceqq_u8, vdupq_n_u8, vgetq_lane_u64, vld1q_u8, vld1q_u8_x4,
        vpaddq_u8, vreinterpretq_u64_u8,
    };

    use super::VECTOR_BYTES;

    /// A 64-byte input vector in four Q registers.
    #[derive(Clone, Copy)]
    pub(in crate::parser::simd) struct NeonLanes(uint8x16x4_t);

    impl NeonLanes {
        /// # Safety
        /// None beyond aarch64, where NEON is baseline; `unsafe` only to
        /// share the other kernels' signature.
        #[inline]
        pub unsafe fn load(input: &[u8; VECTOR_BYTES]) -> Self {
            // SAFETY: `input` is 64 readable bytes.
            unsafe { Self(vld1q_u8_x4(input.as_ptr())) }
        }

        /// Bit `i` set iff byte `i` equals `b`. NEON has no `movemask`:
        /// weight each matching lane by its bit within the byte, then fold
        /// 64 lanes into 8 bytes with pairwise adds (simdjson's reduction).
        ///
        /// # Safety
        /// As for [`Self::load`].
        #[inline]
        pub unsafe fn eq_mask(self, b: u8) -> u64 {
            const WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            // SAFETY: NEON is baseline on aarch64; `WEIGHTS` is 16 readable bytes.
            unsafe {
                let w = vld1q_u8(WEIGHTS.as_ptr());
                let s = vdupq_n_u8(b);
                let m0 = vandq_u8(vceqq_u8(self.0.0, s), w);
                let m1 = vandq_u8(vceqq_u8(self.0.1, s), w);
                let m2 = vandq_u8(vceqq_u8(self.0.2, s), w);
                let m3 = vandq_u8(vceqq_u8(self.0.3, s), w);
                let sum = vpaddq_u8(vpaddq_u8(m0, m1), vpaddq_u8(m2, m3));
                let sum = vpaddq_u8(sum, sum);
                vgetq_lane_u64::<0>(vreinterpretq_u64_u8(sum))
            }
        }
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod scalar {
    use super::VECTOR_BYTES;

    /// A 64-byte input vector, compared byte by byte.
    #[derive(Clone, Copy)]
    pub(in crate::parser::simd) struct ScalarLanes([u8; VECTOR_BYTES]);

    impl ScalarLanes {
        /// # Safety
        /// None; `unsafe` only to share the other kernels' signature.
        #[inline]
        pub unsafe fn load(input: &[u8; VECTOR_BYTES]) -> Self {
            Self(*input)
        }

        /// Bit `i` set iff byte `i` equals `b`.
        ///
        /// # Safety
        /// None; `unsafe` only to share the other kernels' signature.
        #[inline]
        pub unsafe fn eq_mask(self, b: u8) -> u64 {
            super::eq_mask_scalar(&self.0, b)
        }
    }
}

/// Scalar byte-equality bitmask.
#[allow(dead_code)] // live on targets without a vector ISA; also the tests' oracle
#[inline]
fn eq_mask_scalar(input: &[u8; VECTOR_BYTES], b: u8) -> u64 {
    let mut mask = 0u64;
    for (i, &byte) in input.iter().enumerate() {
        mask |= ((byte == b) as u64) << i;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check one kernel against the scalar oracle: every byte value in
    /// every lane, then repeated high bytes, where signed compares trip.
    fn check_kernel(name: &str, eq_mask: impl Fn(&[u8; VECTOR_BYTES], u8) -> u64) {
        for rot in 0..=255u8 {
            let input: [u8; VECTOR_BYTES] =
                std::array::from_fn(|i| (i as u8).wrapping_mul(7).wrapping_add(rot));
            for b in 0..=255u8 {
                assert_eq!(
                    eq_mask(&input, b),
                    eq_mask_scalar(&input, b),
                    "{name}: rot {rot}, byte {b:#04x}"
                );
            }
        }
        let input: [u8; VECTOR_BYTES] =
            std::array::from_fn(|i| if i % 3 == 0 { 0xff } else { 0x80 });
        for b in [0x00, 0x7f, 0x80, 0xff] {
            assert_eq!(
                eq_mask(&input, b),
                eq_mask_scalar(&input, b),
                "{name}: byte {b:#04x}"
            );
        }
    }

    #[test]
    fn compile_time_kernel_matches_scalar() {
        // SAFETY: `CompileTimeLanes` is a kernel the target enables.
        check_kernel("compile-time", |v, b| unsafe {
            CompileTimeLanes::load(v).eq_mask(b)
        });
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_kernels_match_scalar() {
        // SAFETY: SSE2 is baseline on x86_64.
        check_kernel("sse2", |v, b| unsafe { Sse2Lanes::load(v).eq_mask(b) });
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just detected.
            check_kernel("avx2", |v, b| unsafe { Avx2Lanes::load(v).eq_mask(b) });
        }
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: AVX-512F and AVX-512BW were just detected.
            check_kernel("avx512", |v, b| unsafe { Avx512Lanes::load(v).eq_mask(b) });
        }
    }
}
