//! The SIMD parser's instruction-set seam: [`Isa`] bundles the primitives
//! whose best implementation depends on the CPU, and everything above
//! them is generic over it. A token is a zero-sized value, so calling a
//! primitive needs one in hand: the constructor of an ISA that is not
//! enabled at compile time is where its availability gets established.

use super::bitmask::VECTOR_BYTES;
use super::lanes::Lanes;

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
}

/// Whatever the build targets: each primitive takes the widest
/// implementation `cfg(target_feature)` enables, so the token is free to
/// construct.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CompileTime;

impl Isa for CompileTime {
    type Lanes = Lanes;

    #[inline(always)]
    fn load(self, input: &[u8; VECTOR_BYTES]) -> Lanes {
        Lanes::load(input)
    }

    #[inline(always)]
    fn eq_mask(self, v: Lanes, b: u8) -> u64 {
        v.eq_mask(b)
    }

    #[inline(always)]
    fn xor_prefix_sum(self, q: u64) -> u64 {
        super::bitmask::xor_prefix_sum(q)
    }

    #[inline(always)]
    fn pext(self, src: u64, mask: u64) -> u64 {
        super::bitops::pext(src, mask)
    }
}
