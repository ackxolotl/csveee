//! Bit operations like pext and pdep.

/// Parallel bit extract via the widest implementation `cfg(target_feature)`
/// enables: gather the bits of `src` selected by `mask` into the low bits.
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline]
pub(super) fn pext(src: u64, mask: u64) -> u64 {
    // SAFETY: cfg gate guarantees the BMI2 target feature is available.
    unsafe { pext_bmi2(src, mask) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
#[inline]
pub(super) fn pext(src: u64, mask: u64) -> u64 {
    pext_scalar(src, mask)
}

/// [`pext`] as the BMI2 instruction.
///
/// # Safety
/// The CPU must support BMI2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
pub(super) unsafe fn pext_bmi2(src: u64, mask: u64) -> u64 {
    std::arch::x86_64::_pext_u64(src, mask)
}

/// [`pext`] one selected bit at a time.
#[allow(dead_code)] // live without BMI2; also the tests' oracle
#[inline]
fn pext_scalar(src: u64, mut mask: u64) -> u64 {
    let mut result = 0u64;
    let mut out_pos: u32 = 0;
    while mask != 0 {
        // use `u64::isolate_lowest_one` once we are at Rust >=1.97
        let lo = mask & mask.wrapping_neg();
        if src & lo != 0 {
            result |= 1u64 << out_pos;
        }
        mask &= mask - 1;
        out_pos += 1;
    }
    result
}

#[allow(dead_code)]
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline]
pub(super) fn pdep(src: u64, mask: u64) -> u64 {
    // SAFETY: cfg gate guarantees the BMI2 target feature is available.
    unsafe { std::arch::x86_64::_pdep_u64(src, mask) }
}

#[allow(dead_code)]
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
#[inline]
pub(super) fn pdep(src: u64, mut mask: u64) -> u64 {
    let mut result = 0u64;
    let mut in_pos: u32 = 0;
    while mask != 0 {
        // use `u64::isolate_lowest_one` once we are at Rust >=1.97
        let lo = mask & mask.wrapping_neg();
        if (src >> in_pos) & 1 != 0 {
            result |= lo;
        }
        mask &= mask - 1;
        in_pos += 1;
    }
    result
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn bmi2_pext_matches_scalar() {
        if !is_x86_feature_detected!("bmi2") {
            return;
        }
        let words = [
            0,
            !0,
            1,
            1 << 63,
            0xAAAA_AAAA_AAAA_AAAA,
            0xDEAD_BEEF_CAFE_F00D,
        ];
        for &src in &words {
            for &mask in &words {
                // SAFETY: BMI2 was just detected.
                let hw = unsafe { pext_bmi2(src, mask) };
                assert_eq!(hw, pext_scalar(src, mask), "src {src:#x}, mask {mask:#x}");
            }
        }
    }
}
