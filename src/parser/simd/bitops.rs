//! Bit operations like pext and pdep.

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline]
pub(super) fn pext(src: u64, mask: u64) -> u64 {
    // SAFETY: cfg gate guarantees the BMI2 target feature is available.
    unsafe { std::arch::x86_64::_pext_u64(src, mask) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
#[inline]
pub(super) fn pext(src: u64, mut mask: u64) -> u64 {
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
