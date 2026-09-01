//! Bitmask construction: 64-byte input → u64 structural bitmasks, bit
//! `i` per byte `i`. See the paper's §Vectorization for the math and
//! [`super::scan`] for how the outputs become B/E/R indices.

use std::simd::Simd;
use std::simd::cmp::SimdPartialEq;

/// Vector width in bytes — one cacheline.
pub(super) const VECTOR_BYTES: usize = 64;

type ByteVec = Simd<u8, VECTOR_BYTES>;

/// Per-vector structural-character bitmasks; bit `i` is byte `i` of the
/// input vector. The byte values must be distinct (`Config::validate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Structural {
    /// `C` — field delimiters.
    pub delim: u64,
    /// `N` — record terminators, both bytes' matches for CRLF dialects.
    pub term: u64,
    /// `Q` — quote characters. `0` when quoting is disabled.
    pub quote: u64,
}

impl Structural {
    /// `D` — delimiters or terminators (mask of all field-boundary chars).
    #[inline]
    pub fn delim_or_term(&self) -> u64 {
        self.delim | self.term
    }

    /// `D_struct = (C | N) & !M` — delimiters/terminators outside quotes.
    #[inline]
    pub fn structural_delims(&self, m: u64) -> u64 {
        self.delim_or_term() & !m
    }
}

/// Match structural characters in a 64-byte vector, yielding the
/// `(C, N, Q)` masks (`Q = 0` without quoting). `term2` — the `\r` of a
/// CRLF dialect — folds into `term`.
#[inline]
pub(super) fn match_structural(
    input: &[u8; VECTOR_BYTES],
    delim: u8,
    term: u8,
    term2: Option<u8>,
    quote: Option<u8>,
) -> Structural {
    let v = ByteVec::from_array(*input);
    let delim_mask = v.simd_eq(ByteVec::splat(delim)).to_bitmask();
    let mut term_mask = v.simd_eq(ByteVec::splat(term)).to_bitmask();
    if let Some(t2) = term2 {
        term_mask |= v.simd_eq(ByteVec::splat(t2)).to_bitmask();
    }
    let quote_mask = match quote {
        Some(q) => v.simd_eq(ByteVec::splat(q)).to_bitmask(),
        None => 0,
    };
    Structural {
        delim: delim_mask,
        term: term_mask,
        quote: quote_mask,
    }
}

/// Pick a byte that is neither delimiter, terminator nor quote, so that
/// padding a partial vector with it synthesizes no structural bits.
pub(super) fn neutral_pad(delim: u8, term: u8, term2: Option<u8>, quote: Option<u8>) -> u8 {
    for candidate in 0u8..=255u8 {
        if candidate == delim || candidate == term {
            continue;
        }
        if term2 == Some(candidate) || quote == Some(candidate) {
            continue;
        }
        return candidate;
    }
    unreachable!("at most four structural bytes; 256 candidates suffice");
}

/// Why [`head_padded_prologue`] did or didn't build a vector; the two
/// declines differ, so the caller must not conflate them.
pub(super) enum Prologue {
    /// Head-padded first vector, and the misalignment its pad absorbs.
    Vector([u8; VECTOR_BYTES], usize),
    /// Already 64-byte aligned; scan straight from byte 0.
    Aligned,
    /// Fewer than `skip + 64` bytes: no aligned vector follows yet.
    TooShort,
}

/// Build a head-padded first vector so every later load is 64-byte
/// aligned: `misalign` lanes of neutral pad put its virtual byte 0 at
/// `buf.as_ptr() - misalign`.
pub(super) fn head_padded_prologue(buf: &[u8], scan_end: usize, pad: u8) -> Prologue {
    let misalign = buf.as_ptr() as usize % VECTOR_BYTES;
    if misalign == 0 {
        return Prologue::Aligned;
    }
    let skip = VECTOR_BYTES - misalign;
    if scan_end < skip + VECTOR_BYTES {
        return Prologue::TooShort;
    }
    let mut v = [pad; VECTOR_BYTES];
    v[misalign..].copy_from_slice(&buf[..skip]);
    Prologue::Vector(v, misalign)
}

/// First terminator byte of each maximal run in the out-of-quotes mask
/// `t` — where the preceding field ends. `prev_is_term` is bit 63 of the
/// previous vector's `t`, so a straddling run isn't split.
#[inline]
pub(super) fn terminator_run_starts(t: u64, prev_is_term: bool) -> u64 {
    t & !((t << 1) | (prev_is_term as u64))
}

/// Last terminator byte of each maximal run — the next record begins one
/// past these. `next_term_lsb` is bit 0 of the next vector's `t`.
#[inline]
pub(super) fn terminator_run_ends(t: u64, next_term_lsb: bool) -> u64 {
    t & !((t >> 1) | ((next_term_lsb as u64) << 63))
}

/// Compute the in-quotes bitmask `M`, the XOR prefix sum of `Q`: set iff
/// the byte is inside a quoted field, opener included, closer excluded.
/// `carry_in` / `carry_out` are the previous / this vector's `M`-MSB.
#[inline]
pub(super) fn compute_in_quotes(quote: u64, carry_in: bool) -> (u64, bool) {
    // No quote bytes: `M` is the carry broadcast, skipping the pclmulqdq.
    if quote == 0 {
        let m = (carry_in as u64).wrapping_neg();
        return (m, carry_in);
    }
    let prefix = xor_prefix_sum(quote);
    // Branchless: false → 0, true → !0 (broadcast bit to whole word).
    let carry_mask = (carry_in as u64).wrapping_neg();
    let m = prefix ^ carry_mask;
    let carry_out = (m >> 63) != 0;
    (m, carry_out)
}

/// Forward carry into the next vector's `B` mask. Two channels because
/// `B` is built in two stages: `d_struct_msb` feeds `B_init[0]` (still
/// subject to the quote-promotion shift), `qb_msb` feeds `B_final[0]`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct BeginsCarry {
    /// `D_struct[63]`, feeding `B_init[0]` of the next vector.
    pub d_struct_msb: bool,
    /// `QB[63]`, feeding `B_final[0]` of the next vector.
    pub qb_msb: bool,
}

/// Compute the field-beginnings bitmask `B` — each field's first content
/// byte — from `B_init = D_struct << 1` and `QB = B_init & Q` as
/// `(B_init & !QB) | (QB << 1)`, plus carries. `B[0]` is the prologue's.
#[inline]
pub(super) fn compute_field_begins(
    d_struct: u64,
    quote: u64,
    carry_in: BeginsCarry,
) -> (u64, BeginsCarry) {
    let b_init = (d_struct << 1) | (carry_in.d_struct_msb as u64);
    let qb = b_init & quote;
    let b_final = ((b_init & !qb) | (qb << 1)) | (carry_in.qb_msb as u64);
    let carry_out = BeginsCarry {
        d_struct_msb: (d_struct >> 63) & 1 != 0,
        qb_msb: (qb >> 63) & 1 != 0,
    };
    (b_final, carry_out)
}

/// Compute the field-ends bitmask `E` — one past each field's last
/// content byte — as `D_struct` with a closing quote pulled back over the
/// delim after it. The neighbour bits are `false` at a chunk edge.
#[inline]
pub(super) fn compute_field_ends(
    d_struct: u64,
    quote: u64,
    m: u64,
    next_d_struct_lsb: bool,
    prev_qe_msb: bool,
) -> (u64, bool) {
    let e_init = d_struct;
    let qr = quote & !m;
    let d_struct_lookahead = (d_struct >> 1) | ((next_d_struct_lsb as u64) << 63);
    let qe = qr & d_struct_lookahead;
    let qe_clear = (qe << 1) | (prev_qe_msb as u64);
    let e = (e_init & !qe_clear) | qe;
    let qe_msb = (qe >> 63) & 1 != 0;
    (e, qe_msb)
}

/// Compute the chars-to-remove bitmask `R`: every quote elided from field
/// content, keeping those `B`/`E` already account for and, under
/// `doubled_quotes`, the first half of a `""` pair.
#[inline]
pub(super) fn compute_chars_to_remove(
    quote: u64,
    m: u64,
    b: u64,
    e: u64,
    next_b_lsb: bool,
    next_q_lsb: bool,
    doubled_quotes: bool,
) -> u64 {
    let ql = quote & m;
    let qr = quote & !m;
    let b_lookahead = (b >> 1) | ((next_b_lsb as u64) << 63);
    let ql_remove = ql & !b_lookahead;
    let qr_remove = if doubled_quotes {
        let q_lookahead = (quote >> 1) | ((next_q_lsb as u64) << 63);
        qr & !q_lookahead & !e
    } else {
        qr & !e
    };
    ql_remove | qr_remove
}

/// XOR prefix sum of `q`: bit `i` is the XOR of input bits `0..=i`.
/// Equivalent to a GF(2) multiply by `~0u64` — one `pclmulqdq` on x86_64,
/// one `pmull` on aarch64, else a six-step scalar prefix.
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline]
fn xor_prefix_sum(q: u64) -> u64 {
    use std::arch::x86_64::{
        _mm_clmulepi64_si128, _mm_cvtsi64_si128, _mm_cvtsi128_si64, _mm_set1_epi64x,
    };
    // SAFETY: the cfg gate guarantees `pclmulqdq` on this target; the
    // intrinsics take only register-resident scalars.
    unsafe {
        let q_v = _mm_cvtsi64_si128(q as i64);
        let ones = _mm_set1_epi64x(-1);
        let result = _mm_clmulepi64_si128(q_v, ones, 0);
        _mm_cvtsi128_si64(result) as u64
    }
}

/// `pmull` sits in the `aes` feature, which is on by default for
/// `aarch64-apple-darwin` and any target-cpu with the crypto extensions.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
fn xor_prefix_sum(q: u64) -> u64 {
    // SAFETY: the cfg gate guarantees `aes` on this target; the intrinsic
    // takes only register-resident scalars.
    unsafe { std::arch::aarch64::vmull_p64(q, !0) as u64 }
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    all(target_arch = "aarch64", target_feature = "aes"),
)))]
#[inline]
fn xor_prefix_sum(q: u64) -> u64 {
    xor_prefix_sum_scalar(q)
}

/// Scalar fallback: six doubling steps cover all 64 bits.
#[allow(dead_code)] // live on targets without a clmul; also the tests' oracle
#[inline]
fn xor_prefix_sum_scalar(mut q: u64) -> u64 {
    q ^= q << 1;
    q ^= q << 2;
    q ^= q << 4;
    q ^= q << 8;
    q ^= q << 16;
    q ^= q << 32;
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 64-byte input from a shorter slice, padding with `pad`.
    fn pad64(prefix: &[u8], pad: u8) -> [u8; VECTOR_BYTES] {
        assert!(prefix.len() <= VECTOR_BYTES);
        let mut buf = [pad; VECTOR_BYTES];
        buf[..prefix.len()].copy_from_slice(prefix);
        buf
    }

    /// Build a u64 bitmask from a list of set-bit positions.
    fn bits(positions: &[u32]) -> u64 {
        let mut m = 0u64;
        for &p in positions {
            assert!(p < 64);
            m |= 1u64 << p;
        }
        m
    }

    /// Reference XOR prefix sum: a bit-by-bit O(n²) oracle.
    fn xor_prefix_sum_brute(q: u64) -> u64 {
        let mut acc = 0u64;
        let mut result = 0u64;
        for i in 0..64 {
            acc ^= (q >> i) & 1;
            result |= acc << i;
        }
        result
    }

    #[test]
    fn empty_input_zero_masks() {
        // All-`x` input has no structurals.
        let input = [b'x'; VECTOR_BYTES];
        let s = match_structural(&input, b',', b'\n', None, Some(b'"'));
        assert_eq!(s.delim, 0);
        assert_eq!(s.term, 0);
        assert_eq!(s.quote, 0);
    }

    #[test]
    fn single_delimiter_at_byte_zero() {
        // Lane-0 → bit-0 ordering check.
        let input = pad64(b",xxx", b'x');
        let s = match_structural(&input, b',', b'\n', None, Some(b'"'));
        assert_eq!(s.delim, 1);
        assert_eq!(s.term, 0);
        assert_eq!(s.quote, 0);
    }

    #[test]
    fn delimiters_at_arbitrary_positions() {
        let input = pad64(b"a,b,c,d", b'.');
        let s = match_structural(&input, b',', b'\n', None, None);
        assert_eq!(s.delim, bits(&[1, 3, 5]));
        assert_eq!(s.term, 0);
        assert_eq!(s.quote, 0);
    }

    #[test]
    fn terminator_byte_separately_classified() {
        let input = pad64(b"a,b\nc", b'.');
        let s = match_structural(&input, b',', b'\n', None, None);
        assert_eq!(s.delim, bits(&[1]));
        assert_eq!(s.term, bits(&[3]));
    }

    #[test]
    fn quote_byte_classified_when_enabled() {
        let input = pad64(b"\"a\",b", b'.');
        let s = match_structural(&input, b',', b'\n', None, Some(b'"'));
        assert_eq!(s.quote, bits(&[0, 2]));
        assert_eq!(s.delim, bits(&[3]));
    }

    #[test]
    fn quote_byte_ignored_when_disabled() {
        // Even though `"` is in the input, no quote config means Q = 0.
        let input = pad64(b"\"hello\"", b'.');
        let s = match_structural(&input, b',', b'\n', None, None);
        assert_eq!(s.quote, 0);
    }

    #[test]
    fn structurals_at_last_byte() {
        // Bit 63 must be set — boundary for `to_bitmask` width.
        let mut input = [b'.'; VECTOR_BYTES];
        input[63] = b',';
        let s = match_structural(&input, b',', b'\n', None, None);
        assert_eq!(s.delim, 1u64 << 63);
    }

    #[test]
    fn full_vector_of_delimiters() {
        let input = [b','; VECTOR_BYTES];
        let s = match_structural(&input, b',', b'\n', None, None);
        assert_eq!(s.delim, !0u64);
        assert_eq!(s.term, 0);
    }

    #[test]
    fn design_example_first_vector() {
        // Running example: `Tom,"5'11"", Chicago",28\nAmy`.
        // Length 28; pad rest with '.'.
        let raw = b"Tom,\"5'11\"\", Chicago\",28\nAmy";
        let input = pad64(raw, b'.');
        let s = match_structural(&input, b',', b'\n', None, Some(b'"'));
        // Commas at offsets 3, 11, 21.
        assert_eq!(s.delim, bits(&[3, 11, 21]));
        // Newline at offset 24.
        assert_eq!(s.term, bits(&[24]));
        // Quotes at offsets 4, 9, 10, 20.
        assert_eq!(s.quote, bits(&[4, 9, 10, 20]));
    }

    #[test]
    fn delim_or_term_combines_masks() {
        let input = pad64(b"a,b\nc,d", b'.');
        let s = match_structural(&input, b',', b'\n', None, None);
        // Commas at 1, 5; newline at 3.
        assert_eq!(s.delim_or_term(), bits(&[1, 3, 5]));
    }

    // ── XOR prefix sum primitives ─────────────────────────────────

    /// Check one implementation against the oracle: hand-picked patterns
    /// plus a walking-bit sweep for off-by-ones at the word boundaries.
    fn check_prefix_sum(f: impl Fn(u64) -> u64, name: &str) {
        let cases = [
            0u64,
            !0u64,
            1u64,
            1u64 << 63,
            0xAAAA_AAAA_AAAA_AAAAu64,
            0x5555_5555_5555_5555u64,
            0xDEAD_BEEF_CAFE_F00Du64,
        ];
        for &q in &cases {
            assert_eq!(
                f(q),
                xor_prefix_sum_brute(q),
                "{name} mismatch for q={q:#018x}"
            );
        }
        for i in 0..64 {
            let q = 1u64 << i;
            assert_eq!(
                f(q),
                xor_prefix_sum_brute(q),
                "{name} walking-bit mismatch at i={i}"
            );
        }
    }

    #[test]
    fn scalar_prefix_sum_matches_brute_force() {
        check_prefix_sum(xor_prefix_sum_scalar, "scalar");
    }

    /// Whichever arch arm `xor_prefix_sum` resolved to here — `pclmulqdq`,
    /// `pmull`, or the scalar fallback.
    #[test]
    fn dispatched_prefix_sum_matches_brute_force() {
        check_prefix_sum(xor_prefix_sum, "dispatched");
    }

    // ── compute_in_quotes ─────────────────────────────────────────

    #[test]
    fn no_quotes_no_carry_zero_mask() {
        let (m, carry) = compute_in_quotes(0, false);
        assert_eq!(m, 0);
        assert!(!carry);
    }

    #[test]
    fn carry_in_with_no_quotes_flips_to_all_ones() {
        // Previous vector ended in-quotes, no quotes here → all in-quotes.
        let (m, carry) = compute_in_quotes(0, true);
        assert_eq!(m, !0u64);
        assert!(carry);
    }

    #[test]
    fn opening_quote_only_extends_to_msb() {
        // Single open quote at position 0, nothing else.
        let (m, carry) = compute_in_quotes(1u64, false);
        assert_eq!(m, !0u64);
        assert!(carry);
    }

    #[test]
    fn open_close_pair_marks_interior_only() {
        // Open at 0, close at 5: M includes the opener, excludes the closer.
        let q = bits(&[0, 5]);
        let (m, carry) = compute_in_quotes(q, false);
        assert_eq!(m, bits(&[0, 1, 2, 3, 4]));
        assert!(!carry);
    }

    #[test]
    fn quote_at_msb_with_open_state_carries_out() {
        // Open at 0, no close → we exit the vector still in-quotes.
        let (_m, carry) = compute_in_quotes(1u64, false);
        assert!(carry);
    }

    #[test]
    fn carry_in_with_closing_quote_clears_to_zero() {
        // Continues a quoted field, closing at 5 → bits 0..=4 in-quotes.
        let q = 1u64 << 5;
        let (m, carry) = compute_in_quotes(q, true);
        assert_eq!(m, bits(&[0, 1, 2, 3, 4]));
        assert!(!carry);
    }

    #[test]
    fn design_example_in_quotes_mask() {
        // Running example: quotes at 4, 9, 10, 20 → runs [4..=8], [10..=19].
        let q = bits(&[4, 9, 10, 20]);
        let (m, carry) = compute_in_quotes(q, false);
        let expected = bits(&[4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
        assert_eq!(m, expected);
        assert!(!carry);
    }

    #[test]
    fn carry_round_trip_across_two_vectors() {
        // V1 opens at 60 with no close; V2 closes at 5.
        let q1 = 1u64 << 60;
        let (m1, carry1) = compute_in_quotes(q1, false);
        // V1: bits 60..=63 in-quotes.
        assert_eq!(m1, bits(&[60, 61, 62, 63]));
        assert!(carry1);

        let q2 = 1u64 << 5;
        let (m2, carry2) = compute_in_quotes(q2, carry1);
        // V2: bits 0..=4 in-quotes, 5..=63 out.
        assert_eq!(m2, bits(&[0, 1, 2, 3, 4]));
        assert!(!carry2);
    }

    // ── structural_delims ─────────────────────────────────────────

    #[test]
    fn structural_filters_quoted_delims() {
        // The comma at 6 is inside the quoted run [4, 9), so it is content.
        let s = Structural {
            delim: bits(&[1, 6, 12]),
            term: bits(&[15]),
            quote: bits(&[4, 9]),
        };
        let (m, _) = compute_in_quotes(s.quote, false);
        // M covers 4..=8.
        assert_eq!(s.structural_delims(m), bits(&[1, 12, 15]));
    }

    // ── compute_field_ends ────────────────────────────────────────

    #[test]
    fn ends_with_no_quotes_match_delim_positions() {
        // Pure unquoted record `a,b\nc`: D_struct = {1, 3}.
        let d = bits(&[1, 3]);
        let (e, qe_msb) = compute_field_ends(d, 0, 0, false, false);
        assert_eq!(e, bits(&[1, 3]));
        assert!(!qe_msb);
    }

    #[test]
    fn ends_pull_back_to_closing_quote() {
        // `"x",y`: Q at {0, 2}, M at {0, 1}, D_struct = {3}, so QE = {2}
        // and E moves from the delim at 3 to the closing quote at 2.
        let d_struct = bits(&[3]);
        let m = bits(&[0, 1]);
        let q = bits(&[0, 2]);
        let (e, qe_msb) = compute_field_ends(d_struct, q, m, false, false);
        assert_eq!(e, bits(&[2]));
        assert!(!qe_msb);
    }

    #[test]
    fn ends_lookahead_into_next_vector() {
        // Closing quote at byte 63, delimiter at byte 0 of the next
        // vector: the lookahead sets E[63] and carries qe_msb forward.
        let q = 1u64 << 63;
        // M[63] = 0 makes Q[63] a closer; D_struct is in the next vector.
        let m = !(1u64 << 63);
        let (e, qe_msb) = compute_field_ends(0, q, m, true, false);
        assert_eq!(e, 1u64 << 63);
        assert!(qe_msb, "qe_msb must propagate forward");
    }

    #[test]
    fn ends_prev_qe_msb_clears_bit_zero() {
        // With `prev_qe_msb`, the delim at byte 0 must not produce a
        // second E for the boundary already reported by vector i.
        let d_struct = 1u64; // comma at byte 0
        let (e, qe_msb) = compute_field_ends(d_struct, 0, 0, false, true);
        assert_eq!(e, 0, "e_init[0] must be suppressed by prev_qe_msb");
        assert!(!qe_msb);
    }

    #[test]
    fn ends_design_example() {
        // Running example: Q at {4, 9, 10, 20}, (C|N) at {3, 11, 21, 24}.
        let q = bits(&[4, 9, 10, 20]);
        let cn = bits(&[3, 11, 21, 24]);
        let (m, _) = compute_in_quotes(q, false);
        // Bit 11 falls inside the run [10..=19], so D_struct strips it.
        let d_struct = cn & !m;
        assert_eq!(d_struct, bits(&[3, 21, 24]));
        let (e, _) = compute_field_ends(d_struct, q, m, false, false);
        // Design table: E set at {3, 20, 24}.
        assert_eq!(e, bits(&[3, 20, 24]));
    }

    // ── compute_field_begins ──────────────────────────────────────

    #[test]
    fn begins_shift_unquoted_delims_by_one() {
        // `a,b,c`: D_struct = {1, 3} → B = B_init = {2, 4}.
        let d_struct = bits(&[1, 3]);
        let (b, carry) = compute_field_begins(d_struct, 0, BeginsCarry::default());
        assert_eq!(b, bits(&[2, 4]));
        assert_eq!(carry, BeginsCarry::default());
    }

    #[test]
    fn begins_skip_opening_quote() {
        // Delim at 1, quote at 2 → field starts at byte 3, not byte 2.
        let d_struct = bits(&[1]);
        let q = bits(&[2]);
        let (b, _) = compute_field_begins(d_struct, q, BeginsCarry::default());
        assert_eq!(b, bits(&[3]));
    }

    #[test]
    fn begins_d_struct_msb_carries_forward() {
        // Delim at byte 63: B[0] of next vector should be set.
        let d_struct = 1u64 << 63;
        let (b, carry) = compute_field_begins(d_struct, 0, BeginsCarry::default());
        // No bit set within this vector's B (the shift carries out).
        assert_eq!(b, 0);
        assert!(carry.d_struct_msb);
        assert!(!carry.qb_msb);

        // Next vector with no delims/quotes: B[0] = 1 from carry.
        let (b_next, _) = compute_field_begins(0, 0, carry);
        assert_eq!(b_next, 1u64);
    }

    #[test]
    fn begins_d_struct_msb_carry_skips_quote_at_next_zero() {
        // Delim at 63, quote at 0 of i+1 → the field starts at byte 1.
        let d_struct_i = 1u64 << 63;
        let (_, carry) = compute_field_begins(d_struct_i, 0, BeginsCarry::default());
        let q_next = 1u64;
        let (b_next, _) = compute_field_begins(0, q_next, carry);
        assert_eq!(b_next, 1u64 << 1);
    }

    #[test]
    fn begins_qb_msb_carries_forward() {
        // Delim at 62, quote at 63 → the field starts at byte 0 of i+1.
        let d_struct_i = 1u64 << 62;
        let q_i = 1u64 << 63;
        let (b_i, carry) = compute_field_begins(d_struct_i, q_i, BeginsCarry::default());
        // Bit 63 was a quote at a field beginning, so the shift carries out.
        assert_eq!(b_i, 0);
        assert!(!carry.d_struct_msb);
        assert!(carry.qb_msb);

        // Next vector: B[0] = 1 (post-quote, content begins).
        let (b_next, _) = compute_field_begins(0, 0, carry);
        assert_eq!(b_next, 1u64);
    }

    #[test]
    fn begins_design_example() {
        // Running example: (C|N) at {3, 11, 21, 24}, Q at {4, 9, 10, 20}.
        let q = bits(&[4, 9, 10, 20]);
        let cn = bits(&[3, 11, 21, 24]);
        let (m, _) = compute_in_quotes(q, false);
        let d_struct = cn & !m;
        let (b, _) = compute_field_begins(d_struct, q, BeginsCarry::default());
        // Design table B = {0, 5, 22, 25}; bit 0 is the prologue's.
        assert_eq!(b, bits(&[5, 22, 25]));
    }

    // ── compute_chars_to_remove ───────────────────────────────────

    #[test]
    fn remove_no_quotes_no_removal() {
        // No quotes anywhere → R = 0 regardless of B/E.
        let r = compute_chars_to_remove(0, 0, bits(&[5, 10]), bits(&[3, 9]), false, false, true);
        assert_eq!(r, 0);
    }

    #[test]
    fn remove_proper_quoted_field_keeps_both_quotes() {
        // `,"hi",` — opener at 1, closer at 4, both accounted for by B/E.
        let q = bits(&[1, 4]);
        let m = bits(&[1, 2, 3]);
        let b = bits(&[2, 6]);
        let e = bits(&[4]);
        let r = compute_chars_to_remove(q, m, b, e, false, false, true);
        assert_eq!(r, 0);
    }

    #[test]
    fn remove_mid_field_quotes_get_dropped() {
        // `John,say "hello" world,42` — the mid-field quotes at 9 and 15
        // are content under Postgres semantics, so both bytes go.
        let q = bits(&[9, 15]);
        let m = bits(&[9, 10, 11, 12, 13, 14]);
        // Field begins at byte 5 ("say") and 23 ("4"); ends at 4 and 22.
        let b = bits(&[5, 23]);
        let e = bits(&[4, 22]);
        let r = compute_chars_to_remove(q, m, b, e, false, false, true);
        assert_eq!(r, bits(&[9, 15]));
    }

    #[test]
    fn remove_doubled_quote_escape_drops_second_only() {
        // `,"a""b",`: 0=',', 1='"', 2='a', 3='"', 4='"', 5='b', 6='"', 7=','.
        let q = bits(&[1, 3, 4, 6]);
        let (m, _) = compute_in_quotes(q, false);
        assert_eq!(m, bits(&[1, 2, 4, 5]), "M setup sanity check");

        // QL = {1, 4}, QR = {3, 6}; the field is [2, 6).
        let b = bits(&[2]);
        let e = bits(&[6]);
        let r = compute_chars_to_remove(q, m, b, e, false, false, true);
        // Only QL[4] is mid-field: 1 opens, 3 is the pair's first half,
        // 6 is the field end.
        assert_eq!(r, bits(&[4]));
    }

    #[test]
    fn remove_lookahead_b_into_next_vector() {
        // An opening quote at byte 63 whose field begins at byte 0 of the
        // next vector must be kept, via `next_b_lsb`.
        let q = 1u64 << 63;
        let m = 1u64 << 63;
        let b = 0;
        let e = 0;
        let r = compute_chars_to_remove(q, m, b, e, true, false, true);
        assert_eq!(r, 0);
    }

    #[test]
    fn remove_lookahead_q_into_next_vector() {
        // An escape pair split across the boundary: the closer at byte 63
        // is the pair's first half, so `next_q_lsb` keeps it.
        let q = 1u64 << 63;
        let m = 0;
        let b = 0;
        let e = 0;
        let r = compute_chars_to_remove(q, m, b, e, false, true, true);
        assert_eq!(r, 0);
    }

    #[test]
    fn remove_design_example_full_pipeline() {
        // `Tom,"5'11"", Chicago",28\nAmy`: only the escape
        // pair's second quote, at 10, is removed.
        let q = bits(&[4, 9, 10, 20]);
        let cn = bits(&[3, 11, 21, 24]);
        let (m, _) = compute_in_quotes(q, false);
        let d_struct = cn & !m;
        let (b, _) = compute_field_begins(d_struct, q, BeginsCarry::default());
        let (e, _) = compute_field_ends(d_struct, q, m, false, false);
        // The prologue's chunk-start B[0], for the design's full B.
        let b_full = b | 1u64;
        let r = compute_chars_to_remove(q, m, b_full, e, false, false, true);
        assert_eq!(r, bits(&[10]));
    }

    #[test]
    fn remove_paired_quotes_under_pure_toggle_drops_both() {
        // The same `,"a""b",` layout under pure Toggle: every quote is
        // structural, so both bytes of the pair go.
        let q = bits(&[1, 3, 4, 6]);
        let (m, _) = compute_in_quotes(q, false);
        assert_eq!(m, bits(&[1, 2, 4, 5]), "M setup sanity check");
        let b = bits(&[2]);
        let e = bits(&[6]);
        let r = compute_chars_to_remove(q, m, b, e, false, false, false);
        // 1 opens the field and 6 ends it; 3 and 4 both go.
        assert_eq!(r, bits(&[3, 4]));
    }
}
