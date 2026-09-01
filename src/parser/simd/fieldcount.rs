//! Per-vector field-count verification: extract a boundary-class bitmask
//! `P` (0 delimiter, 1 terminator) and compare it against a pre-built
//! rotation table for the running phase.

use super::bitops::pext;

/// Field-count verifier, one per chunk parse. Supports `1 ≤ N < 64`;
/// wider records route to the DFA backend.
pub(super) struct FieldCounter {
    /// Fields per record.
    n: u32,
    /// Expected `P` pattern per starting phase; `& 63` keeps the load unchecked.
    rotations: [u64; 64],
    /// Boundaries of the current record already seen; `0 ≤ phase < n`.
    phase: u32,
}

impl FieldCounter {
    /// Build a verifier for records with exactly `n` fields.
    pub(super) fn new(n: u32) -> Self {
        assert!(
            (1..64).contains(&n),
            "FieldCounter currently supports only 1 ≤ N < 64 (got {n})",
        );
        let mut rotations = [0u64; 64];
        for p in 0..n {
            rotations[p as usize] = expected_pattern(n, p);
        }
        Self {
            n,
            rotations,
            phase: 0,
        }
    }

    /// Verify one vector's boundary pattern: PEXT `term` at the positions
    /// `d_struct` selects, then compare the resulting `P` against this
    /// phase's rotation, masked to `popcnt(d_struct)` bits.
    #[inline]
    pub(super) fn verify(&mut self, d_struct: u64, term: u64) -> bool {
        let count = d_struct.count_ones();
        if count == 0 {
            return true;
        }
        let p = pext(term, d_struct);
        let expected = self.rotations[(self.phase & 63) as usize];
        let mask = if count == 64 {
            !0u64
        } else {
            (1u64 << count) - 1
        };
        if (p ^ expected) & mask != 0 {
            return false;
        }
        let carry = if p == 0 { self.phase } else { 0 };
        self.phase = count + p.leading_zeros() - 64 + carry;
        true
    }

    /// Fields-per-record this counter validates.
    pub(super) fn n(&self) -> usize {
        self.n as usize
    }

    /// Current running phase, exposed for tests.
    #[cfg(test)]
    pub(super) fn phase(&self) -> u32 {
        self.phase
    }
}

/// Pin down the record the verifier rejected, as `(record_start, found)`,
/// by walking the bytes from `from` — which must start an already
/// accepted record. `found` is a lower bound past the end of `buf`.
pub(super) fn locate_bad_record(
    buf: &[u8],
    from: usize,
    n: usize,
    delim: u8,
    term: u8,
    term_b: Option<u8>,
    quote: Option<u8>,
) -> (usize, usize) {
    let is_term = |c: u8| c == term || term_b == Some(c);
    let mut i = from.min(buf.len());
    let mut start = i;
    // Fields = delimiters + 1, so an empty record already counts as one.
    let mut fields = 1usize;
    let mut in_quotes = false;

    while i < buf.len() {
        let c = buf[i];
        if Some(c) == quote {
            in_quotes = !in_quotes;
        } else if !in_quotes {
            if c == delim {
                fields += 1;
            } else if is_term(c) {
                if fields != n {
                    return (start, fields);
                }
                // Collapse the run so `start` lands on real content.
                i += 1;
                while i < buf.len() && is_term(buf[i]) {
                    i += 1;
                }
                start = i;
                fields = 1;
                continue;
            }
        }
        i += 1;
    }

    // Out of buffer mid-record: only reachable with too *many* fields.
    debug_assert!(
        fields != n,
        "locate_bad_record found no mismatch at or after {from}",
    );
    (start, fields)
}

/// Expected-`P` pattern for a vector starting at `phase` with `n` fields
/// per record: bit `i` is set iff `(i + phase) mod n == n - 1`, i.e. the
/// `i`-th boundary completes a record.
#[inline]
fn expected_pattern(n: u32, phase: u32) -> u64 {
    let mut out = 0u64;
    let target = (n - 1) as usize;
    let phase = phase as usize;
    let n = n as usize;
    let mut i = 0usize;
    while i < 64 {
        if (i + phase) % n == target {
            out |= 1u64 << i;
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a u64 from a list of set-bit positions.
    fn bits(positions: &[u32]) -> u64 {
        let mut m = 0u64;
        for &p in positions {
            assert!(p < 64);
            m |= 1u64 << p;
        }
        m
    }

    // ── expected_pattern ──────────────────────────────────────────

    #[test]
    fn expected_pattern_n1_all_terminators() {
        // N=1: every boundary is a record terminator.
        assert_eq!(expected_pattern(1, 0), !0u64);
    }

    #[test]
    fn expected_pattern_n3_phase0() {
        // N=3: term at boundaries 2, 5, 8, ..., 62.
        let expected: Vec<u32> = (2..64).step_by(3).collect();
        assert_eq!(expected_pattern(3, 0), bits(&expected));
    }

    #[test]
    fn expected_pattern_n3_phase1() {
        // Phase 1: one boundary seen, so terms at 1, 4, 7, ..., 61.
        let expected: Vec<u32> = (1..64).step_by(3).collect();
        assert_eq!(expected_pattern(3, 1), bits(&expected));
    }

    #[test]
    fn expected_pattern_n5_phase1_includes_msb() {
        // Bits where (i+1) mod 5 == 4, i.e. i mod 5 == 3 — bit 63 included.
        let expected: Vec<u32> = (3..64).step_by(5).collect();
        let pat = expected_pattern(5, 1);
        assert_eq!(pat, bits(&expected));
        // Catches the off-by-one a plain u64 rotation would introduce.
        assert_ne!(pat & (1u64 << 63), 0);
    }

    // ── FieldCounter ──────────────────────────────────────────────

    #[test]
    fn verify_n1_all_term_passes() {
        // N=1 means every boundary is a terminator. D=term must hold.
        let mut fc = FieldCounter::new(1);
        let d = bits(&[2, 5, 8]);
        let term = d;
        assert!(fc.verify(d, term));
    }

    #[test]
    fn verify_n3_clean_record_passes() {
        // `a,b,c\nd,e,f\n`: boundaries {1, 3, 5, 7, 9, 11}, terms {5, 11}.
        let mut fc = FieldCounter::new(3);
        let d = bits(&[1, 3, 5, 7, 9, 11]);
        let term = bits(&[5, 11]);
        assert!(fc.verify(d, term));
        // 6 boundaries, N=3 → phase wraps to 0.
        assert_eq!(fc.phase(), 0);
    }

    #[test]
    fn verify_n3_partial_record_advances_phase() {
        // One full record + one delim of the next: phase ends at 1.
        let mut fc = FieldCounter::new(3);
        let d = bits(&[1, 3, 5, 7]);
        let term = bits(&[5]);
        assert!(fc.verify(d, term));
        assert_eq!(fc.phase(), 1);
    }

    #[test]
    fn verify_phase_threads_across_vectors() {
        // V1 = `a,b,` leaves phase 2; V2 = `c\n` brings the awaited term.
        let mut fc = FieldCounter::new(3);

        // V1: boundaries {1, 3}, no terminator → P = 0b00.
        let d1 = bits(&[1, 3]);
        assert!(fc.verify(d1, 0));
        assert_eq!(fc.phase(), 2);

        // V2: one boundary, a terminator → P = 0b1, as phase 2 expects.
        let d2 = bits(&[1]);
        let term2 = bits(&[1]);
        assert!(fc.verify(d2, term2));
        // The record completes, so the phase wraps to 0.
        assert_eq!(fc.phase(), 0);
    }

    #[test]
    fn verify_missing_terminator_fails() {
        // N=3 expects a terminator at boundary 2, but we give all delims.
        let mut fc = FieldCounter::new(3);
        let d = bits(&[1, 3, 5]);
        let term = 0;
        assert!(!fc.verify(d, term));
    }

    #[test]
    fn verify_misplaced_terminator_fails() {
        // N=3 expects the term at boundary 2, not boundary 0.
        let mut fc = FieldCounter::new(3);
        let d = bits(&[1, 3, 5]);
        let term = bits(&[1]);
        assert!(!fc.verify(d, term));
    }

    #[test]
    fn verify_extra_terminator_fails() {
        // Two terminators where only one is expected.
        let mut fc = FieldCounter::new(3);
        let d = bits(&[1, 3, 5]);
        let term = bits(&[3, 5]);
        assert!(!fc.verify(d, term));
    }

    #[test]
    fn verify_empty_vector_is_a_noop() {
        // No boundaries → no work, phase unchanged.
        let mut fc = FieldCounter::new(5);
        // Bring phase to a non-zero value first.
        let _ = fc.verify(bits(&[1, 3]), 0);
        let phase_before = fc.phase();
        assert!(fc.verify(0, 0));
        assert_eq!(fc.phase(), phase_before);
    }

    // ── locate_bad_record ─────────────────────────────────────────

    /// `locate_bad_record` with the default dialect: `,` / `\n` / `"`.
    fn locate(buf: &[u8], from: usize, n: usize) -> (usize, usize) {
        locate_bad_record(buf, from, n, b',', b'\n', None, Some(b'"'))
    }

    #[test]
    fn locate_finds_a_short_record() {
        let buf = b"aa,bb,cc\nxx,yy\ndd,ee,ff\n";
        assert_eq!(locate(buf, 0, 3), (9, 2));
    }

    #[test]
    fn locate_finds_a_long_record() {
        // `found` counts the whole record, not just up to the mismatch.
        let buf = b"aa,bb,cc\nww,xx,yy,zz\ndd,ee,ff\n";
        assert_eq!(locate(buf, 0, 3), (9, 4));
    }

    #[test]
    fn locate_skips_quoted_delimiters_and_terminators() {
        // None of the quoted `,`, `\n` or `""` may count as a boundary.
        let buf = b"\"a\nb\",\"c,d\",\"e\"\"f\"\nxx,yy\n";
        let bad = buf.len() - 6;
        assert_eq!(locate(buf, 0, 3), (bad, 2));
    }

    #[test]
    fn locate_collapses_terminator_runs() {
        // Blank lines aren't records: the bad one starts past the run.
        let buf = b"aa,bb,cc\n\n\n\nxx,yy\n";
        assert_eq!(locate(buf, 0, 3), (12, 2));
    }

    #[test]
    fn locate_starts_from_the_given_anchor() {
        // The anchor skips a record that would otherwise look short.
        let buf = b"xx,yy\naa,bb,cc\npp,qq\n";
        assert_eq!(locate(buf, 6, 3), (15, 2));
    }

    #[test]
    fn locate_reports_a_lower_bound_past_the_buffer() {
        // Buffer ends mid-record, so `found` is what has been read.
        let buf = b"aa,bb,cc\nww,xx,yy,zz";
        assert_eq!(locate(buf, 0, 3), (9, 4));
    }

    #[test]
    fn locate_handles_a_final_record_without_a_terminator() {
        let buf = b"aa,bb,cc\nxx,yy";
        assert_eq!(locate(buf, 0, 3), (9, 2));
    }

    #[test]
    fn locate_without_a_quote_char_treats_quotes_as_content() {
        // Quote-free: `"` is content, so the `,` in `"x,1"` separates.
        let buf = b"aa,bb,cc\n\"x,1\",yy,zz\n";
        assert_eq!(
            locate_bad_record(buf, 0, 3, b',', b'\n', None, None),
            (9, 4)
        );
        // With quoting on, `"x,1"` is one field, so this record is short.
        let buf = b"aa,bb,cc\n\"x,1\",yy\n";
        assert_eq!(locate(buf, 0, 3), (9, 2));
    }

    #[test]
    #[should_panic(expected = "1 ≤ N < 64")]
    fn ctor_rejects_n_zero() {
        let _ = FieldCounter::new(0);
    }

    #[test]
    #[should_panic(expected = "1 ≤ N < 64")]
    fn ctor_rejects_n_64() {
        let _ = FieldCounter::new(64);
    }
}
