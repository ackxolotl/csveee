//! Cross-vector bitmask pipeline: vector `i`'s `E` and `R` depend on bit
//! 0 of vector `i+1`'s masks, so this lags a vector, owning that delay
//! and the forward carries.

use super::bitmask::{
    BeginsCarry, Structural, VECTOR_BYTES, compute_chars_to_remove, compute_field_begins,
    compute_field_ends, compute_in_quotes, match_structural, terminator_run_ends,
    terminator_run_starts,
};

/// Bit 0 of the next vector's `B` — the `next_b_lsb` that `R` needs.
/// Only `qb_msb` can reach an output bit (`d_struct_msb` rules out a quote
/// at byte 63), but dropping the dead term measured 2.5% slower on epyc5.
#[inline]
fn derive_b_lsb(carry: BeginsCarry, next_quote_lsb: bool) -> bool {
    (carry.d_struct_msb && !next_quote_lsb) || carry.qb_msb
}

/// The slice of the parser config [`Scanner`] cares about.
#[derive(Debug, Clone, Copy)]
pub(super) struct ScannerConfig {
    /// Field delimiter byte.
    pub delim: u8,
    /// Primary record terminator byte.
    pub term: u8,
    /// Second terminator byte, folded into `term` (the `\r` of a CRLF).
    pub term_b: Option<u8>,
    /// Quote byte, or `None` when quoting is disabled.
    pub quote: Option<u8>,
    /// `escape == quote`: the first `"` of a `""` pair is kept as content.
    pub doubled_quotes: bool,
}

/// Per-vector output: the three indexing bitmasks `b`/`e`/`r`, plus
/// `d_struct` and `term` for the field-count verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VectorOutput {
    /// Field beginnings.
    pub b: u64,
    /// Field ends.
    pub e: u64,
    /// Chars to remove.
    pub r: u64,
    /// Out-of-quotes delimiters and terminator-run ends, for the verifier.
    pub d_struct: u64,
    /// The run-collapsed terminator subset of `d_struct`.
    pub term: u64,
}

/// Stateful scanner: stream 64-byte input vectors in via [`Scanner::step`];
/// stream completed [`VectorOutput`]s out, lagging by one vector. End
/// the stream with [`Scanner::finalize`] to flush the pending vector.
pub(super) struct Scanner {
    /// Dialect bytes the per-vector masks are built from.
    config: ScannerConfig,
    /// `M`-MSB of the last vector stepped in.
    in_quotes_carry: bool,
    /// `B`-carries of the last resolved vector.
    begins_carry: BeginsCarry,
    /// `QE[63]` of the last emitted vector, so no field-end reports twice.
    ends_carry: bool,
    /// Bit 63 of the last-resolved `term` mask, so run continuations get no `E`.
    prev_term_msb: bool,
    /// The vector held back for the next call's lookahead.
    pending: Option<Pending>,
}

/// Raw masks of the previous vector, awaiting the next vector's LSBs.
/// `B`, `E`, `R` and the run-collapsed `d_struct` are all computed at
/// resolve time; the cascade stays at one vector — see [`derive_b_lsb`].
#[derive(Debug, Clone, Copy)]
struct Pending {
    /// Raw `(C, N, Q)` masks of the held vector.
    structural: Structural,
    /// Its in-quotes mask.
    m: u64,
}

/// The lookahead a pending vector resolves against — `None` at end of
/// input, where no byte follows byte 63.
#[derive(Debug, Clone, Copy)]
struct Lookahead {
    /// Out-of-quotes delimiters of the next vector.
    delim_ooq: u64,
    /// Its out-of-quotes terminators.
    term_ooq: u64,
    /// Its raw `Q`.
    quote: u64,
}

impl Scanner {
    pub(super) fn new(config: ScannerConfig) -> Self {
        Self {
            config,
            in_quotes_carry: false,
            begins_carry: BeginsCarry::default(),
            ends_carry: false,
            prev_term_msb: false,
            pending: None,
        }
    }

    /// Consume one 64-byte vector, returning the previous vector's
    /// completed output (`None` on a fresh scanner's first call).
    pub(super) fn step(&mut self, input: &[u8; VECTOR_BYTES]) -> Option<VectorOutput> {
        let cfg = self.config;
        let s = match_structural(input, cfg.delim, cfg.term, cfg.term_b, cfg.quote);
        let (m, in_quotes_carry_out) = compute_in_quotes(s.quote, self.in_quotes_carry);
        // The lookahead the pending (previous) vector needs to finalize.
        let next = Lookahead {
            delim_ooq: s.delim & !m,
            term_ooq: s.term & !m,
            quote: s.quote,
        };

        let prev_out = self
            .pending
            .take()
            .map(|prev| self.resolve(&prev, Some(next)));

        self.pending = Some(Pending { structural: s, m });
        self.in_quotes_carry = in_quotes_carry_out;
        prev_out
    }

    /// Finish the held vector against `next`, advancing the forward
    /// carries. `None` is end of input: no vector follows to supply the
    /// LSBs, so a structural at byte 63 terminates within the vector.
    #[inline]
    fn resolve(&mut self, prev: &Pending, next: Option<Lookahead>) -> VectorOutput {
        let p_delim_ooq = prev.structural.delim & !prev.m;
        let p_term_ooq = prev.structural.term & !prev.m;
        let (next_delim_ooq, next_term_ooq, next_quote) =
            next.map_or((0, 0, 0), |n| (n.delim_ooq, n.term_ooq, n.quote));

        // `B` keys off the run's end, `E` off its start — a `D_struct` each.
        let run_end = terminator_run_ends(p_term_ooq, (next_term_ooq & 1) != 0);
        let run_start = terminator_run_starts(p_term_ooq, self.prev_term_msb);
        let d_struct_b = p_delim_ooq | run_end;
        let d_struct_e = p_delim_ooq | run_start;

        let (b, begins_carry) =
            compute_field_begins(d_struct_b, prev.structural.quote, self.begins_carry);

        // Next `d_struct_e[0]`: a byte-0 terminator starts a run only if pending ended clear.
        let next_run_start0 = (next_term_ooq & 1) != 0 && (p_term_ooq >> 63) & 1 == 0;
        let next_d_lsb = (next_delim_ooq & 1) != 0 || next_run_start0;
        let (e, qe_msb) = compute_field_ends(
            d_struct_e,
            prev.structural.quote,
            prev.m,
            next_d_lsb,
            self.ends_carry,
        );

        // At end of input no byte follows byte 63, so no `B` lands past it.
        let next_b_lsb = next.is_some() && derive_b_lsb(begins_carry, (next_quote & 1) != 0);
        let r = compute_chars_to_remove(
            prev.structural.quote,
            prev.m,
            b,
            e,
            next_b_lsb,
            (next_quote & 1) != 0,
            self.config.doubled_quotes,
        );

        self.begins_carry = begins_carry;
        self.ends_carry = qe_msb;
        self.prev_term_msb = (p_term_ooq >> 63) & 1 != 0;

        VectorOutput {
            b,
            e,
            r,
            // One boundary per record: the verifier keys off collapsed run ends.
            d_struct: d_struct_b,
            term: run_end,
        }
    }

    /// Whether the scan is inside a quoted field; read at EOF to detect
    /// unclosed quotes.
    pub(super) fn in_quotes(&self) -> bool {
        self.in_quotes_carry
    }

    /// Drain the pending vector, ending the stream: the resolve assumes no
    /// vector follows, so stepping on afterwards resumes from spent state.
    pub(super) fn finalize(&mut self) -> Option<VectorOutput> {
        let prev = self.pending.take()?;
        Some(self.resolve(&prev, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default CSV semantics (`,`, `\n`, `"`) with doubled-quote escapes.
    const DEFAULT: ScannerConfig = ScannerConfig {
        delim: b',',
        term: b'\n',
        term_b: None,
        quote: Some(b'"'),
        doubled_quotes: true,
    };

    /// Build a 64-byte input from a shorter slice, padding with `pad`.
    fn pad64(prefix: &[u8], pad: u8) -> [u8; VECTOR_BYTES] {
        assert!(prefix.len() <= VECTOR_BYTES);
        let mut buf = [pad; VECTOR_BYTES];
        buf[..prefix.len()].copy_from_slice(prefix);
        buf
    }

    fn bits(positions: &[u32]) -> u64 {
        let mut m = 0u64;
        for &p in positions {
            assert!(p < 64);
            m |= 1u64 << p;
        }
        m
    }

    /// Drive the scanner over `inputs`, collecting every output
    /// (including the one from `finalize`).
    fn run(inputs: &[[u8; VECTOR_BYTES]]) -> Vec<VectorOutput> {
        let mut s = Scanner::new(DEFAULT);
        let mut out = Vec::new();
        for v in inputs {
            if let Some(o) = s.step(v) {
                out.push(o);
            }
        }
        if let Some(o) = s.finalize() {
            out.push(o);
        }
        out
    }

    // ── Single-vector behaviour ──────────────────────────────────

    #[test]
    fn first_step_emits_nothing() {
        let mut s = Scanner::new(DEFAULT);
        let v = pad64(b"a,b,c\n", b'.');
        assert!(s.step(&v).is_none());
    }

    #[test]
    fn finalize_after_one_step_emits_one_output() {
        let v = pad64(b"a,b,c\n", b'.');
        let outs = run(&[v]);
        assert_eq!(outs.len(), 1);
        let o = outs[0];
        // No quotes, no removals.
        assert_eq!(o.r, 0);
        // Delimiters at bytes 1, 3; terminator at byte 5.
        assert_eq!(o.d_struct, bits(&[1, 3, 5]));
        // E sits on the structural delim/term positions.
        assert_eq!(o.e, bits(&[1, 3, 5]));
        // B sits one past each delim/term; B[0] is the prologue's.
        assert_eq!(o.b, bits(&[2, 4, 6]));
    }

    #[test]
    fn design_example_full_pipeline() {
        // Running example: `Tom,"5'11"", Chicago",28\nAmy`.
        let raw = b"Tom,\"5'11\"\", Chicago\",28\nAmy";
        let outs = run(&[pad64(raw, b'.')]);
        assert_eq!(outs.len(), 1);
        let o = outs[0];
        // Excluding chunk-start B[0] (prologue's job), B = {5, 22, 25}.
        assert_eq!(o.b, bits(&[5, 22, 25]));
        assert_eq!(o.e, bits(&[3, 20, 24]));
        assert_eq!(o.r, bits(&[10]));
    }

    // ── Cross-vector behaviour ───────────────────────────────────

    #[test]
    fn delim_at_byte_63_carries_b_to_next_vector() {
        // A comma at V1's byte 63 must promote V2's B[0] via `BeginsCarry`.
        let mut v1 = [b'.'; VECTOR_BYTES];
        v1[63] = b',';
        let v2 = pad64(b"x,y\n", b'.');

        let outs = run(&[v1, v2]);
        assert_eq!(outs.len(), 2);
        // Nothing inside V1's 64 bytes; V2's B[0] is set by the carry.
        assert_eq!(outs[0].b, 0);
        assert_eq!(outs[1].b & 1, 1, "V2 B[0] must be set");
    }

    #[test]
    fn closing_quote_at_byte_63_pulls_e_back() {
        // V1 quotes bytes 0..=63; V2's leading comma sets V1's E[63].
        let mut v1 = [b'a'; VECTOR_BYTES];
        v1[0] = b'"';
        v1[63] = b'"';
        let mut v2 = [b'.'; VECTOR_BYTES];
        v2[0] = b',';
        v2[1] = b'\n';

        let outs = run(&[v1, v2]);
        assert_eq!(outs.len(), 2);
        // V1's D_struct is 0, but V2's comma follows the quote at 63.
        assert_ne!(
            outs[0].e & (1u64 << 63),
            0,
            "V1 E[63] must be set via lookahead",
        );
        // Without the qe-msb carry V2's comma would re-report that boundary.
        assert_eq!(
            outs[1].e & 1u64,
            0,
            "V2 E[0] must be cleared by V1's qe[63] carry",
        );
    }

    #[test]
    fn quote_run_spans_vector_boundary() {
        // A quoted field spanning both vectors: M-carry plus the E pullback.
        let mut v1 = [b'a'; VECTOR_BYTES];
        v1[0] = b'"';
        let mut v2 = [b'.'; VECTOR_BYTES];
        v2[0] = b'"';
        v2[1] = b',';
        v2[2] = b'b';
        v2[3] = b'\n';

        let outs = run(&[v1, v2]);
        assert_eq!(outs.len(), 2);
        // V1 is all in-quotes: no boundary, its opener the only removal.
        assert_eq!(outs[0].e, 0, "V1 must hold no field end");
        assert_eq!(outs[0].r, 1, "V1's opening quote is removed");
        // V2's E pulls back to the closer at 0, plus the terminator at 3.
        assert_eq!(outs[1].e, bits(&[0, 3]));
        // That closing quote sits at a field end, so it is not in R.
        assert_eq!(outs[1].r, 0);
    }

    #[test]
    fn empty_scanner_finalize_emits_nothing() {
        let mut s = Scanner::new(DEFAULT);
        assert!(s.finalize().is_none());
    }

    #[test]
    fn single_vector_with_no_quote_config() {
        let cfg = ScannerConfig {
            delim: b',',
            term: b'\n',
            term_b: None,
            quote: None,
            doubled_quotes: false,
        };
        let mut s = Scanner::new(cfg);
        let v = pad64(b"a,\"b\",c\n", b'.');
        assert!(s.step(&v).is_none());
        let o = s.finalize().unwrap();
        // Without quote config the `"` bytes are content, so no removals.
        assert_eq!(o.r, 0);
        // Delims at 1, 5; terminator at 7.
        assert_eq!(o.d_struct, bits(&[1, 5, 7]));
    }

    #[test]
    fn multi_record_across_three_vectors() {
        // Three records across two boundaries: nothing dropped or doubled.
        let v1 = pad64(b"aa,bb,cc\ndd,ee,ff\n", b'.');
        let v2 = pad64(b"gg,hh,ii\n", b'.');
        let v3 = pad64(b"jj,kk,ll\n", b'.');
        let outs = run(&[v1, v2, v3]);
        assert_eq!(outs.len(), 3);
    }
}
