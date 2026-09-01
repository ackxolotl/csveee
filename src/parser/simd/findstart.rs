//! SIMD-flavored `find_record_start`: scans a chunk under a CSV-semantic
//! [`Assumption`] for the first `(N & D_struct)` bit, and under `strict`
//! rejects a quote that contradicts it — see [`StrictProbe`].

use super::bitmask::{Structural, VECTOR_BYTES, compute_in_quotes, match_structural, neutral_pad};
use crate::config::Config;
use crate::io::ChunkReader;
use crate::parser::chunk::Assumption;

/// Initial in-quotes carry for the M-bitmask: the M-state of the byte
/// just before the chunk. Mid-escape, a doubled quote has already
/// toggled M back out, so only a distinct escape byte carries `true`.
fn initial_carry(config: &Config, assumption: Assumption) -> bool {
    match assumption {
        Assumption::OutOfQuotes => false,
        Assumption::InQuotes => true,
        Assumption::InQuotesAfterEscape => config.quote != config.escape,
    }
}

/// Scan forward under `assumption` for the first out-of-quotes record
/// terminator, consuming the bytes before it and returning its offset,
/// relative to the reader's position. `None` if the chunk has none.
pub(super) fn find_record_start<R: ChunkReader>(
    reader: &mut R,
    config: &Config,
    assumption: Assumption,
    strict: bool,
) -> crate::Result<Option<usize>> {
    let (term, term_b) = config.terminator.bytes();
    let delim = config.delimiter;
    let quote = config.quote;
    let pad = neutral_pad(delim, term, term_b, quote);

    let mut carry = initial_carry(config, assumption);
    let mut total_consumed: usize = 0;
    let mut probe = StrictProbe::new();

    loop {
        // Near EOF `buffer()` is shorter; the `scan_end` cap covers that.
        reader.fill(VECTOR_BYTES)?;
        let buf = reader.buffer();
        if buf.is_empty() {
            return Ok(None);
        }

        let chunk_end = reader.remaining_in_chunk();
        if chunk_end == 0 {
            return Ok(None);
        }

        // Never read past the chunk boundary; the merge phase reparses a
        // record that straddles chunks.
        let scan_end = chunk_end.min(buf.len());
        let avail = scan_end.min(VECTOR_BYTES);

        // The neutral pad produces no structural bits, so any terminator
        // found sits at a real input position.
        let v: [u8; VECTOR_BYTES] = if avail == VECTOR_BYTES {
            buf[..VECTOR_BYTES].try_into().expect("64-byte slice")
        } else {
            let mut padded = [pad; VECTOR_BYTES];
            padded[..avail].copy_from_slice(&buf[..avail]);
            padded
        };

        let s = match_structural(&v, delim, term, term_b, quote);
        let (m, new_carry) = compute_in_quotes(s.quote, carry);
        let d_struct = s.structural_delims(m);
        let structural_term = s.term & d_struct;

        // First out-of-quotes terminator within the real (non-pad) bytes.
        let term_bit = (structural_term != 0)
            .then(|| structural_term.trailing_zeros() as usize)
            .filter(|&b| b < avail);

        if strict && let Some(e) = probe.check(&s, m, avail, term_bit, total_consumed) {
            return Err(e);
        }

        if let Some(bit) = term_bit {
            reader.consume(bit);
            return Ok(Some(total_consumed + bit));
        }

        if avail < VECTOR_BYTES {
            // Tail vector with no terminator: the merge path takes over.
            return Ok(None);
        }

        carry = new_carry;
        if strict {
            probe.advance(&s);
        }
        reader.consume(VECTOR_BYTES);
        total_consumed += VECTOR_BYTES;
    }
}

/// Strict-probe error at a chunk-relative offset; the scheduler rebases
/// it via `Error::with_base`.
fn strict_err(byte_offset: usize) -> crate::Error {
    crate::Error::InvalidQuote {
        position: crate::error::Position { byte_offset },
    }
}

/// Cross-vector state for the strict probe: the two neighbour bits that
/// straddle 64-byte vectors.
struct StrictProbe {
    /// Whether the previous vector's last real byte was a field boundary.
    prev_last_bnd: bool,
    /// A closing quote parked in bit 63, with the chunk-relative offset
    /// of the byte that has to follow it.
    pending_closer_msb: Option<usize>,
}

impl StrictProbe {
    fn new() -> Self {
        Self {
            prev_last_bnd: true,
            pending_closer_msb: None,
        }
    }

    /// Inspect one vector, returning `Some(err)` if it proves the assumed
    /// in-quotes state impossible: an opener without a preceding boundary,
    /// or a closer without a following one (`""` pairs count as one).
    fn check(
        &mut self,
        s: &Structural,
        m: u64,
        avail: usize,
        term_bit: Option<usize>,
        total_consumed: usize,
    ) -> Option<crate::Error> {
        let bnd = s.delim | s.term | s.quote;

        // Resolve a deferred bit-63 closer against this vector's byte 0.
        if let Some(pos) = self.pending_closer_msb.take()
            && bnd & 1 == 0
        {
            return Some(strict_err(pos));
        }

        let openers = s.quote & m;
        let closers = s.quote & !m;
        let left_bnd = (bnd << 1) | self.prev_last_bnd as u64;
        let right_bnd = bnd >> 1;
        let bad_openers = openers & !left_bnd;
        let mut bad_closers = closers & !right_bnd;

        // A bit-63 closer's right neighbour is in the next vector.
        const MSB: u64 = 1 << 63;
        if avail == VECTOR_BYTES && bad_closers & MSB != 0 {
            bad_closers &= !MSB;
            self.pending_closer_msb = Some(total_consumed + VECTOR_BYTES);
        }
        // A tail vector's last byte borders the chunk/EOF edge.
        if avail < VECTOR_BYTES && avail > 0 {
            bad_closers &= !(1u64 << (avail - 1));
        }

        // Bad bits past the terminator belong to the next record.
        let cutoff = term_bit.unwrap_or(avail);
        let mask_below = if cutoff >= 64 {
            u64::MAX
        } else {
            (1u64 << cutoff) - 1
        };
        let bad_openers = bad_openers & mask_below;
        let bad_closers = bad_closers & mask_below;

        let opener = (bad_openers != 0).then(|| bad_openers.trailing_zeros() as usize);
        // Like the DFA, anchor a misplaced closer on the byte after it.
        let closer = (bad_closers != 0).then(|| bad_closers.trailing_zeros() as usize + 1);
        Some(strict_err(
            total_consumed + opener.into_iter().chain(closer).min()?,
        ))
    }

    /// Carry the last byte forward as the next vector's left neighbour.
    fn advance(&mut self, s: &Structural) {
        self.prev_last_bnd = (s.delim | s.term | s.quote) >> 63 != 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, RecordTerminator};

    /// In-memory `ChunkReader` for tests.
    struct VecReader {
        /// The whole input.
        data: Vec<u8>,
        /// Read cursor into `data`.
        pos: usize,
        /// Simulated chunk boundary, as an offset into `data`.
        chunk_end: usize,
    }

    impl VecReader {
        fn new(data: &[u8], chunk_end: usize) -> Self {
            Self {
                data: data.to_vec(),
                pos: 0,
                chunk_end,
            }
        }
    }

    impl ChunkReader for VecReader {
        fn buffer(&self) -> &[u8] {
            &self.data[self.pos..]
        }
        fn buffer_mut(&mut self) -> &mut [u8] {
            &mut self.data[self.pos..]
        }
        fn fill(&mut self, _n: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn consume(&mut self, n: usize) {
            self.pos += n;
        }
        fn remaining_in_chunk(&self) -> usize {
            self.chunk_end.saturating_sub(self.pos)
        }
    }

    fn lf_config() -> Config {
        let mut c = Config::default();
        c.terminator = RecordTerminator::LF;
        c
    }

    #[test]
    fn out_of_quotes_finds_first_terminator() {
        // "field_a,field_b\nrecord2\n" — first \n at byte 15.
        let data = b"field_a,field_b\nrecord2\n";
        let mut r = VecReader::new(data, data.len());
        let start =
            find_record_start(&mut r, &lf_config(), Assumption::OutOfQuotes, false).unwrap();
        assert_eq!(start, Some(15));
    }

    #[test]
    fn in_quotes_finds_first_terminator_after_close() {
        // `inside",field\nrec2\n`: the closer is at 6, so the first
        // out-of-quotes \n is at 13.
        let data = b"inside\",field\nrec2\n";
        let mut r = VecReader::new(data, data.len());
        let start = find_record_start(&mut r, &lf_config(), Assumption::InQuotes, false).unwrap();
        assert_eq!(start, Some(13));
    }

    #[test]
    fn in_quotes_ignores_terminator_inside_quotes() {
        // The `\n` at byte 6 is inside the quoted field.
        let data = b"inside\nstill\",x\nrec2\n";
        let mut r = VecReader::new(data, data.len());
        let start = find_record_start(&mut r, &lf_config(), Assumption::InQuotes, false).unwrap();
        assert_eq!(start, Some(15));
    }

    #[test]
    fn returns_none_when_no_terminator_in_chunk() {
        // No \n at all → None.
        let data = b"a,b,c";
        let mut r = VecReader::new(data, data.len());
        let start =
            find_record_start(&mut r, &lf_config(), Assumption::OutOfQuotes, false).unwrap();
        assert_eq!(start, None);
    }

    #[test]
    fn respects_chunk_boundary() {
        // The terminator at byte 30 is past chunk_end = 20.
        let mut data = vec![b'a'; 30];
        data.push(b'\n');
        data.extend_from_slice(b"rec2\n");
        let mut r = VecReader::new(&data, 20);
        let start =
            find_record_start(&mut r, &lf_config(), Assumption::OutOfQuotes, false).unwrap();
        assert_eq!(start, None);
        // The reader must not have consumed past the chunk end.
        assert!(r.pos <= 20);
    }

    #[test]
    fn terminator_in_second_vector() {
        // Pad with non-structural content for 70 bytes, then a \n.
        let mut data = vec![b'a'; 70];
        data.push(b'\n');
        data.extend_from_slice(b"rec\n");
        let mut r = VecReader::new(&data, data.len());
        let start =
            find_record_start(&mut r, &lf_config(), Assumption::OutOfQuotes, false).unwrap();
        assert_eq!(start, Some(70));
    }

    #[test]
    fn terminator_at_vector_boundary() {
        // \n at exactly byte 63 (last byte of the first vector).
        let mut data = vec![b'a'; 63];
        data.push(b'\n');
        data.extend_from_slice(b"rec\n");
        let mut r = VecReader::new(&data, data.len());
        let start =
            find_record_start(&mut r, &lf_config(), Assumption::OutOfQuotes, false).unwrap();
        assert_eq!(start, Some(63));
    }

    #[test]
    fn quoted_terminator_then_real_one() {
        // `"a\nb",x\ny`: the \n at byte 2 is quoted, the one at 7 is
        // structural.
        let data = b"\"a\nb\",x\ny";
        let mut r = VecReader::new(data, data.len());
        let start =
            find_record_start(&mut r, &lf_config(), Assumption::OutOfQuotes, false).unwrap();
        assert_eq!(start, Some(7));
    }

    #[test]
    fn in_quotes_after_escape_treats_carry_as_out_of_quotes() {
        // Byte 0 is the second `"` of a `""` pair, so the toggle puts us
        // back inside quotes and the first out-of-quotes `\n` is at 8.
        let data = b"\"more\",x\n";
        let mut r = VecReader::new(data, data.len());
        let start = find_record_start(&mut r, &lf_config(), Assumption::InQuotesAfterEscape, false)
            .unwrap();
        assert_eq!(start, Some(8));
    }

    #[test]
    fn empty_input_returns_none() {
        let mut r = VecReader::new(b"", 0);
        let start =
            find_record_start(&mut r, &lf_config(), Assumption::OutOfQuotes, false).unwrap();
        assert_eq!(start, None);
    }

    #[test]
    fn semicolon_delimiter() {
        // Sanity: non-default delimiters work.
        let mut c = lf_config();
        c.delimiter = b';';
        let data = b"a;b;c\nrec2\n";
        let mut r = VecReader::new(data, data.len());
        let start = find_record_start(&mut r, &c, Assumption::OutOfQuotes, false).unwrap();
        assert_eq!(start, Some(5));
    }

    // ── strict probe ─────────────────────────────────────────────────

    /// Config with a quote char so the strict probe is meaningful.
    fn quoted_lf_config() -> Config {
        let mut c = lf_config();
        c.quote = Some(b'"');
        c
    }

    #[test]
    fn strict_rejects_wrong_out_of_quotes_assumption() {
        // The chunk starts inside a quoted field, so under `OutOfQuotes`
        // the closer looks like an opener preceded by content (`l"`).
        let data = b"g-gu, Seoul\",37\nnext,rec\n";
        let mut r = VecReader::new(data, data.len());
        let err = find_record_start(&mut r, &quoted_lf_config(), Assumption::OutOfQuotes, true);
        assert!(
            matches!(err, Err(crate::Error::InvalidQuote { .. })),
            "got {err:?}"
        );
    }

    #[test]
    fn strict_accepts_correct_in_quotes_assumption() {
        // Correct assumption: the `"` is a closer followed by `,`.
        let data = b"g-gu, Seoul\",37\nnext,rec\n";
        let mut r = VecReader::new(data, data.len());
        let start =
            find_record_start(&mut r, &quoted_lf_config(), Assumption::InQuotes, true).unwrap();
        assert_eq!(start, Some(15));
    }

    #[test]
    fn strict_accepts_well_formed_out_of_quotes() {
        // A properly-quoted middle field: opener after `,`, closer before.
        let data = b"a,\"b,c\",d\nrec2\n";
        let mut r = VecReader::new(data, data.len());
        let start =
            find_record_start(&mut r, &quoted_lf_config(), Assumption::OutOfQuotes, true).unwrap();
        assert_eq!(start, Some(9));
    }

    #[test]
    fn strict_accepts_doubled_quote_escape() {
        // In a `""` pair each `"` is the other's legal neighbour.
        let data = b"a,\"x\"\"y\",b\nr2\n";
        let mut r = VecReader::new(data, data.len());
        let mut c = quoted_lf_config();
        c.escape = Some(b'"');
        let start = find_record_start(&mut r, &c, Assumption::OutOfQuotes, true).unwrap();
        assert_eq!(start, Some(10));
    }

    /// A misplaced closer is anchored on the byte that follows it, as the
    /// DFA probe anchors it.
    #[test]
    fn strict_closer_error_anchors_on_following_byte() {
        // Under `InQuotes` the `"` at 3 closes, but `d` is no boundary.
        let data = b"abc\"def,x\n";
        let mut r = VecReader::new(data, data.len());
        let err = find_record_start(&mut r, &quoted_lf_config(), Assumption::InQuotes, true);
        assert!(
            matches!(
                err,
                Err(crate::Error::InvalidQuote {
                    position: crate::error::Position { byte_offset: 4 }
                })
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn strict_opener_error_anchors_on_the_quote() {
        // Under `OutOfQuotes` the `"` at 11 opens, but `l` is no boundary.
        let data = b"g-gu, Seoul\",37\nnext,rec\n";
        let mut r = VecReader::new(data, data.len());
        let err = find_record_start(&mut r, &quoted_lf_config(), Assumption::OutOfQuotes, true);
        assert!(
            matches!(
                err,
                Err(crate::Error::InvalidQuote {
                    position: crate::error::Position { byte_offset: 11 }
                })
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn strict_closer_at_bit_63_anchors_on_next_vector() {
        // The closer sits in bit 63, so its right neighbour — byte 64 of
        // the next vector — decides, and carries the offset.
        let mut data = vec![b'a'; 63];
        data.push(b'"');
        data.extend_from_slice(b"x,y\n");
        let mut r = VecReader::new(&data, data.len());
        let err = find_record_start(&mut r, &quoted_lf_config(), Assumption::InQuotes, true);
        assert!(
            matches!(
                err,
                Err(crate::Error::InvalidQuote {
                    position: crate::error::Position { byte_offset: 64 }
                })
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn strict_rejects_across_vector_boundary() {
        // A contradicting quote past the first vector: the carry must see it.
        let mut data = vec![b'a'; 70];
        data.extend_from_slice(b" text\",99\nnext\n");
        let mut r = VecReader::new(&data, data.len());
        let err = find_record_start(&mut r, &quoted_lf_config(), Assumption::OutOfQuotes, true);
        assert!(
            matches!(err, Err(crate::Error::InvalidQuote { .. })),
            "got {err:?}"
        );
    }
}
