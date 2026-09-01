mod bitmask;
mod bitops;
mod cursor_stepper;
mod fieldcount;
mod findstart;
mod index;
mod index_stepper;
mod scan;

use self::cursor_stepper::SimdCursorStepper;
use self::index_stepper::SimdIndexStepper;
use super::chunk::{Assumption, ChunkParser, skip_empty_lines};
use super::driver::ChunkDriver;
use super::output::Output;
use crate::config::{Config, QuoteHandling};
use crate::io::ChunkReader;

/// A SIMD-vectorized chunk parser: 64-byte vectors → structural bitmasks
/// → field indices → records. `config.quote` picks the stepper;
/// [`SimdChunkParser::supports`] gates the dialect.
#[derive(Debug)]
pub struct SimdChunkParser {
    /// Dialect and mode this parser was built for.
    config: Config,
}

impl SimdChunkParser {
    pub fn new(config: Config) -> Self {
        Self { config }
    }
}

impl ChunkParser for SimdChunkParser {
    fn config(&self) -> &Config {
        &self.config
    }

    /// Reject configs the bitmask pipeline does not model. Every
    /// terminator setting is supported: a `\r\n` run collapses to one
    /// boundary via [`bitmask::terminator_run_ends`], as blank lines do.
    fn supports(config: &Config) -> Result<(), &'static str> {
        // 1..=63 fixed fields per record (flexible or wider records route to the DFA).
        match config.field_count {
            Some(n) if (1..=index::FIXED_MAX_ARITY).contains(&n) => {}
            Some(_) => return Err("SIMD parser supports at most 63 fields per record"),
            None => return Err("SIMD parser requires a fixed field count (Arity mode)"),
        }
        if config.trim {
            return Err("SIMD parser does not support `trim`");
        }
        if config.flexible {
            return Err("SIMD parser does not support `flexible` (variable field count)");
        }
        if config.comment.is_some() {
            return Err("SIMD parser does not support comment lines");
        }
        if !matches!(config.quote_handling, QuoteHandling::Toggle) {
            return Err("SIMD parser only supports `QuoteHandling::Toggle`");
        }
        if let (Some(quote), Some(escape)) = (config.quote, config.escape)
            && escape != quote
        {
            // todo: implement backslash escapes in the SIMD parser
            return Err("SIMD parser does not yet support escape characters distinct from quote");
        }
        Ok(())
    }

    fn scan_record_start<R: ChunkReader>(
        &self,
        reader: &mut R,
        assumption: Assumption,
        strict_probe: bool,
    ) -> crate::Result<Option<usize>> {
        findstart::find_record_start(reader, &self.config, assumption, strict_probe)
    }

    /// Blank lines go first, or the header row is never found.
    fn scan_header_end<R: ChunkReader>(&self, reader: &mut R) -> crate::Result<Option<usize>> {
        let (leading, _) = skip_empty_lines(reader, &self.config)?;
        let end =
            findstart::find_record_start(reader, &self.config, Assumption::OutOfQuotes, false)?;
        Ok(end.map(|off| leading + off))
    }

    fn parse_from<S, A, R: ChunkReader, O: Output + ?Sized>(
        &self,
        reader: &mut R,
        state: &mut S,
        base: usize,
        acc: &A,
    ) -> crate::Result<Option<usize>>
    where
        A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        // No quote character means no in-quotes masks, so the cheaper
        // single-stream stepper applies.
        let driver = ChunkDriver::new(reader, base);
        if self.config.quote.is_none() {
            driver.run(&mut SimdCursorStepper::new(&self.config), state, acc)
        } else {
            driver.run(&mut SimdIndexStepper::new(&self.config), state, acc)
        }
    }
}

/// First terminator byte at or past the chunk boundary: the run's start,
/// or the boundary itself when the run straddles it (`stop`).
#[inline]
pub(super) fn handoff(run_start: usize, remaining_in_chunk: usize, stop: bool) -> usize {
    if stop {
        remaining_in_chunk.max(run_start)
    } else {
        run_start
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RecordTerminator;
    use crate::io::ChunkReader;
    use crate::parser::chunk::{FindRecordStart, PassResult};

    fn lf_config(columns: usize) -> Config {
        let mut c = Config::default();
        c.terminator = RecordTerminator::LF;
        c.field_count = Some(columns);
        c
    }

    /// In-memory `ChunkReader` for tests; `chunk_end` simulates a chunk
    /// boundary.
    struct VecChunkReader {
        /// The whole input.
        data: Vec<u8>,
        /// Read cursor into `data`.
        pos: usize,
        /// Simulated chunk boundary, as an offset into `data`.
        chunk_end: usize,
    }

    impl VecChunkReader {
        fn new(data: &[u8], chunk_end: usize) -> Self {
            Self {
                data: data.to_vec(),
                pos: 0,
                chunk_end,
            }
        }
    }

    impl ChunkReader for VecChunkReader {
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

    /// The accumulator every text-mode helper below parses with.
    fn push_strings(
        state: &mut Vec<Vec<String>>,
        fields: &mut [&mut str],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        state.push(fields.iter().map(|s| s.to_string()).collect());
        Ok(())
    }

    fn parse_collect(parser: &SimdChunkParser, input: &[u8]) -> crate::Result<Vec<Vec<String>>> {
        chunk_collect_with(
            parser,
            input,
            input.len(),
            Assumption::OutOfQuotes,
            FindRecordStart::No,
        )
        .result
    }

    /// Parse with an explicit `find` mode and `chunk_end`, returning the
    /// full `PassResult`.
    fn chunk_collect_with(
        parser: &SimdChunkParser,
        data: &[u8],
        chunk_end: usize,
        assumption: Assumption,
        find: FindRecordStart,
    ) -> PassResult<Vec<Vec<String>>> {
        let mut reader = VecChunkReader::new(data, chunk_end);
        parser.parse(&mut reader, Vec::new(), assumption, &push_strings, find)
    }

    fn parse_collect_bytes(
        parser: &SimdChunkParser,
        input: &[u8],
    ) -> crate::Result<Vec<Vec<Vec<u8>>>> {
        let mut reader = VecChunkReader::new(input, input.len());
        let pr = parser.parse(
            &mut reader,
            Vec::<Vec<Vec<u8>>>::new(),
            Assumption::OutOfQuotes,
            &|state: &mut Vec<Vec<Vec<u8>>>, fields: &mut [&mut [u8]]| {
                state.push(fields.iter().map(|s| s.to_vec()).collect());
                Ok(())
            },
            FindRecordStart::No,
        );
        pr.result
    }

    #[test]
    fn parse_simple_csv() {
        let p = SimdChunkParser::new(lf_config(3));
        let recs = parse_collect(&p, b"a,b,c\n1,2,3\n").unwrap();
        assert_eq!(recs, vec![vec!["a", "b", "c"], vec!["1", "2", "3"]]);
    }

    #[test]
    fn parse_no_trailing_newline() {
        let p = SimdChunkParser::new(lf_config(2));
        let recs = parse_collect(&p, b"a,b\n1,2").unwrap();
        assert_eq!(recs, vec![vec!["a", "b"], vec!["1", "2"]]);
    }

    #[test]
    fn parse_single_record_no_terminator() {
        let p = SimdChunkParser::new(lf_config(3));
        let recs = parse_collect(&p, b"a,b,c").unwrap();
        assert_eq!(recs, vec![vec!["a", "b", "c"]]);
    }

    #[test]
    fn parse_empty_input() {
        let p = SimdChunkParser::new(lf_config(1));
        let recs = parse_collect(&p, b"").unwrap();
        assert!(recs.is_empty());
    }

    #[test]
    fn parse_empty_fields() {
        let p = SimdChunkParser::new(lf_config(4));
        let recs = parse_collect(&p, b",,,\n").unwrap();
        assert_eq!(recs, vec![vec!["", "", "", ""]]);
    }

    #[test]
    fn parse_quoted_fields() {
        let p = SimdChunkParser::new(lf_config(2));
        let recs = parse_collect(&p, b"\"hello\",world\n").unwrap();
        assert_eq!(recs, vec![vec!["hello", "world"]]);
    }

    #[test]
    fn parse_doubled_quotes() {
        let p = SimdChunkParser::new(lf_config(2));
        let recs = parse_collect(&p, b"\"he said \"\"hi\"\"\",ok\n").unwrap();
        assert_eq!(recs, vec![vec!["he said \"hi\"", "ok"]]);
    }

    #[test]
    fn parse_quoted_with_delimiter() {
        let p = SimdChunkParser::new(lf_config(2));
        let recs = parse_collect(&p, b"\"a,b\",c\n").unwrap();
        assert_eq!(recs, vec![vec!["a,b", "c"]]);
    }

    #[test]
    fn parse_quoted_with_newline() {
        let p = SimdChunkParser::new(lf_config(2));
        let recs = parse_collect(&p, b"\"a\nb\",c\n").unwrap();
        assert_eq!(recs, vec![vec!["a\nb", "c"]]);
    }

    #[test]
    fn parse_semicolon_delimiter() {
        let mut c = lf_config(3);
        c.delimiter = b';';
        let p = SimdChunkParser::new(c);
        let recs = parse_collect(&p, b"a;b;c\n").unwrap();
        assert_eq!(recs, vec![vec!["a", "b", "c"]]);
    }

    #[test]
    fn parse_unclosed_quote_errors() {
        let p = SimdChunkParser::new(lf_config(1));
        let err = parse_collect(&p, b"\"unclosed").unwrap_err();
        assert!(matches!(err, crate::Error::UnclosedQuote { .. }));
    }

    #[test]
    fn unclosed_quote_reports_opening_quote_position() {
        // `UnclosedQuote` reports the opening quote, but B sits one past
        // it, so `finalize` must step back.
        let p = SimdChunkParser::new(lf_config(2));
        for (input, expected) in [
            // Offset 0: `step` seeds B = 0, so no step-back applies.
            (&b"\"unclosed,x"[..], 0),
            // Second field of the first record.
            (&b"foo,\"bar"[..], 4),
            // Later record, so the buffer has been stepped over first.
            (&b"a,b\nfoo,\"bar"[..], 8),
        ] {
            match parse_collect(&p, input).unwrap_err() {
                crate::Error::UnclosedQuote { position } => {
                    assert_eq!(position.byte_offset, expected, "input {input:?}");
                }
                e => panic!("expected UnclosedQuote, got {e:?}"),
            }
        }

        // Multi-vector: the offset must survive cross-vector pipelining.
        let mut input = b"aaaaaaaa,bbbbbbbb\n".repeat(8);
        let expected = input.len() + 3;
        input.extend_from_slice(b"cc,\"unclosed");
        match parse_collect(&p, &input).unwrap_err() {
            crate::Error::UnclosedQuote { position } => {
                assert_eq!(position.byte_offset, expected);
            }
            e => panic!("expected UnclosedQuote, got {e:?}"),
        }
    }

    #[test]
    fn unclosed_quote_reports_position_when_the_prefix_is_consumed() {
        // An opening quote as a record's last byte leaves no
        // buffer-relative cursor, so both steppers take the position from
        // live scan state — here the quote at byte 2.
        let p = SimdChunkParser::new(lf_config(1));
        match parse_collect(&p, b"x\n\"").unwrap_err() {
            crate::Error::UnclosedQuote { position } => {
                assert_eq!(position.byte_offset, 2);
            }
            e => panic!("expected UnclosedQuote, got {e:?}"),
        }
    }

    #[test]
    fn parse_multi_vector_input() {
        // ~200 bytes over ≥3 vectors: cross-vector pipeline and emission.
        let mut input = Vec::new();
        let mut expected: Vec<Vec<String>> = Vec::new();
        for i in 0..40 {
            let row = format!("aaa{i:03},bbb{i:03},ccc{i:03}\n");
            expected.push(row.trim_end().split(',').map(|s| s.to_string()).collect());
            input.extend_from_slice(row.as_bytes());
        }
        let p = SimdChunkParser::new(lf_config(3));
        let recs = parse_collect(&p, &input).unwrap();
        assert_eq!(recs, expected);
    }

    #[test]
    fn parse_record_spans_vector_boundary() {
        // A long quoted field that crosses multiple 64-byte boundaries.
        let mut quoted = b"\"".to_vec();
        quoted.extend(std::iter::repeat_n(b'x', 200));
        quoted.extend_from_slice(b"\",end\n");
        let p = SimdChunkParser::new(lf_config(2));
        let recs = parse_collect_bytes(&p, &quoted).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0][0].len(), 200);
        assert!(recs[0][0].iter().all(|&b| b == b'x'));
        assert_eq!(recs[0][1], b"end");
    }

    #[test]
    fn parse_all_quoted_fields() {
        // The last field's closing quote pulls E back, so the record end
        // comes from the forward scan to the terminator.
        let p = SimdChunkParser::new(lf_config(3));
        let recs = parse_collect(&p, b"\"a\",\"b\",\"c\"\n").unwrap();
        assert_eq!(recs, vec![vec!["a", "b", "c"]]);
    }

    #[test]
    fn parse_all_quoted_fields_two_records() {
        let p = SimdChunkParser::new(lf_config(2));
        let recs = parse_collect(&p, b"\"a\",\"b\"\n\"c\",\"d\"\n").unwrap();
        assert_eq!(recs, vec![vec!["a", "b"], vec!["c", "d"]]);
    }

    #[test]
    fn parse_gtfs_full_file() {
        // Regression for the cross-vector E pullback: this corpus has rows
        // whose closing quote sits at byte 63 with the delimiter at byte 0
        // of the next vector, which needs the `prev_qe_msb` carry.
        let path = crate::test_support::data::data_dir("rust-csv")
            .join("examples_data_bench_gtfs-mbta-stop-times.csv");
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut c = lf_config(9);
        c.has_headers = true;
        let p = SimdChunkParser::new(c);
        let mut reader = VecChunkReader::new(&data, data.len());
        let pr = p.parse(
            &mut reader,
            0usize,
            Assumption::OutOfQuotes,
            &|state: &mut usize, _fields: &mut [&mut str]| {
                *state += 1;
                Ok(())
            },
            FindRecordStart::SkipHeaders,
        );
        let count = pr.result.unwrap();
        assert_eq!(count, 9999);
    }

    #[test]
    fn parse_bytes_field() {
        let p = SimdChunkParser::new(lf_config(2));
        let recs = parse_collect_bytes(&p, b"raw,bytes\n").unwrap();
        assert_eq!(recs, vec![vec![b"raw".to_vec(), b"bytes".to_vec()]]);
    }

    // ── find_record_start integration ────────────────────────────

    #[test]
    fn find_skips_partial_first_record() {
        // Bytes 0..5 belong to a previous chunk's record, ending at 4.
        let p = SimdChunkParser::new(lf_config(2));
        let data = b"eld2\nrec2_f1,rec2_f2\n";
        let pr = chunk_collect_with(
            &p,
            data,
            data.len(),
            Assumption::OutOfQuotes,
            FindRecordStart::Lenient,
        );
        assert_eq!(pr.record_start, Some(4));
        let recs = pr.result.unwrap();
        assert_eq!(recs, vec![vec!["rec2_f1", "rec2_f2"]]);
    }

    #[test]
    fn find_in_quotes() {
        // Starting mid-quoted-field, the first complete record is "rec2".
        let p = SimdChunkParser::new(lf_config(1));
        let data = b"oted\"\nrec2\n";
        let pr = chunk_collect_with(
            &p,
            data,
            data.len(),
            Assumption::InQuotes,
            FindRecordStart::Lenient,
        );
        let recs = pr.result.unwrap();
        assert_eq!(recs, vec![vec!["rec2"]]);
    }

    #[test]
    fn skip_headers_drops_first_row() {
        // SkipHeaders is just OutOfQuotes find_record_start at chunk 0.
        let p = SimdChunkParser::new(lf_config(2));
        let data = b"col_a,col_b\n1,2\n3,4\n";
        let pr = chunk_collect_with(
            &p,
            data,
            data.len(),
            Assumption::OutOfQuotes,
            FindRecordStart::SkipHeaders,
        );
        let recs = pr.result.unwrap();
        assert_eq!(recs, vec![vec!["1", "2"], vec!["3", "4"]]);
    }

    #[test]
    fn no_record_start_in_chunk() {
        // Chunk has no terminator at all → record_start == None.
        let p = SimdChunkParser::new(lf_config(1));
        let data = b"still in prev record";
        let pr = chunk_collect_with(
            &p,
            data,
            data.len(),
            Assumption::OutOfQuotes,
            FindRecordStart::Lenient,
        );
        assert!(pr.record_start.is_none());
        assert!(pr.record_end.is_none());
        let recs = pr.result.unwrap();
        assert!(recs.is_empty());
    }

    #[test]
    fn find_skips_leading_blank_run_index() {
        // A blank run straddles the boundary: anchor at 4, but 5 and 6 are
        // consumed too, so the stepper opens at 7. Index path.
        let p = SimdChunkParser::new(lf_config(2));
        let data = b"eld2\n\n\nr1a,r1b\nr2a,r2b\n";
        let pr = chunk_collect_with(
            &p,
            data,
            data.len(),
            Assumption::OutOfQuotes,
            FindRecordStart::Lenient,
        );
        assert_eq!(pr.record_start, Some(4));
        let recs = pr.result.unwrap();
        assert_eq!(recs, vec![vec!["r1a", "r1b"], vec!["r2a", "r2b"]]);
    }

    #[test]
    fn find_skips_leading_blank_run_cursor() {
        // Same, but quote-free → the single-stream cursor stepper.
        let mut c = lf_config(2);
        c.quote = None;
        c.escape = None;
        let p = SimdChunkParser::new(c);
        let data = b"eld2\n\n\nr1a,r1b\nr2a,r2b\n";
        let pr = chunk_collect_with(
            &p,
            data,
            data.len(),
            Assumption::OutOfQuotes,
            FindRecordStart::Lenient,
        );
        assert_eq!(pr.record_start, Some(4));
        let recs = pr.result.unwrap();
        assert_eq!(recs, vec![vec!["r1a", "r1b"], vec!["r2a", "r2b"]]);
    }

    #[test]
    fn chunk_stops_past_boundary() {
        // chunk_end=5: start at byte 4, stop at the terminator crossing it.
        let p = SimdChunkParser::new(lf_config(2));
        let data = b"a,b\nc,d\ne,f\n";
        let pr = chunk_collect_with(
            &p,
            data,
            5,
            Assumption::OutOfQuotes,
            FindRecordStart::Lenient,
        );
        let recs = pr.result.unwrap();
        assert_eq!(recs, vec![vec!["c", "d"]]);
    }

    /// Baseline SIMD-supported config; each `supports` test flips exactly
    /// one property so it exercises that single rejection.
    fn supported_config() -> Config {
        let mut c = Config::default();
        c.terminator = RecordTerminator::LF;
        c.field_count = Some(2);
        c
    }

    #[test]
    fn rejects_missing_field_count() {
        // Variadic mode leaves `field_count` unset; SIMD is Arity-only.
        let mut c = supported_config();
        c.field_count = None;
        assert!(SimdChunkParser::supports(&c).is_err());
    }

    #[test]
    fn rejects_default_config() {
        // The crate-wide default has no field count, so SIMD declines it.
        assert!(SimdChunkParser::supports(&Config::default()).is_err());
    }

    #[test]
    fn rejects_trim() {
        let mut c = supported_config();
        c.trim = true;
        assert!(SimdChunkParser::supports(&c).is_err());
    }

    #[test]
    fn rejects_flexible() {
        let mut c = supported_config();
        c.flexible = true;
        assert!(SimdChunkParser::supports(&c).is_err());
    }

    #[test]
    fn rejects_comment() {
        let mut c = supported_config();
        c.comment = Some(b'#');
        assert!(SimdChunkParser::supports(&c).is_err());
    }

    #[test]
    fn rejects_strict_quote_handling() {
        let mut c = supported_config();
        c.quote_handling = QuoteHandling::Strict;
        assert!(SimdChunkParser::supports(&c).is_err());
    }

    #[test]
    fn rejects_literal_quote_handling() {
        let mut c = supported_config();
        c.quote_handling = QuoteHandling::Literal;
        assert!(SimdChunkParser::supports(&c).is_err());
    }

    #[test]
    fn supports_crlf_terminator() {
        // CRLF is supported via the terminator-run-collapse over `{\r, \n}`.
        let mut c = supported_config();
        c.terminator = RecordTerminator::CRLF;
        assert!(SimdChunkParser::supports(&c).is_ok());
    }

    #[test]
    fn rejects_distinct_escape_char() {
        let mut c = supported_config();
        c.quote = Some(b'"');
        c.escape = Some(b'\\');
        assert!(SimdChunkParser::supports(&c).is_err());
    }

    #[test]
    fn accepts_doubled_quote_escape() {
        // RFC 4180 (escape == quote) falls out of the XOR prefix sum on Q.
        let mut c = supported_config();
        c.quote = Some(b'"');
        c.escape = Some(b'"');
        SimdChunkParser::supports(&c).unwrap();
    }

    #[test]
    fn accepts_no_escape() {
        let mut c = supported_config();
        c.quote = Some(b'"');
        c.escape = None;
        SimdChunkParser::supports(&c).unwrap();
    }

    // ── Quote-free path (SimdCursorStepper) ──────────────────────────
    //
    // `quote = None` routes `parse_from` to the cursor stepper; every
    // test above takes the index path.

    /// LF dialect with quoting disabled → `SimdCursorStepper`.
    fn quote_free_config(columns: usize) -> Config {
        let mut c = lf_config(columns);
        c.quote = None;
        c.escape = None;
        c
    }

    #[test]
    fn cursor_simple() {
        let p = SimdChunkParser::new(quote_free_config(3));
        let recs = parse_collect(&p, b"a,b,c\n1,2,3\n").unwrap();
        assert_eq!(recs, vec![vec!["a", "b", "c"], vec!["1", "2", "3"]]);
    }

    #[test]
    fn cursor_no_trailing_newline() {
        let p = SimdChunkParser::new(quote_free_config(2));
        let recs = parse_collect(&p, b"a,b\n1,2").unwrap();
        assert_eq!(recs, vec![vec!["a", "b"], vec!["1", "2"]]);
    }

    #[test]
    fn cursor_quote_byte_is_literal_content() {
        // With `quote = None`, a `"` is ordinary content, not a quote.
        let p = SimdChunkParser::new(quote_free_config(2));
        let recs = parse_collect(&p, b"\"x,y\n").unwrap();
        assert_eq!(recs, vec![vec!["\"x", "y"]]);
    }

    /// Parse `data` (no `"`) through both paths, which are semantically
    /// identical without quote bytes. Not valid with empty lines — only
    /// the cursor skips those, until a follow-up brings parity.
    fn assert_cursor_eq_index(data: &[u8], columns: usize) {
        let cursor = SimdChunkParser::new(quote_free_config(columns));
        let index = SimdChunkParser::new(lf_config(columns));
        let c = parse_collect(&cursor, data);
        let i = parse_collect(&index, data);
        match (&c, &i) {
            (Ok(cr), Ok(ir)) => assert_eq!(cr, ir, "cursor vs index records differ"),
            (Err(_), Err(_)) => {}
            _ => panic!("cursor/index disagree: cursor={c:?} index={i:?}"),
        }
    }

    #[test]
    fn cursor_eq_index_clean_multivector() {
        // No empty lines: cursor and index agree, across a > 64-byte span.
        let big = b"aa,bb,cc,dd\n".repeat(12);
        assert_cursor_eq_index(&big, 4);
    }

    #[test]
    fn cursor_skips_empty_lines() {
        // An empty line is dropped, not emitted and not an error, as in
        // rust-csv / the DFA. Interior:
        let p = SimdChunkParser::new(quote_free_config(2));
        assert_eq!(
            parse_collect(&p, b"a,b\n\nc,d\n").unwrap(),
            vec![vec!["a", "b"], vec!["c", "d"]],
        );
        // Trailing (any number):
        let p = SimdChunkParser::new(quote_free_config(2));
        assert_eq!(
            parse_collect(&p, b"a,b\nc,d\n\n\n").unwrap(),
            vec![vec!["a", "b"], vec!["c", "d"]],
        );
        // Interior run spanning > 64 bytes worth of records:
        let mut big = b"aa,bb\n".repeat(12);
        big.extend_from_slice(b"\nxx,yy\n");
        let p = SimdChunkParser::new(quote_free_config(2));
        let recs = parse_collect(&p, &big).unwrap();
        assert_eq!(recs.len(), 13);
        assert_eq!(recs.last().unwrap(), &vec!["xx", "yy"]);
    }

    /// LF cursor config with quoting disabled, but CRLF terminator.
    fn crlf_cursor_config(columns: usize) -> Config {
        let mut c = quote_free_config(columns);
        c.terminator = RecordTerminator::CRLF;
        c
    }

    #[test]
    fn cursor_crlf() {
        // CRLF is a 2-byte terminator run, so the run-collapse handles it.
        let p = SimdChunkParser::new(crlf_cursor_config(2));
        assert_eq!(
            parse_collect(&p, b"a,b\r\nc,d\r\n").unwrap(),
            vec![vec!["a", "b"], vec!["c", "d"]],
        );
        // Empty CRLF line dropped:
        let p = SimdChunkParser::new(crlf_cursor_config(2));
        assert_eq!(
            parse_collect(&p, b"a,b\r\n\r\nc,d\r\n").unwrap(),
            vec![vec!["a", "b"], vec!["c", "d"]],
        );
        // No trailing terminator:
        let p = SimdChunkParser::new(crlf_cursor_config(2));
        assert_eq!(
            parse_collect(&p, b"a,b\r\nc,d").unwrap(),
            vec![vec!["a", "b"], vec!["c", "d"]],
        );
    }

    /// Quoted (index) config with a CRLF terminator.
    fn crlf_index_config(columns: usize) -> Config {
        let mut c = lf_config(columns);
        c.terminator = RecordTerminator::CRLF;
        c
    }

    #[test]
    fn index_crlf() {
        // Quoted path with CRLF: run-collapse plus the emit-time forward
        // scan to the run end. Plain records:
        let p = SimdChunkParser::new(crlf_index_config(2));
        assert_eq!(
            parse_collect(&p, b"a,b\r\nc,d\r\n").unwrap(),
            vec![vec!["a", "b"], vec!["c", "d"]],
        );
        // An embedded comma and CRLF stay content, masked by M:
        let p = SimdChunkParser::new(crlf_index_config(2));
        assert_eq!(
            parse_collect(&p, b"\"x,y\",b\r\n\"p\r\nq\",d\r\n").unwrap(),
            vec![vec!["x,y", "b"], vec!["p\r\nq", "d"]],
        );
        // Empty CRLF line dropped:
        let p = SimdChunkParser::new(crlf_index_config(2));
        assert_eq!(
            parse_collect(&p, b"a,b\r\n\r\nc,d\r\n").unwrap(),
            vec![vec!["a", "b"], vec!["c", "d"]],
        );
        // No trailing terminator:
        let p = SimdChunkParser::new(crlf_index_config(2));
        assert_eq!(
            parse_collect(&p, b"a,b\r\nc,d").unwrap(),
            vec![vec!["a", "b"], vec!["c", "d"]],
        );
    }

    // ── 64-byte alignment prologue ───────────────────────────────
    //
    // `bitmask::head_padded_prologue` shifts the scan's origin so that
    // every vector load after the first is 64-byte aligned. It keys off
    // `buffer().as_ptr() % 64`, which in production is any value at all,
    // so these tests sweep all 64.

    /// `ChunkReader` whose `buffer()` base sits `misalign` bytes past a
    /// 64-aligned point — the one knob the prologue reads.
    struct AlignedChunkReader {
        /// Over-allocated backing store; the input sits at `base`.
        data: Vec<u8>,
        /// Offset of the input, chosen to hit the requested misalignment.
        base: usize,
        /// Length of the input within `data`.
        len: usize,
        /// Read cursor, relative to `base`.
        pos: usize,
        /// Simulated chunk boundary, relative to `base`.
        chunk_end: usize,
    }

    impl AlignedChunkReader {
        fn new(data: &[u8], misalign: usize, chunk_end: usize) -> Self {
            // Allocated once and never resized, so the pointer stays put.
            let mut buf = vec![0u8; data.len() + 2 * 64];
            let pad = (64 - buf.as_ptr() as usize % 64) % 64 + misalign;
            buf[pad..pad + data.len()].copy_from_slice(data);
            let r = Self {
                data: buf,
                base: pad,
                len: data.len(),
                pos: 0,
                chunk_end,
            };
            assert_eq!(
                r.buffer().as_ptr() as usize % 64,
                misalign,
                "reader did not land on the requested misalignment"
            );
            r
        }
    }

    impl ChunkReader for AlignedChunkReader {
        fn buffer(&self) -> &[u8] {
            &self.data[self.base + self.pos..self.base + self.len]
        }

        fn buffer_mut(&mut self) -> &mut [u8] {
            &mut self.data[self.base + self.pos..self.base + self.len]
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

    /// Parse `reader` with either backend, so the DFA can serve as ground
    /// truth for the SIMD path. The two arms must stay one call apart.
    fn collect_with<R: ChunkReader>(
        config: &Config,
        reader: &mut R,
        find: FindRecordStart,
        simd: bool,
    ) -> crate::Result<Vec<Vec<String>>> {
        let pr = if simd {
            SimdChunkParser::new(config.clone()).parse(
                reader,
                Vec::new(),
                Assumption::OutOfQuotes,
                &push_strings,
                find,
            )
        } else {
            crate::parser::dfa::DfaChunkParser::new(config.clone()).parse(
                reader,
                Vec::new(),
                Assumption::OutOfQuotes,
                &push_strings,
                find,
            )
        };
        pr.result
    }

    fn collect_at_misalign(
        config: &Config,
        input: &[u8],
        misalign: usize,
        simd: bool,
    ) -> crate::Result<Vec<Vec<String>>> {
        let mut reader = AlignedChunkReader::new(input, misalign, input.len());
        collect_with(config, &mut reader, FindRecordStart::No, simd)
    }

    /// The prologue must not change what is parsed, at any misalignment;
    /// the DFA, which has neither vectors nor prologue, is ground truth.
    fn assert_misalignment_invariant(config: &Config, input: &[u8]) {
        let expected = collect_at_misalign(config, input, 0, false)
            .expect("DFA reference parse should succeed");
        assert!(
            input.len() >= 127,
            "input too short to reach the prologue's `skip + 64` floor",
        );
        for misalign in 0..64 {
            let got = collect_at_misalign(config, input, misalign, true)
                .unwrap_or_else(|e| panic!("SIMD parse failed at misalign {misalign}: {e}"));
            assert_eq!(got, expected, "SIMD output differs at misalign {misalign}");
        }
    }

    /// Quote-free path, with records short enough that the sweep lands a
    /// terminator on the prologue seam.
    #[test]
    fn cursor_prologue_matches_dfa_at_every_misalignment() {
        let config = quote_free_config(3);
        let mut input = Vec::new();
        for i in 0..64 {
            input.extend_from_slice(format!("{i},b{i},c\n").as_bytes());
        }
        assert_misalignment_invariant(&config, &input);
    }

    /// Same sweep at 7 bytes per record: coprime with 64, so the seam
    /// visits every byte position within a record.
    #[test]
    fn cursor_prologue_matches_dfa_with_odd_record_length() {
        let config = quote_free_config(2);
        let input = b"ab,cde\n".repeat(40);
        assert_misalignment_invariant(&config, &input);
    }

    /// Quoted path, which drives the full `Scanner`: every carry has to
    /// come out of the padded vector as if the scan began at the record.
    #[test]
    fn index_prologue_matches_dfa_at_every_misalignment() {
        let config = lf_config(3);
        let mut input = Vec::new();
        for i in 0..40 {
            input.extend_from_slice(format!("\"q{i},x\",plain{i},\"z\"\n").as_bytes());
        }
        assert_misalignment_invariant(&config, &input);
    }

    /// A quote on the seam for some misalignment, so the in-quotes carry
    /// has to cross the boundary.
    #[test]
    fn index_prologue_matches_dfa_with_quotes_on_the_seam() {
        let config = lf_config(2);
        let input = b"\"a\",\"bc\"\n".repeat(40);
        assert_misalignment_invariant(&config, &input);
    }

    /// CRLF: for half the misalignments the two-byte run straddles the
    /// seam and must still collapse to one boundary.
    #[test]
    fn crlf_prologue_matches_dfa_at_every_misalignment() {
        let config = crlf_index_config(2);
        let input = b"ab,cd\r\n".repeat(40);
        assert_misalignment_invariant(&config, &input);
    }

    #[test]
    fn head_padded_prologue_aligns_the_following_vector() {
        use super::bitmask::{Prologue, VECTOR_BYTES, head_padded_prologue};
        let input = b"x".repeat(512);
        for misalign in 0..64 {
            let reader = AlignedChunkReader::new(&input, misalign, input.len());
            let buf = reader.buffer();
            match head_padded_prologue(buf, buf.len(), b'.') {
                Prologue::Aligned => assert_eq!(misalign, 0, "only an aligned base is `Aligned`"),
                Prologue::TooShort => panic!("512 bytes always leave an aligned vector"),
                Prologue::Vector(v, m) => {
                    assert_eq!(m, misalign);
                    let skip = VECTOR_BYTES - misalign;
                    // Low lanes are pad, high lanes are the real head.
                    assert!(v[..misalign].iter().all(|&b| b == b'.'));
                    assert_eq!(&v[misalign..], &buf[..skip]);
                    // The whole point: the next vector is aligned.
                    assert_eq!(buf[skip..].as_ptr() as usize % VECTOR_BYTES, 0);
                }
            }
        }
    }

    /// Below `skip + 64` no aligned vector follows, so the prologue
    /// declines rather than leave a wrapped base across a suspension.
    #[test]
    fn head_padded_prologue_declines_when_no_aligned_vector_follows() {
        use super::bitmask::{Prologue, VECTOR_BYTES, head_padded_prologue};
        let input = b"x".repeat(512);
        for misalign in 1..64 {
            let reader = AlignedChunkReader::new(&input, misalign, input.len());
            let buf = reader.buffer();
            let skip = VECTOR_BYTES - misalign;
            // `TooShort`, not `Aligned`: the caller must retry, not give up.
            assert!(matches!(
                head_padded_prologue(buf, skip + VECTOR_BYTES - 1, b'.'),
                Prologue::TooShort
            ));
            assert!(matches!(
                head_padded_prologue(buf, skip + VECTOR_BYTES, b'.'),
                Prologue::Vector(..)
            ));
        }
    }

    /// A chunk must own exactly the records the DFA assigns it, at every
    /// boundary — including one landing in a sub-vector tail, which
    /// `finalize` only gets right in the current buffer frame.
    fn assert_chunk_ownership_matches_dfa(config: &Config, input: &[u8]) {
        let collect = |chunk_end: usize, simd: bool| -> Vec<Vec<String>> {
            let mut reader = VecChunkReader::new(input, chunk_end);
            collect_with(config, &mut reader, FindRecordStart::No, simd)
                .expect("parse should succeed")
        };
        for chunk_end in 1..=input.len() {
            assert_eq!(
                collect(chunk_end, true),
                collect(chunk_end, false),
                "SIMD and DFA disagree at chunk_end {chunk_end}"
            );
        }
    }

    #[test]
    fn cursor_chunk_ownership_matches_dfa() {
        assert_chunk_ownership_matches_dfa(&quote_free_config(2), &b"a,b\n".repeat(40));
        assert_chunk_ownership_matches_dfa(
            &quote_free_config(2),
            &b"aaaaaaaa,bbbbbbbb\n".repeat(20),
        );
    }

    #[test]
    fn index_chunk_ownership_matches_dfa() {
        assert_chunk_ownership_matches_dfa(&lf_config(2), &b"a,b\n".repeat(40));
        assert_chunk_ownership_matches_dfa(&lf_config(2), &b"aaaaaaaa,bbbbbbbb\n".repeat(20));
    }

    /// Blank-line runs must yield the same records under either parser.
    /// Ownership of a run legitimately differs, so this varies the find
    /// mode over one chunk instead of sweeping `chunk_end`.
    fn assert_run_handling_matches_dfa(config: &Config, input: &[u8], find: FindRecordStart) {
        let collect = |simd: bool| -> Result<Vec<Vec<String>>, String> {
            let mut reader = VecChunkReader::new(input, input.len());
            collect_with(config, &mut reader, find, simd).map_err(|e| e.to_string())
        };
        assert_eq!(
            collect(true),
            collect(false),
            "SIMD and DFA disagree on {:?} (find {find:?})",
            String::from_utf8_lossy(input),
        );
    }

    #[test]
    fn blank_line_runs_match_dfa() {
        let mut cases: Vec<Vec<u8>> = vec![
            b"\n\n\na,b\nc,d\n".to_vec(),
            b"a,b\n\n\nc,d\n".to_vec(),
            b"a,b\nc,d\n\n\n".to_vec(),
            b"\n\na,b\n\n\nc,d\n\n".to_vec(),
            b"\n".to_vec(),
            b"\n\n\n".to_vec(),
        ];
        // A run wider than a vector, so the collapse spans loads.
        let mut long = b"a,b\n".to_vec();
        long.extend(std::iter::repeat_n(b'\n', 65));
        long.extend_from_slice(b"c,d\n");
        cases.push(long);

        for find in [FindRecordStart::No, FindRecordStart::Lenient] {
            for config in [lf_config(2), quote_free_config(2)] {
                for data in &cases {
                    assert_run_handling_matches_dfa(&config, data, find);
                }
            }
        }
    }

    /// Under `FindRecordStart::No` leading blank lines are skipped without
    /// moving the anchor off the chunk's start.
    #[test]
    fn leading_blank_lines_are_skipped() {
        for (data, want_start) in [
            (&b"\n\n\na,b\nc,d\n"[..], 0),
            (&b"\na,b\nc,d\n"[..], 0),
            (&b"a,b\nc,d\n"[..], 0),
        ] {
            for config in [lf_config(2), quote_free_config(2)] {
                let p = SimdChunkParser::new(config);
                let pr = chunk_collect_with(
                    &p,
                    data,
                    data.len(),
                    Assumption::OutOfQuotes,
                    FindRecordStart::No,
                );
                assert_eq!(pr.record_start, Some(want_start));
                assert_eq!(
                    pr.result.unwrap(),
                    vec![vec!["a", "b"], vec!["c", "d"]],
                    "input {:?}",
                    String::from_utf8_lossy(data)
                );
            }
        }
    }
}
