mod stepper;
mod table;

use self::stepper::DfaStepper;
use self::table::{Dfa, ERROR_STATE, FIRST_ACTION, RECORD_END_STATE, State};
use super::chunk::{Assumption, ChunkParser};
use super::driver::ChunkDriver;
use super::output::Output;
use crate::config::{Config, QuoteHandling};
use crate::io::ChunkReader;

/// A DFA-based chunk parser.
///
/// `dfa` implements the user's dialect and drives `parse_records`;
/// `find_dfa` is the same dialect forced to `QuoteHandling::Toggle` and
/// drives `find_record_start`.
#[derive(Debug)]
pub struct DfaChunkParser {
    dfa: Dfa,
    find_dfa: Dfa,
    config: Config,
}

impl DfaChunkParser {
    pub fn new(config: Config) -> Self {
        let dfa = Dfa::new(&config);
        let mut find_config = config.clone();
        find_config.quote_handling = QuoteHandling::Toggle;
        let find_dfa = Dfa::new(&find_config);
        Self {
            dfa,
            find_dfa,
            config,
        }
    }

    /// Map an assumption to the initial DFA state.
    fn assumption_to_state(&self, assumption: Assumption) -> State {
        match assumption {
            Assumption::OutOfQuotes => State::FieldStart,
            Assumption::InQuotes => State::InQuotedField,
            Assumption::InQuotesAfterEscape => {
                let doubled_quotes = matches!(
                    (self.config.quote, self.config.escape),
                    (Some(q), Some(e)) if q == e
                );
                if doubled_quotes {
                    State::AfterClosingQuote
                } else {
                    State::InEscapedQuote
                }
            }
        }
    }

    /// Scan for the first record boundary starting from `initial_state`.
    ///
    /// Bounded to the nominal chunk end via `reader.remaining_in_chunk()`
    /// — without the cap one iteration would scan past the chunk, since
    /// buffers extend to EOF. `Ok(None)` means no boundary inside the
    /// chunk; the scheduler's merge-time reparse covers that.
    ///
    /// `strict_probe` is only meaningful with `find_dfa`, which tags
    /// misplaced quotes instead of erroring on them.
    fn find_record_start_from<R: ChunkReader>(
        &self,
        dfa: &Dfa,
        strict_probe: bool,
        reader: &mut R,
        initial_state: State,
    ) -> crate::Result<Option<usize>> {
        let mut dfa_state = initial_state;
        let mut total_consumed: usize = 0;

        loop {
            reader.fill(1)?;
            let buf = reader.buffer();

            let buf_end = buf.len().min(reader.remaining_in_chunk());
            if buf_end == 0 {
                return Ok(None);
            }

            let mut pos = 0;
            while pos < buf_end {
                // Same scan fast path as the stepper, but the bytes
                // are discarded (they belong to the previous chunk's
                // record or the header row), so there is no write cursor
                // to maintain.
                let tail = &buf[pos..buf_end];
                pos += match dfa_state {
                    State::InField => dfa.in_field_needles().find(tail),
                    State::InQuotedField => dfa.in_quoted_needles().find(tail),
                    // Empty-line skip. Cold relative to the field scan,
                    // so the mask-based scan is fine here.
                    State::RecordStart => {
                        let mask = dfa.record_start_structural();
                        tail.iter().position(|&b| mask[b as usize])
                    }
                    _ => Some(0),
                }
                .unwrap_or(tail.len());
                if pos >= buf_end {
                    break;
                }

                let byte = buf[pos];
                let trans = dfa.transition(dfa_state, byte);
                pos += 1;
                let strict_rejected = strict_probe && trans.is_strict_error();
                dfa_state = trans.next_state();
                let s = dfa_state as u8;
                if s < FIRST_ACTION && !strict_rejected {
                    continue;
                }
                let anchor = pos - 1;
                let byte_offset = total_consumed + anchor;
                if s == ERROR_STATE || strict_rejected {
                    crate::trace::debug!(
                        scan_offset = byte_offset,
                        byte = byte,
                        byte_ch = format!("{:?}", byte as char),
                        "find_record_start: strict rejected byte",
                    );
                    reader.consume(pos);
                    return Err(crate::Error::InvalidQuote {
                        position: crate::error::Position { byte_offset },
                    });
                }
                if s == RECORD_END_STATE {
                    reader.consume(anchor);
                    return Ok(Some(byte_offset));
                }
                // FieldEnd — resume scanning the next field.
                dfa_state = State::FieldStart;
            }

            reader.consume(pos);
            total_consumed += pos;
        }
    }
}

impl ChunkParser for DfaChunkParser {
    fn config(&self) -> &Config {
        &self.config
    }

    fn supports(_config: &Config) -> Result<(), &'static str> {
        Ok(()) // DFA supports all configurations.
    }

    fn scan_record_start<R: ChunkReader>(
        &self,
        reader: &mut R,
        assumption: Assumption,
        strict_probe: bool,
    ) -> crate::Result<Option<usize>> {
        self.find_record_start_from(
            &self.find_dfa,
            strict_probe,
            reader,
            self.assumption_to_state(assumption),
        )
    }

    /// The user's dialect, not `find_dfa`: chunk 0's state is exact.
    /// `RecordStart` consumes leading comment lines before the header.
    fn scan_header_end<R: ChunkReader>(&self, reader: &mut R) -> crate::Result<Option<usize>> {
        self.find_record_start_from(&self.dfa, false, reader, State::RecordStart)
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
        // The driver owns fill / consume / UTF-8 validation; the stepper
        // owns the per-byte DFA.
        let mut stepper = DfaStepper::new(&self.dfa, &self.config, base);
        ChunkDriver::new(reader, base).run(&mut stepper, state, acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::chunk::{FindRecordStart, PassResult};

    fn default_parser() -> DfaChunkParser {
        DfaChunkParser::new(Config::default())
    }

    fn strict_parser() -> DfaChunkParser {
        let mut config = Config::default();
        config.quote_handling = QuoteHandling::Strict;
        DfaChunkParser::new(config)
    }

    /// Helper: parse entire input from the start (no find_record_start).
    fn parse_collect(parser: &DfaChunkParser, input: &[u8]) -> crate::Result<Vec<Vec<String>>> {
        let pr = chunk_collect_with(
            parser,
            input,
            input.len(),
            Assumption::OutOfQuotes,
            FindRecordStart::No,
        );
        pr.result
    }

    #[test]
    fn parse_simple_csv() {
        let p = default_parser();
        let records = parse_collect(&p, b"a,b,c\n1,2,3\n").unwrap();
        assert_eq!(records, vec![vec!["a", "b", "c"], vec!["1", "2", "3"],]);
    }

    #[test]
    fn parse_no_trailing_newline() {
        let p = default_parser();
        let records = parse_collect(&p, b"a,b\n1,2").unwrap();
        assert_eq!(records, vec![vec!["a", "b"], vec!["1", "2"],]);
    }

    #[test]
    fn parse_crlf() {
        let p = default_parser();
        let records = parse_collect(&p, b"a,b\r\n1,2\r\n").unwrap();
        assert_eq!(records, vec![vec!["a", "b"], vec!["1", "2"],]);
    }

    #[test]
    fn parse_quoted_fields() {
        let p = default_parser();
        let records = parse_collect(&p, b"\"hello\",world\n").unwrap();
        assert_eq!(records, vec![vec!["hello", "world"]]);
    }

    #[test]
    fn parse_doubled_quotes() {
        let p = default_parser();
        let records = parse_collect(&p, b"\"he said \"\"hi\"\"\",ok\n").unwrap();
        assert_eq!(records, vec![vec!["he said \"hi\"", "ok"]]);
    }

    #[test]
    fn parse_quoted_with_delimiter() {
        let p = default_parser();
        let records = parse_collect(&p, b"\"a,b\",c\n").unwrap();
        assert_eq!(records, vec![vec!["a,b", "c"]]);
    }

    #[test]
    fn parse_quoted_with_newline() {
        let p = default_parser();
        let records = parse_collect(&p, b"\"a\nb\",c\n").unwrap();
        assert_eq!(records, vec![vec!["a\nb", "c"]]);
    }

    #[test]
    fn parse_empty_fields() {
        let p = default_parser();
        let records = parse_collect(&p, b",,,\n").unwrap();
        assert_eq!(records, vec![vec!["", "", "", ""]]);
    }

    #[test]
    fn parse_empty_input() {
        let p = default_parser();
        let records = parse_collect(&p, b"").unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn parse_empty_lines_skipped() {
        let p = default_parser();
        let records = parse_collect(&p, b"\n\na,b\n\n").unwrap();
        assert_eq!(records, vec![vec!["a", "b"]]);
    }

    #[test]
    fn parse_semicolon_delimiter() {
        let mut config = Config::default();
        config.delimiter = b';';
        let p = DfaChunkParser::new(config);
        let records = parse_collect(&p, b"a;b;c\n").unwrap();
        assert_eq!(records, vec![vec!["a", "b", "c"]]);
    }

    #[test]
    fn parse_backslash_escape() {
        let mut config = Config::default();
        config.escape = Some(b'\\');
        let p = DfaChunkParser::new(config);
        let records = parse_collect(&p, b"\"a\\\"b\",c\n").unwrap();
        assert_eq!(records, vec![vec!["a\"b", "c"]]);
    }

    #[test]
    fn parse_comment_lines() {
        let mut config = Config::default();
        config.comment = Some(b'#');
        let p = DfaChunkParser::new(config);
        let records = parse_collect(&p, b"# comment\na,b\n# another\n1,2\n").unwrap();
        assert_eq!(records, vec![vec!["a", "b"], vec!["1", "2"],]);
    }

    #[test]
    fn parse_unclosed_quote_errors() {
        let p = default_parser();
        let result = parse_collect(&p, b"\"unclosed\n");
        assert!(result.is_err());
    }

    #[test]
    fn parse_unclosed_quote_at_eof_errors() {
        let p = default_parser();
        let result = parse_collect(&p, b"\"unclosed");
        assert!(result.is_err());
    }

    #[test]
    fn invalid_quote_reports_position() {
        // Strict mode: a quote inside an unquoted field is an error,
        // and the reported byte_offset must point at the offending quote.
        let p = strict_parser();
        let err = parse_collect(&p, b"a\"b,c\n").unwrap_err();
        match err {
            crate::Error::InvalidQuote { position } => {
                assert_eq!(position.byte_offset, 1);
            }
            e => panic!("expected InvalidQuote, got {e:?}"),
        }
    }

    #[test]
    fn unclosed_quote_reports_position() {
        // The unclosed quoted field starts at byte 8 (the `"` after
        // `foo,`). Verifies that `field_start` survives a record
        // boundary and the buffer-shift compaction.
        let p = default_parser();
        let err = parse_collect(&p, b"a,b\nfoo,\"bar\n").unwrap_err();
        match err {
            crate::Error::UnclosedQuote { position } => {
                assert_eq!(position.byte_offset, 8);
            }
            e => panic!("expected UnclosedQuote, got {e:?}"),
        }
    }

    #[test]
    fn unclosed_quote_reports_position_when_the_prefix_is_consumed() {
        // The opening quote produces no output, so the record in flight
        // leaves the write cursor aligned and nothing on `pending_spans`
        // — the all-clean consume. `field_in_start` still names the quote
        // at byte 2 and must survive being consumed past.
        let p = default_parser();
        let err = parse_collect(&p, b"x\n\"").unwrap_err();
        match err {
            crate::Error::UnclosedQuote { position } => {
                assert_eq!(position.byte_offset, 2);
            }
            e => panic!("expected UnclosedQuote, got {e:?}"),
        }
    }

    #[test]
    fn unclosed_quote_after_escape_reports_position_of_the_quote() {
        // The same, two bytes deep: quote then escape, both without
        // output.
        let mut config = Config::default();
        config.escape = Some(b'\\');
        let p = DfaChunkParser::new(config);
        let err = parse_collect(&p, b"x\n\"\\").unwrap_err();
        match err {
            crate::Error::UnclosedQuote { position } => {
                assert_eq!(position.byte_offset, 2);
            }
            e => panic!("expected UnclosedQuote, got {e:?}"),
        }
    }

    #[test]
    fn utf8_error_reports_position() {
        // 0xFF at byte 6 is not valid UTF-8; the validate_chunk path
        // should report it before any DFA stepping.
        let p = default_parser();
        let err = parse_collect(&p, b"hello,\xFFworld\n").unwrap_err();
        match err {
            crate::Error::Utf8 { position } => {
                assert_eq!(position.byte_offset, 6);
            }
            e => panic!("expected Utf8, got {e:?}"),
        }
    }

    #[test]
    fn parse_trailing_comma() {
        let p = default_parser();
        let records = parse_collect(&p, b"a,b,\n").unwrap();
        assert_eq!(records, vec![vec!["a", "b", ""]]);
    }

    #[test]
    fn parse_single_field() {
        let p = default_parser();
        let records = parse_collect(&p, b"hello\nworld\n").unwrap();
        assert_eq!(records, vec![vec!["hello"], vec!["world"]]);
    }

    // ── ChunkReader tests ──────────────────────────────────────────

    /// A simple in-memory ChunkReader for testing, backed by owned mutable data.
    struct VecChunkReader {
        data: Vec<u8>,
        pos: usize,
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

    /// A ChunkReader that exposes its data through a sliding window and
    /// returns an I/O error once the parser asks to read past `fail_at`.
    /// Used to verify that a mid-chunk read failure propagates instead of
    /// being silently swallowed (see `docs/swallowed-read-error.md`).
    struct FailAtReader {
        data: Vec<u8>,
        pos: usize,
        revealed: usize,
        fail_at: usize,
    }

    impl FailAtReader {
        fn new(data: &[u8], fail_at: usize) -> Self {
            Self {
                data: data.to_vec(),
                pos: 0,
                revealed: 0,
                fail_at,
            }
        }
    }

    impl ChunkReader for FailAtReader {
        fn buffer(&self) -> &[u8] {
            &self.data[self.pos..self.revealed]
        }

        fn buffer_mut(&mut self) -> &mut [u8] {
            &mut self.data[self.pos..self.revealed]
        }

        fn fill(&mut self, n: usize) -> std::io::Result<()> {
            let want = self.pos + n;
            if want > self.fail_at {
                // Simulate a real read() failure mid-chunk (bad sector,
                // NFS timeout, O_DIRECT EINVAL, ...).
                return Err(std::io::Error::other("simulated mid-chunk read failure"));
            }
            self.revealed = self.revealed.max(want.min(self.data.len()));
            Ok(())
        }

        fn consume(&mut self, n: usize) {
            self.pos += n;
        }

        fn remaining_in_chunk(&self) -> usize {
            // Treat the whole input as one chunk so the parser keeps
            // reading until it either finishes or hits the failure.
            self.data.len().saturating_sub(self.pos)
        }
    }

    /// Regression test for the swallowed-read-error bug
    /// (`docs/swallowed-read-error.md`). When a `ChunkReader::fill` fails
    /// after the chunk has already produced some records, the driver must
    /// surface the error rather than returning a truncated `Ok`.
    #[test]
    fn mid_chunk_read_error_propagates() {
        let p = default_parser();
        // Eight records; `fill` fails once the parser tries to read past
        // byte 10 — well after it has emitted the first records.
        let data = b"a,b\nc,d\ne,f\ng,h\ni,j\nk,l\nm,n\no,p\n";
        let mut reader = FailAtReader::new(data, 10);
        let pr = p.parse(
            &mut reader,
            0usize,
            Assumption::OutOfQuotes,
            &|count: &mut usize, _fields: &mut [&mut str]| {
                *count += 1;
                Ok(())
            },
            FindRecordStart::No,
        );
        // Pre-fix this returned `Ok` with a partial count; the underlying
        // io::Error must now propagate.
        assert!(
            pr.result.is_err(),
            "mid-chunk read error was swallowed instead of surfaced",
        );
    }

    /// Helper: parse a chunk and collect records (strict find by default).
    fn chunk_collect(
        parser: &DfaChunkParser,
        data: &[u8],
        chunk_end: usize,
        assumption: Assumption,
    ) -> PassResult<Vec<Vec<String>>> {
        chunk_collect_with(parser, data, chunk_end, assumption, FindRecordStart::Strict)
    }

    /// Helper: parse a chunk with explicit find mode.
    fn chunk_collect_with(
        parser: &DfaChunkParser,
        data: &[u8],
        chunk_end: usize,
        assumption: Assumption,
        find: FindRecordStart,
    ) -> PassResult<Vec<Vec<String>>> {
        let mut reader = VecChunkReader::new(data, chunk_end);
        parser.parse(
            &mut reader,
            Vec::<Vec<String>>::new(),
            assumption,
            &|state: &mut Vec<Vec<String>>, fields: &mut [&mut str]| {
                state.push(fields.iter().map(|s| s.to_string()).collect());
                Ok(())
            },
            find,
        )
    }

    // ── find_record_start tests ────────────────────────────────────

    #[test]
    fn chunk_find_record_start_skips_partial() {
        let p = default_parser();
        let data = b"eld2\nrec2_f1,rec2_f2\n";
        let pr = chunk_collect(&p, data, data.len(), Assumption::OutOfQuotes);
        assert_eq!(pr.record_start, Some(4));
        let records = pr.result.unwrap();
        assert_eq!(records, vec![vec!["rec2_f1", "rec2_f2"]]);
    }

    #[test]
    fn chunk_find_record_start_in_quotes() {
        let p = default_parser();
        let data = b"oted\"\nrec2\n";
        let pr = chunk_collect(&p, data, data.len(), Assumption::InQuotes);
        let records = pr.result.unwrap();
        assert_eq!(records, vec![vec!["rec2"]]);
    }

    #[test]
    fn chunk_find_record_start_in_quotes_with_delimiter() {
        let p = strict_parser();
        let data = b", safe ice circulation is generally from mid-November to mid-April. Geography Toponymy At various times in history, this territory has been occupied by the Attikameks, the Algonquin and the Cree. The toponym \"\"Ventadour River\"\" was made official on December 5, 1968, at the Commission de toponymie du Qu\xE9bec, when it was created. Notes and references See also Rivers of Nord-du-Qu\xE9bec Jam\xE9sie Nottaway River drainage basin\"\nnext record";
        let pr = chunk_collect(&p, data, data.len(), Assumption::InQuotes);
        let records = pr.result.unwrap();
        assert_eq!(records, vec![vec!["next record"]]);
    }

    #[test]
    fn find_record_start_stops_at_chunk_end() {
        let p = default_parser();
        let mut data = vec![b'x'; 5000];
        data.push(b'\n');
        data.extend_from_slice(b"rec,data\n");
        let chunk_end = 1024;
        let mut reader = VecChunkReader::new(&data, chunk_end);

        let result = p.scan_record_start(&mut reader, Assumption::OutOfQuotes, true);
        assert!(
            matches!(result, Ok(None)),
            "expected Ok(None), got {result:?} — scan crossed chunk_end",
        );
        assert!(
            reader.pos <= chunk_end,
            "reader advanced past chunk_end: pos={}, chunk_end={}",
            reader.pos,
            chunk_end,
        );
    }

    #[test]
    fn chunk_find_record_start_in_quotes_after_escape() {
        let mut config = Config::default();
        config.escape = Some(b'\\');
        let p = DfaChunkParser::new(config);
        let data = b"x\",next\nrec2\n";
        let pr = chunk_collect(&p, data, data.len(), Assumption::InQuotesAfterEscape);
        let records = pr.result.unwrap();
        assert_eq!(records, vec![vec!["rec2"]]);
    }

    // ── parse_records tests ────────────────────────────────────────

    #[test]
    fn chunk_stops_past_boundary() {
        let p = default_parser();
        let data = b"a,b\nc,d\ne,f\n";
        let pr = chunk_collect(&p, data, 5, Assumption::OutOfQuotes);
        let records = pr.result.unwrap();
        assert_eq!(records, vec![vec!["c", "d"]]);
    }

    #[test]
    fn chunk_finishes_record_past_boundary() {
        let p = default_parser();
        let data = b"a\nc,d\ne\n";
        let pr = chunk_collect(&p, data, 5, Assumption::OutOfQuotes);
        let records = pr.result.unwrap();
        assert_eq!(records, vec![vec!["c", "d"]]);
    }

    #[test]
    fn chunk_no_records_all_partial() {
        let p = default_parser();
        let data = b"still in prev record";
        let pr = chunk_collect(&p, data, data.len(), Assumption::OutOfQuotes);
        assert!(pr.record_start.is_none());
        assert!(pr.record_end.is_none());
        let records = pr.result.unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn chunk_record_end_is_the_input_end_not_the_write_cursor() {
        // Last field is quoted and unterminated by a newline, so the
        // compaction cursor ends two bytes short of the input.
        let p = default_parser();
        let data = b"a,b\nc,\"d\"";
        let pr = chunk_collect_with(
            &p,
            data,
            data.len(),
            Assumption::OutOfQuotes,
            FindRecordStart::No,
        );
        assert_eq!(pr.result.unwrap(), vec![vec!["a", "b"], vec!["c", "d"]]);
        assert_eq!(pr.record_end, Some(data.len()));
    }

    #[test]
    fn chunk_error_preserves_first_record_start() {
        let p = strict_parser();
        let data = b"a\nb\"c\n";
        let pr = chunk_collect(&p, data, data.len(), Assumption::OutOfQuotes);
        assert_eq!(pr.record_start, Some(1));
        assert!(pr.result.is_err());
    }

    /// Verdict per assumption, in `[OutOfQuotes, InQuotes]` order:
    /// `Ok(Some(n))` accepted with a boundary at `n`, `Ok(None)` accepted
    /// but no boundary, `Err(())` rejected.
    type ProbeVerdicts = [Result<Option<usize>, ()>; 2];

    fn probe(p: &DfaChunkParser, data: &[u8]) -> ProbeVerdicts {
        [Assumption::OutOfQuotes, Assumption::InQuotes].map(|a| {
            let pr = chunk_collect_with(p, data, data.len(), a, FindRecordStart::Strict);
            match (pr.record_start, pr.result) {
                (start, Ok(_)) => Ok(start),
                (_, Err(_)) => Err(()),
            }
        })
    }

    /// The probe picks the assumption, so it must reject exactly the
    /// misplaced quotes — an opener not at a field boundary, a closer not
    /// followed by one — and nothing else, in every dialect.
    #[test]
    fn strict_probe_rejects_exactly_the_misplaced_quotes() {
        let cases: &[(&[u8], ProbeVerdicts, &str)] = &[
            // Opener at offset 0: the chunk handoff counts as a boundary,
            // so this is the legal start of a quoted field.
            (
                b"\"m\nn\"\nt,u\n",
                [Ok(Some(5)), Err(())],
                "opener at offset 0",
            ),
            (
                b"a,\"m\nn\"\nt,u\n",
                [Ok(Some(7)), Err(())],
                "opener after a delimiter",
            ),
            // Opener inside an unquoted field.
            (b"ab\"m\nn\"\nt,u\n", [Err(()), Err(())], "opener mid-field"),
            // Closer not followed by a delimiter or terminator.
            (
                b"\"mn\"x,y\nt,u\n",
                [Err(()), Err(())],
                "closer before bare text",
            ),
            // No quotes: in-quotes is coherent but finds no boundary.
            (b"ab,cd\nt,u\n", [Ok(Some(5)), Ok(None)], "no quotes"),
            // Chunk opens inside a quoted field.
            (
                b"mn\",x\nt,u\n",
                [Err(()), Ok(Some(5))],
                "opens inside a quoted field",
            ),
        ];
        for qh in [
            QuoteHandling::Toggle,
            QuoteHandling::Literal,
            QuoteHandling::Strict,
        ] {
            let mut config = Config::default();
            config.quote_handling = qh;
            let p = DfaChunkParser::new(config);
            for (data, expected, label) in cases {
                assert_eq!(&probe(&p, data), expected, "{qh:?}: {label}");
            }
        }
    }
}
