use crate::config::{Config, QuoteHandling, RecordTerminator};

/// DFA states for CSV parsing.
///
/// Laid out so that interior states (where parsing continues byte by byte)
/// occupy the low indices and action states (field end / record end / error)
/// occupy the high indices. This lets the hot loop detect boundaries with a
/// single `state as u8 >= FIRST_ACTION` comparison, rather than a tagged
/// variant match that compiles to an indirect jump.
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(u8)]
pub(crate) enum State {
    /// Inside an unquoted field.
    InField = 0,
    /// Inside a quoted field.
    InQuotedField = 1,
    /// Beginning of a record.
    RecordStart = 2,
    /// Beginning of a field. Decides quoted vs unquoted.
    FieldStart = 3,
    /// After a closing quote in a quoted field.
    AfterClosingQuote = 4,
    /// After an escape character inside a quoted field. Next byte is literal.
    InEscapedQuote = 5,
    /// Inside a comment line, consuming until newline.
    InComment = 6,

    /// A field just ended (delimiter consumed).
    FieldEnd = 7,
    /// A record just ended (terminator consumed).
    RecordEnd = 8,
    /// Parse error (e.g. quote not at a field boundary in strict quote style).
    Error = 9,
}

const NUM_STATES: usize = 10;

/// All states with a numeric value >= this are action states.
pub(crate) const FIRST_ACTION: u8 = State::FieldEnd as u8;
/// State value for the RecordEnd action.
pub(crate) const RECORD_END_STATE: u8 = State::RecordEnd as u8;
/// State value for the Error action.
pub(crate) const ERROR_STATE: u8 = State::Error as u8;

/// A single DFA transition: next state and whether the current byte should
/// be emitted as part of the field.
///
/// Packed as a `u16` so a lookup is one load and the hot loop can pull out
/// `next` and `has_output` with a pair of cheap integer ops.
///
/// Bit layout:
/// - bits 0..=7 — next state discriminant
/// - bit 8     — `HAS_OUTPUT`
/// - bit 9     — `STRICT_ERROR` (probe-mode-only error, see `is_strict_error`)
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct Transition(u16);

impl Transition {
    /// Bit set when the byte should be written to the output buffer.
    const HAS_OUTPUT: u16 = 1 << 8;
    /// Bit set when the strict probe in `find_record_start` should treat
    /// this cell as an error.
    const STRICT_ERROR: u16 = 1 << 9;

    #[inline(always)]
    pub const fn new(next: State, has_output: bool) -> Self {
        Self((next as u16) | if has_output { Self::HAS_OUTPUT } else { 0 })
    }

    /// Return a copy of this transition with the `STRICT_ERROR` bit set.
    #[inline(always)]
    pub const fn with_strict_error(self) -> Self {
        Self(self.0 | Self::STRICT_ERROR)
    }

    /// Raw next-state byte.
    #[inline(always)]
    pub fn next_raw(self) -> u8 {
        self.0 as u8
    }

    #[inline(always)]
    pub fn has_output(self) -> bool {
        (self.0 & Self::HAS_OUTPUT) != 0
    }

    /// Whether the strict probe should reject this cell.
    #[inline(always)]
    pub fn is_strict_error(self) -> bool {
        (self.0 & Self::STRICT_ERROR) != 0
    }

    #[inline(always)]
    pub fn next_state(self) -> State {
        // Safety: table entries are only constructed via `new` with a valid
        // `State`, so the low byte is always a valid `State` discriminant.
        unsafe { std::mem::transmute::<u8, State>(self.next_raw()) }
    }
}

/// Up to four structural bytes for the `InField` / `InQuotedField` fast
/// paths, in a fixed-size array so the scan can dispatch on the needle
/// count without allocating.
#[derive(Debug, Copy, Clone)]
pub(crate) struct StructuralNeedles {
    /// The needle bytes themselves; only the first `count` are meaningful.
    bytes: [u8; 4],
    /// `bytes[i]` broadcast to all eight lanes for the SWAR scan; unused lanes repeat `bytes[0]`.
    lanes: [u64; 4],
    /// Number of needles actually present, at most four.
    count: u8,
}

const SWAR_LO: u64 = u64::from_ne_bytes([0x01; 8]);
const SWAR_HI: u64 = u64::from_ne_bytes([0x80; 8]);

impl StructuralNeedles {
    fn from_mask(mask: &[bool; 256]) -> Self {
        let mut bytes = [0u8; 4];
        let mut count: u8 = 0;
        for b in 0u8..=255 {
            if mask[b as usize] {
                debug_assert!(
                    (count as usize) < bytes.len(),
                    "more than 4 structural bytes in mask — adjust StructuralNeedles capacity",
                );
                if (count as usize) < bytes.len() {
                    bytes[count as usize] = b;
                    count += 1;
                }
            }
        }
        let mut lanes = [u64::from_ne_bytes([bytes[0]; 8]); 4];
        for i in 0..count as usize {
            lanes[i] = u64::from_ne_bytes([bytes[i]; 8]);
        }
        Self {
            bytes,
            lanes,
            count,
        }
    }

    /// Offset of the first byte of `word` matching one of the first `N`
    /// needles. Per needle a `haszero(x ^ broadcast)` test tags the high
    /// bit of every matching lane; the lowest tagged lane wins. `N` is
    /// const so the loop unrolls.
    #[inline(always)]
    fn match_lanes<const N: usize>(&self, word: u64) -> Option<usize> {
        let mut hits = 0u64;
        for &lane in &self.lanes[..N] {
            let x = word ^ lane;
            hits |= x.wrapping_sub(SWAR_LO) & !x & SWAR_HI;
        }
        if hits == 0 {
            return None;
        }
        Some(if cfg!(target_endian = "little") {
            hits.trailing_zeros() as usize / 8
        } else {
            hits.leading_zeros() as usize / 8
        })
    }

    /// `find` for a statically known needle count.
    #[inline(always)]
    fn find_n<const N: usize>(&self, hay: &[u8]) -> Option<usize> {
        let mut pos = 0;
        while pos + 8 <= hay.len() {
            let word = u64::from_ne_bytes(hay[pos..pos + 8].try_into().unwrap());
            if let Some(off) = self.match_lanes::<N>(word) {
                return Some(pos + off);
            }
            pos += 8;
        }
        if pos < hay.len() {
            if hay.len() >= 8 {
                let start = hay.len() - 8;
                let word = u64::from_ne_bytes(hay[start..start + 8].try_into().unwrap());
                if let Some(off) = self.match_lanes::<N>(word) {
                    debug_assert!(start + off >= pos, "match below an already-scanned offset");
                    return Some(start + off);
                }
            } else {
                // Shorter than one word: at most 7 iterations.
                let needles = &self.bytes[..N];
                return hay.iter().position(|b| needles.contains(b));
            }
        }
        None
    }

    /// Offset of the next structural byte in `hay`, or `None` if none.
    #[inline(always)]
    pub fn find(&self, hay: &[u8]) -> Option<usize> {
        match self.count {
            1 => self.find_n::<1>(hay),
            2 => self.find_n::<2>(hay),
            3 => self.find_n::<3>(hay),
            4 => self.find_n::<4>(hay),
            _ => None,
        }
    }
}

/// A precomputed CSV state machine.
///
/// All configuration (delimiter, quote, escape, strictness, terminator,
/// comment) is baked into the transition table at construction time. The
/// hot path is a single table lookup plus a handful of branchless updates.
///
/// Cells that would be `Error` only under `QuoteHandling::Strict` are kept
/// at their parse-mode (lenient/literal) `next_state` and tagged with the
/// `STRICT_ERROR` bit. The strict probe in `find_record_start` checks
/// that bit to reject wrong assumptions; the parsing hot loop ignores it.
#[derive(Debug)]
pub(crate) struct Dfa {
    table: [[Transition; 256]; NUM_STATES],
    /// Byte mask for `RecordStart`. Non-structural bytes are terminator
    /// bytes that self-loop in `RecordStart` with no output (empty-line
    /// skip); the scan-only fast path advances `pos` past runs of them
    /// without going through the table.
    record_start_structural: [bool; 256],
    /// Structural needle sets for the two "content" states. Their
    /// non-structural bytes loop back with `has_output = true`, so once
    /// the scan jumps past a run it can be bulk-shifted with a single
    /// `copy_within`.
    in_field_needles: StructuralNeedles,
    in_quoted_needles: StructuralNeedles,
}

impl Dfa {
    /// Build a DFA transition table from the given configuration.
    ///
    /// Cells that would be `Error` only under `QuoteHandling::Strict` keep
    /// their parse-mode `next_state` and gain the `STRICT_ERROR` flag,
    /// so a single table serves both the user's parse semantics and the
    /// strict probe used by `find_record_start`.
    pub fn new(config: &Config) -> Self {
        let mut table = [[Transition::new(State::Error, false); 256]; NUM_STATES];
        let mut in_field_structural = [false; 256];
        let mut in_quoted_structural = [false; 256];
        let mut record_start_structural = [true; 256];

        let quote_handling = config.quote_handling;
        let doubled_quotes = matches!(
            (config.quote, config.escape),
            (Some(q), Some(e)) if q == e
        );

        // First pass: build every row that is reached without replay.
        for byte in 0u8..=255 {
            let b = byte;

            let is_terminator = match config.terminator {
                RecordTerminator::CRLF => b == b'\r' || b == b'\n',
                RecordTerminator::Byte(t) => b == t,
            };

            // ── FieldStart ───────────────────────────────────────────
            table[State::FieldStart as usize][b as usize] = if config.quote == Some(b) {
                Transition::new(State::InQuotedField, false)
            } else if b == config.delimiter {
                Transition::new(State::FieldEnd, false)
            } else if is_terminator {
                Transition::new(State::RecordEnd, false)
            } else {
                Transition::new(State::InField, true)
            };

            // ── InField ──────────────────────────────────────────────
            let (in_field_trans, in_field_struct) = if b == config.delimiter {
                (Transition::new(State::FieldEnd, false), true)
            } else if is_terminator {
                (Transition::new(State::RecordEnd, false), true)
            } else if config.quote == Some(b) {
                // Toggle/Literal: the cell follows the user's chosen
                // semantics but is tagged STRICT_ERROR so the probe
                // rejects it. Literal keeps the byte as content, hence
                // not structural for parse mode.
                let t = match quote_handling {
                    QuoteHandling::Toggle => {
                        Transition::new(State::InQuotedField, false).with_strict_error()
                    }
                    QuoteHandling::Literal => {
                        Transition::new(State::InField, true).with_strict_error()
                    }
                    QuoteHandling::Strict => Transition::new(State::Error, false),
                };
                (t, quote_handling != QuoteHandling::Literal)
            } else {
                (Transition::new(State::InField, true), false)
            };
            table[State::InField as usize][b as usize] = in_field_trans;
            in_field_structural[b as usize] = in_field_struct;

            // ── InQuotedField ────────────────────────────────────────
            let (in_quoted_trans, in_quoted_struct) = if config.escape == Some(b) && !doubled_quotes
            {
                (Transition::new(State::InEscapedQuote, false), true)
            } else if config.quote == Some(b) {
                (Transition::new(State::AfterClosingQuote, false), true)
            } else {
                (Transition::new(State::InQuotedField, true), false)
            };
            table[State::InQuotedField as usize][b as usize] = in_quoted_trans;
            in_quoted_structural[b as usize] = in_quoted_struct;

            // ── AfterClosingQuote ────────────────────────────────────
            table[State::AfterClosingQuote as usize][b as usize] =
                if doubled_quotes && config.quote == Some(b) {
                    Transition::new(State::InQuotedField, true)
                } else if b == config.delimiter {
                    Transition::new(State::FieldEnd, false)
                } else if is_terminator {
                    Transition::new(State::RecordEnd, false)
                } else if quote_handling == QuoteHandling::Strict {
                    Transition::new(State::Error, false)
                } else if quote_handling == QuoteHandling::Toggle && config.quote == Some(b) {
                    // Every quote toggles under Toggle — re-enter the
                    // quoted field rather than smoothing into InField.
                    Transition::new(State::InQuotedField, false).with_strict_error()
                } else {
                    // Toggle/Literal: keep the lenient continuation but
                    // flag the probe — a non-structural byte after a
                    // closing quote means a wrong InQuotes assumption.
                    Transition::new(State::InField, true).with_strict_error()
                };

            // ── InEscapedQuote ───────────────────────────────────────
            // Any byte is literal content in the quoted field.
            table[State::InEscapedQuote as usize][b as usize] =
                Transition::new(State::InQuotedField, true);

            // ── InComment ────────────────────────────────────────────
            table[State::InComment as usize][b as usize] =
                if b == b'\n' || (config.terminator == RecordTerminator::CRLF && b == b'\r') {
                    Transition::new(State::RecordStart, false)
                } else {
                    Transition::new(State::InComment, false)
                };
        }

        // Second pass: epsilon-expand RecordStart. Any byte that would
        // have been `Replay(FieldStart)` is replaced by whatever
        // `FieldStart` does with that byte; this eliminates the Replay
        // variant so every main-loop iteration advances `pos` exactly
        // once.
        for byte in 0u8..=255 {
            let b = byte;
            let is_terminator = match config.terminator {
                RecordTerminator::CRLF => b == b'\r' || b == b'\n',
                RecordTerminator::Byte(t) => b == t,
            };

            table[State::RecordStart as usize][b as usize] = if is_terminator {
                // Empty line: skip terminator, stay in RecordStart.
                record_start_structural[b as usize] = false;
                Transition::new(State::RecordStart, false)
            } else if config.comment == Some(b) {
                Transition::new(State::InComment, false)
            } else {
                // Replay to FieldStart — inline its transition.
                table[State::FieldStart as usize][b as usize]
            };
        }

        let in_field_needles = StructuralNeedles::from_mask(&in_field_structural);
        let in_quoted_needles = StructuralNeedles::from_mask(&in_quoted_structural);

        Self {
            table,
            record_start_structural,
            in_field_needles,
            in_quoted_needles,
        }
    }

    /// Look up the transition for the given state and byte.
    #[inline(always)]
    pub fn transition(&self, state: State, byte: u8) -> Transition {
        self.table[state as usize][byte as usize]
    }

    /// Byte mask of structural bytes in `RecordStart`. Non-structural bytes
    /// are terminators that self-loop without producing output.
    #[inline(always)]
    pub fn record_start_structural(&self) -> &[bool; 256] {
        &self.record_start_structural
    }

    /// Structural needle set for `InField`, for the scan fast path.
    /// Excludes the quote byte under `QuoteHandling::Literal` (where a
    /// quote in an unquoted field is content) — so the strict probe has
    /// to run on a Toggle-built table, not this one.
    #[inline(always)]
    pub fn in_field_needles(&self) -> &StructuralNeedles {
        &self.in_field_needles
    }

    /// Structural needle set for `InQuotedField`, for the scan fast path.
    #[inline(always)]
    pub fn in_quoted_needles(&self) -> &StructuralNeedles {
        &self.in_quoted_needles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_dfa() -> Dfa {
        Dfa::new(&Config::default())
    }

    fn literal_dfa() -> Dfa {
        let mut config = Config::default();
        config.quote_handling = QuoteHandling::Literal;
        Dfa::new(&config)
    }

    fn strict_dfa() -> Dfa {
        let mut config = Config::default();
        config.quote_handling = QuoteHandling::Strict;
        Dfa::new(&config)
    }

    /// Short alias for building expected transitions in tests.
    fn t(next: State, has_output: bool) -> Transition {
        Transition::new(next, has_output)
    }

    /// Same as `t`, but with the strict-probe error flag set.
    fn te(next: State, has_output: bool) -> Transition {
        Transition::new(next, has_output).with_strict_error()
    }

    // -- RecordStart --

    #[test]
    fn record_start_skips_lf() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::RecordStart, b'\n'),
            t(State::RecordStart, false)
        );
    }

    #[test]
    fn record_start_skips_cr() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::RecordStart, b'\r'),
            t(State::RecordStart, false)
        );
    }

    #[test]
    fn record_start_expanded_to_field_start_for_regular_byte() {
        let d = default_dfa();
        // Epsilon-expanded: RecordStart + 'a' should behave like FieldStart + 'a'.
        assert_eq!(
            d.transition(State::RecordStart, b'a'),
            t(State::InField, true)
        );
    }

    #[test]
    fn record_start_expanded_to_field_end_for_delimiter() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::RecordStart, b','),
            t(State::FieldEnd, false)
        );
    }

    // -- FieldStart --

    #[test]
    fn field_start_quote_enters_quoted_field() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::FieldStart, b'"'),
            t(State::InQuotedField, false)
        );
    }

    #[test]
    fn field_start_delimiter_ends_empty_field() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::FieldStart, b','),
            t(State::FieldEnd, false)
        );
    }

    #[test]
    fn field_start_lf_ends_record() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::FieldStart, b'\n'),
            t(State::RecordEnd, false)
        );
    }

    #[test]
    fn field_start_cr_ends_record() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::FieldStart, b'\r'),
            t(State::RecordEnd, false)
        );
    }

    #[test]
    fn field_start_regular_byte_enters_field() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::FieldStart, b'x'),
            t(State::InField, true)
        );
    }

    // -- InField --

    #[test]
    fn in_field_delimiter_ends_field() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::InField, b','),
            t(State::FieldEnd, false)
        );
    }

    #[test]
    fn in_field_lf_ends_record() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::InField, b'\n'),
            t(State::RecordEnd, false)
        );
    }

    #[test]
    fn in_field_regular_byte_continues() {
        let d = default_dfa();
        assert_eq!(d.transition(State::InField, b'z'), t(State::InField, true));
    }

    #[test]
    fn in_field_quote_lenient_toggles() {
        let d = default_dfa();
        // STRICT_ERROR flag is set so the probe rejects, but parse-mode
        // semantics (toggle into the quoted field) are preserved.
        assert_eq!(
            d.transition(State::InField, b'"'),
            te(State::InQuotedField, false)
        );
    }

    #[test]
    fn in_field_quote_literal_emits() {
        let d = literal_dfa();
        assert_eq!(d.transition(State::InField, b'"'), te(State::InField, true));
    }

    #[test]
    fn in_field_quote_strict_errors() {
        let d = strict_dfa();
        assert_eq!(d.transition(State::InField, b'"'), t(State::Error, false));
    }

    // -- InQuotedField --

    #[test]
    fn in_quoted_field_regular_byte_emits() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::InQuotedField, b'a'),
            t(State::InQuotedField, true)
        );
    }

    #[test]
    fn in_quoted_field_delimiter_emits() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::InQuotedField, b','),
            t(State::InQuotedField, true)
        );
    }

    #[test]
    fn in_quoted_field_newline_emits() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::InQuotedField, b'\n'),
            t(State::InQuotedField, true)
        );
    }

    #[test]
    fn in_quoted_field_quote_enters_after_closing() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::InQuotedField, b'"'),
            t(State::AfterClosingQuote, false)
        );
    }

    #[test]
    fn in_quoted_field_backslash_escape() {
        let mut config = Config::default();
        config.escape = Some(b'\\');
        let d = Dfa::new(&config);
        assert_eq!(
            d.transition(State::InQuotedField, b'\\'),
            t(State::InEscapedQuote, false)
        );
    }

    // -- AfterClosingQuote --

    #[test]
    fn after_closing_quote_doubled_quote_emits() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::AfterClosingQuote, b'"'),
            t(State::InQuotedField, true)
        );
    }

    #[test]
    fn after_closing_quote_delimiter_ends_field() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::AfterClosingQuote, b','),
            t(State::FieldEnd, false)
        );
    }

    #[test]
    fn after_closing_quote_lf_ends_record() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::AfterClosingQuote, b'\n'),
            t(State::RecordEnd, false)
        );
    }

    #[test]
    fn after_closing_quote_other_strict_errors() {
        let d = strict_dfa();
        assert_eq!(
            d.transition(State::AfterClosingQuote, b'x'),
            t(State::Error, false)
        );
    }

    #[test]
    fn after_closing_quote_other_lenient_continues() {
        let d = default_dfa();
        // Permissive continuation, but tagged STRICT_ERROR so the probe
        // rejects a wrong InQuotes assumption.
        assert_eq!(
            d.transition(State::AfterClosingQuote, b'x'),
            te(State::InField, true)
        );
    }

    #[test]
    fn strict_error_flag_only_set_on_lenient_cells() {
        let d = default_dfa();
        // STRICT_ERROR sits on the cells that would have been Error
        // under QuoteHandling::Strict.
        assert!(d.transition(State::InField, b'"').is_strict_error());
        assert!(
            d.transition(State::AfterClosingQuote, b'x')
                .is_strict_error()
        );
        // Regular cells don't carry the flag.
        assert!(!d.transition(State::InField, b'a').is_strict_error());
        assert!(!d.transition(State::FieldStart, b',').is_strict_error());
    }

    #[test]
    fn strict_error_flag_absent_under_strict_quote_handling() {
        // Under QuoteHandling::Strict the cell is a real Error, not a
        // probe-only flag.
        let d = strict_dfa();
        let trans = d.transition(State::InField, b'"');
        assert!(!trans.is_strict_error());
        assert_eq!(trans.next_raw(), ERROR_STATE);
    }

    // -- InEscapedQuote --

    #[test]
    fn in_escaped_quote_any_byte_emits() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::InEscapedQuote, b'"'),
            t(State::InQuotedField, true)
        );
        assert_eq!(
            d.transition(State::InEscapedQuote, b'\n'),
            t(State::InQuotedField, true)
        );
        assert_eq!(
            d.transition(State::InEscapedQuote, b'x'),
            t(State::InQuotedField, true)
        );
    }

    // -- Structural needles --

    fn needles_set(n: &StructuralNeedles) -> std::collections::HashSet<u8> {
        n.bytes[..n.count as usize].iter().copied().collect()
    }

    #[test]
    fn in_field_needles_cover_structural_bytes() {
        let d = default_dfa();
        let set = needles_set(d.in_field_needles());
        // Default dialect (CRLF terminator, Toggle quote handling): {',', '\n', '\r', '"'}
        assert_eq!(set, b",\n\r\"".iter().copied().collect());
    }

    #[test]
    fn in_field_needles_literal_excludes_quote() {
        let d = literal_dfa();
        let set = needles_set(d.in_field_needles());
        // Literal quote style treats `"` as content in InField.
        assert_eq!(set, b",\n\r".iter().copied().collect());
    }

    #[test]
    fn in_quoted_needles_cover_quote_only() {
        let d = default_dfa();
        let set = needles_set(d.in_quoted_needles());
        // No escape configured → just the closing quote.
        assert_eq!(set, b"\"".iter().copied().collect());
    }

    #[test]
    fn in_quoted_needles_with_backslash_escape() {
        let mut config = Config::default();
        config.escape = Some(b'\\');
        let d = Dfa::new(&config);
        let set = needles_set(d.in_quoted_needles());
        assert_eq!(set, b"\"\\".iter().copied().collect());
    }

    #[test]
    fn structural_needles_find_basics() {
        let d = default_dfa();
        let hay = b"abcdefg,hij\nklm";
        let n = d.in_field_needles();
        assert_eq!(n.find(hay), Some(7)); // first structural byte is ','
        assert_eq!(n.find(&hay[8..]), Some(3)); // next is '\n' at relative 3
        assert_eq!(n.find(b"no-structural-bytes-here"), None);
    }

    /// Differential test against a naive scan across the SWAR word
    /// boundary.
    #[test]
    fn structural_needles_find_matches_naive_scan() {
        fn naive(needles: &[u8], hay: &[u8]) -> Option<usize> {
            hay.iter().position(|b| needles.contains(b))
        }

        for count in 1..=4usize {
            let needles: Vec<u8> = (b'0'..).take(count).collect();
            let mut mask = [false; 256];
            for &b in &needles {
                mask[b as usize] = true;
            }
            let n = StructuralNeedles::from_mask(&mask);

            for len in 0..40usize {
                let clean = vec![b'x'; len];
                assert_eq!(n.find(&clean), naive(&needles, &clean), "clean len {len}");

                for at in 0..len {
                    for &b in &needles {
                        let mut hay = clean.clone();
                        hay[at] = b;
                        assert_eq!(
                            n.find(&hay),
                            naive(&needles, &hay),
                            "count {count} len {len} at {at} byte {b}"
                        );
                    }
                }

                // Two matches: the earlier must win.
                for a in 0..len {
                    for b in a + 1..len {
                        let mut hay = clean.clone();
                        hay[a] = needles[0];
                        hay[b] = needles[count - 1];
                        assert_eq!(
                            n.find(&hay),
                            naive(&needles, &hay),
                            "count {count} len {len} pair {a},{b}"
                        );
                    }
                }
            }
        }
    }

    // -- Multi-byte sequences --

    #[test]
    fn doubled_quote_in_quoted_field() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::InQuotedField, b'"'),
            t(State::AfterClosingQuote, false)
        );
        assert_eq!(
            d.transition(State::AfterClosingQuote, b'"'),
            t(State::InQuotedField, true)
        );
    }

    #[test]
    fn empty_line_crlf_skipped() {
        let d = default_dfa();
        assert_eq!(
            d.transition(State::RecordStart, b'\r'),
            t(State::RecordStart, false)
        );
        assert_eq!(
            d.transition(State::RecordStart, b'\n'),
            t(State::RecordStart, false)
        );
    }
}
