use crate::io::ringbuf::RingBufSettings;

/// The record terminator.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub enum RecordTerminator {
    /// A single byte.
    Byte(u8),
    /// Carriage return + line feed (`\r\n`).
    #[default]
    CRLF,
}

impl RecordTerminator {
    /// Line feed (`\n`) only.
    pub const LF: Self = Self::Byte(b'\n');
    /// Carriage return (`\r`) only.
    pub const CR: Self = Self::Byte(b'\r');

    /// The terminator byte set: the primary byte plus an optional
    /// second one. A maximal run over this set is one record boundary,
    /// which is how the SIMD parser unifies `\r\n`, empty lines and
    /// mixed newlines.
    pub(crate) fn bytes(self) -> (u8, Option<u8>) {
        match self {
            RecordTerminator::Byte(b) => (b, None),
            RecordTerminator::CRLF => (b'\n', Some(b'\r')),
        }
    }
}

/// The I/O backend to use for reading file data.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub enum IoBackend {
    /// Automatically select based on input type and file size.
    #[default]
    Auto,
    /// Memory-mapped I/O. Each thread gets its own MAP_PRIVATE mapping.
    Mmap,
    /// Ring buffer I/O. Each thread reads into its own buffer via
    /// syscalls. See [`RingBufSettings`] for tuning knobs.
    RingBuf(RingBufSettings),
    /// Per-chunk read from chunk start to file end into a per-thread buffer.
    InMemory,
}

/// How quote characters are handled.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub enum QuoteHandling {
    /// Every quote character unconditionally toggles quoting, regardless of
    /// position. Matches the behavior of Postgres and is the most
    /// vectorization-friendly mode.
    #[default]
    Toggle,
    /// RFC 4180-compliant: a quote only opens a quoted field at a field
    /// boundary; a quote anywhere inside an unquoted field is a parse error.
    Strict,
    /// A quote only opens a quoted field when it immediately follows a
    /// delimiter or record start. A quote mid-field is treated as a literal
    /// character. Matches the behavior of rust-csv and Python's `csv` module.
    Literal,
}

/// The chunk parser implementation to use.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub enum ParserBackend {
    /// Automatically select the fastest parser that supports the current config.
    #[default]
    Auto,
    /// Force the DFA-based parser. Works with any configuration.
    Dfa,
    /// Force the SIMD-vectorized parser. Requires the crate's `simd`
    /// feature (nightly-only); without it, selecting `Simd` errors at
    /// parse time. Also errors if the config is unsupported (e.g.
    /// trimming, flexible field counts, certain escape modes).
    Simd,
}

/// Internal parser configuration.
#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub delimiter: u8,
    pub terminator: RecordTerminator,
    pub quote: Option<u8>,
    pub escape: Option<u8>,
    pub comment: Option<u8>,
    pub has_headers: bool,
    pub concurrency: usize,
    pub chunk_size: usize,
    pub io_buffer_limit: Option<usize>,
    pub quote_handling: QuoteHandling,
    pub flexible: bool,
    pub field_count: Option<usize>,
    pub trim: bool,
    pub io_backend: IoBackend,
    pub parser_backend: ParserBackend,
}

impl Config {
    /// Check that the configuration is consistent.
    #[inline]
    pub(crate) fn validate(&self) -> crate::Result<()> {
        if let IoBackend::RingBuf(ring) = self.io_backend {
            if ring.buffer_size == 0 {
                return Err(crate::Error::InvalidConfig(
                    "ring buffer buffer_size must be at least 1",
                ));
            }
            // Some(0) is the auto sentinel
            if let Some(limit) = ring.buffer_limit
                && limit != 0
                && ring.buffer_size > limit
            {
                return Err(crate::Error::InvalidConfig(
                    "ring buffer buffer_size must be <= its buffer_limit",
                ));
            }
            if let Some(limit) = self.io_buffer_limit
                && limit != 0
                && ring.buffer_size > limit
            {
                return Err(crate::Error::InvalidConfig(
                    "io_buffer_limit is a total across threads and must leave room \
                     for at least one ring buffer",
                ));
            }
        }
        // a list of "control bytes" that must be distinct
        let (term_a, term_b) = self.terminator.bytes();
        let bytes = [
            ("delimiter", Some(self.delimiter)),
            ("terminator", Some(term_a)),
            ("terminator", term_b),
            ("quote", self.quote),
            ("escape", self.escape),
            ("comment", self.comment),
        ];
        for i in 0..bytes.len() {
            for j in (i + 1)..bytes.len() {
                if let (Some(a), Some(b)) = (bytes[i].1, bytes[j].1)
                    && a == b
                {
                    // quote/escape are commonly the same byte (RFC 4180 doubling)
                    if bytes[i].0 == "quote" && bytes[j].0 == "escape" {
                        continue;
                    }
                    return Err(crate::Error::InvalidConfig(
                        "delimiter, terminator, quote, escape, and comment must be distinct",
                    ));
                }
            }
        }
        if self.escape.is_some() && self.quote.is_none() {
            return Err(crate::Error::InvalidConfig(
                "quote must be set when escape is set",
            ));
        }
        Ok(())
    }

    /// Check that every control byte is ASCII.
    #[inline]
    pub(crate) fn validate_ascii_control_bytes(&self) -> crate::Result<()> {
        let (term_a, term_b) = self.terminator.bytes();
        let bytes = [
            Some(self.delimiter),
            Some(term_a),
            term_b,
            self.quote,
            self.escape,
            self.comment,
        ];
        if bytes.iter().flatten().any(|b| !b.is_ascii()) {
            return Err(crate::Error::InvalidConfig(
                "delimiter, terminator, quote, escape, and comment must be ASCII \
                 in text mode; use `ParserBuilder::bytes` for non-UTF-8 encodings",
            ));
        }
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            delimiter: b',',
            terminator: RecordTerminator::default(),
            quote: Some(b'"'),
            escape: Some(b'"'),
            comment: None,
            has_headers: true,
            // 0 means "auto-detect"
            concurrency: 0,
            // 0 means "auto-detect".
            chunk_size: 0,
            // Some(0) means "auto-detect"
            io_buffer_limit: Some(0),
            quote_handling: QuoteHandling::default(),
            flexible: false,
            // set by the parse functions
            field_count: None,
            trim: false,
            io_backend: IoBackend::default(),
            parser_backend: ParserBackend::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn allows_zero_concurrency_as_auto() {
        let mut c = Config::default();
        c.concurrency = 0;
        c.validate().unwrap();
    }

    #[test]
    fn allows_zero_chunk_size_as_auto() {
        let mut c = Config::default();
        c.chunk_size = 0;
        c.io_buffer_limit = Some(64 * 1024);
        // Sentinel chunk_size bypasses the chunk_size >= io_buffer_limit check.
        c.validate().unwrap();
    }

    #[test]
    fn rejects_delimiter_equals_quote() {
        let mut c = Config::default();
        c.delimiter = b'"';
        assert!(c.validate().is_err());
    }

    #[test]
    fn allows_quote_equals_escape() {
        let mut c = Config::default();
        c.quote = Some(b'"');
        c.escape = Some(b'"');
        c.validate().unwrap();
    }

    #[test]
    fn rejects_delimiter_cr_with_crlf() {
        let mut c = Config::default();
        c.delimiter = b'\r';
        c.terminator = RecordTerminator::CRLF;
        assert!(c.validate().is_err());
    }

    #[test]
    fn ascii_check_accepts_default_and_crlf() {
        Config::default().validate_ascii_control_bytes().unwrap();
        let mut c = Config::default();
        c.terminator = RecordTerminator::CRLF;
        c.validate_ascii_control_bytes().unwrap();
    }

    #[test]
    fn ascii_check_rejects_each_non_ascii_control_byte() {
        // 0xA6 is a lead byte, 0x80 a continuation byte — both would
        // slice a multi-byte sequence apart if used as a control byte.
        for setter in [
            (|c: &mut Config| c.delimiter = 0xA6) as fn(&mut Config),
            |c| c.terminator = RecordTerminator::Byte(0x80),
            |c| c.quote = Some(0xA6),
            |c| c.escape = Some(0x80),
            |c| c.comment = Some(0xA6),
        ] {
            // Non-ASCII bytes can't collide with the ASCII defaults, so
            // each config stays dialect-valid and isolates the ASCII check.
            let mut c = Config::default();
            setter(&mut c);
            assert!(
                c.validate_ascii_control_bytes().is_err(),
                "expected non-ASCII control byte to be rejected: {c:?}"
            );
            // `validate` is dialect-only and must stay indifferent to it.
            c.validate().unwrap();
        }
    }

    #[test]
    fn allows_chunk_smaller_than_buffer_limit() {
        let mut c = Config::default();
        c.chunk_size = 100;
        c.io_backend = IoBackend::RingBuf(RingBufSettings::default().buffer_size(50));
        c.io_buffer_limit = Some(200);
        c.validate().unwrap();
    }

    #[test]
    fn rejects_buffer_size_greater_than_the_parser_limit() {
        let mut c = Config::default();
        c.io_backend = IoBackend::RingBuf(RingBufSettings::default().buffer_size(64 * 1024));
        c.io_buffer_limit = Some(32 * 1024);
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_buffer_size_greater_than_its_own_limit() {
        let mut c = Config::default();
        c.io_backend = IoBackend::RingBuf(
            RingBufSettings::default()
                .buffer_size(64 * 1024)
                .buffer_limit(Some(32 * 1024)),
        );
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_zero_buffer_size() {
        let mut c = Config::default();
        c.io_backend = IoBackend::RingBuf(RingBufSettings::default().buffer_size(0));
        assert!(c.validate().is_err());
    }

    #[test]
    fn ring_buffer_checks_only_apply_when_the_backend_is_named() {
        // The payload is the point: settings that cannot be set for a
        // backend cannot be wrong for it either. A tiny limit is fine
        // under mmap, which has no buffer to size.
        let mut c = Config::default();
        c.io_backend = IoBackend::Mmap;
        c.io_buffer_limit = Some(1);
        c.validate().unwrap();
    }
}
