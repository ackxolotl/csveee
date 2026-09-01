use std::marker::PhantomData;

use crate::config::{Config, IoBackend, ParserBackend, QuoteHandling, RecordTerminator};
use crate::parser::ChunkParser;
use crate::parser::dfa::DfaChunkParser;
#[cfg(feature = "simd")]
use crate::parser::simd::SimdChunkParser;
use crate::{Arity, Bytes, Parser, Text, Variadic};

/// A CSV parser builder.
pub struct ParserBuilder<Mode = Arity, Encoding: ?Sized = Text> {
    config: Config,
    _marker: PhantomData<(Mode, Encoding)>,
}

impl<Mode, Encoding: ?Sized> Clone for ParserBuilder<Mode, Encoding> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Mode, Encoding: ?Sized> std::fmt::Debug for ParserBuilder<Mode, Encoding> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The phantom marker carries no state worth printing.
        f.debug_struct("ParserBuilder")
            .field("config", &self.config)
            .finish()
    }
}

impl Default for ParserBuilder<Arity, Text> {
    fn default() -> Self {
        Self {
            config: Config::default(),
            _marker: PhantomData,
        }
    }
}

impl ParserBuilder<Arity, Text> {
    /// Start from the default configuration.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<Encoding: ?Sized> ParserBuilder<Arity, Encoding> {
    /// Switch to variadic mode: fields are handed to the callback as a slice `&mut [T]`
    /// instead of a fixed-size array `&mut [T; N]`.
    ///
    /// Records are expected to have a uniform field count across the file.
    /// If field counts may vary between records, use [`ParserBuilder::flexible`] instead.
    pub fn variadic(self) -> ParserBuilder<Variadic, Encoding> {
        ParserBuilder {
            config: self.config,
            _marker: PhantomData,
        }
    }

    /// Switch to flexible mode: fields are handed to the callback as a slice `&mut [T]`,
    /// and records are allowed to have different numbers of fields.
    ///
    /// Equivalent to `.variadic()` with field-count variation explicitly permitted.
    ///
    /// With the field count unconstrained, the accumulator is the only oracle a
    /// speculative parse is checked against. An accumulation function that accepts
    /// any record may lead to expensive sequential reparsing at merge time.
    pub fn flexible(self) -> ParserBuilder<Variadic, Encoding> {
        let mut config = self.config;
        config.flexible = true;
        ParserBuilder {
            config,
            _marker: PhantomData,
        }
    }
}

impl<Mode> ParserBuilder<Mode, Text> {
    /// Use raw byte fields instead of UTF-8 validated strings.
    ///
    /// Fields are handed out as `&mut [u8]` instead of `&mut str`,
    /// skipping UTF-8 validation. Use this for non-UTF-8 encodings
    /// like Latin-1 or Windows-1252.
    pub fn bytes(self) -> ParserBuilder<Mode, Bytes> {
        ParserBuilder {
            config: self.config,
            _marker: PhantomData,
        }
    }
}

impl<Mode> ParserBuilder<Mode, Text> {
    /// Validate the config and build a parser.
    ///
    /// Returns an error if the configuration is invalid, or if a control
    /// byte is non-ASCII — that could split a multi-byte UTF-8 sequence
    /// across two fields. Use [`Self::bytes`] to parse such dialects.
    pub fn try_build(self) -> crate::Result<Parser<Mode, Text>> {
        crate::parser::try_from_config(self.config)
    }

    /// Validate the config and build a parser, panicking on error.
    ///
    /// Use [`Self::try_build`] if the config might come from untrusted input.
    pub fn build(self) -> Parser<Mode, Text> {
        self.try_build().expect("invalid parser configuration")
    }
}

impl<Mode> ParserBuilder<Mode, Bytes> {
    /// Validate the config and build a parser.
    ///
    /// Returns an error if the configuration is invalid. Unlike text mode,
    /// non-ASCII control bytes are allowed.
    pub fn try_build(self) -> crate::Result<Parser<Mode, Bytes>> {
        crate::parser::try_from_config(self.config)
    }

    /// Validate the config and build a parser, panicking on error.
    ///
    /// Use [`Self::try_build`] if the config might come from untrusted input.
    pub fn build(self) -> Parser<Mode, Bytes> {
        self.try_build().expect("invalid parser configuration")
    }
}

/// Dialect: how the bytes of the file spell out fields and records.
impl<Mode, Encoding: ?Sized> ParserBuilder<Mode, Encoding> {
    /// Byte that separates fields within a record (default: `b','`).
    ///
    /// Must differ from every other control byte. In text mode it must
    /// also be ASCII — use [`Self::bytes`] for a non-UTF-8 dialect.
    pub fn delimiter(mut self, delimiter: u8) -> Self {
        self.config.delimiter = delimiter;
        self
    }

    /// Byte sequence that ends a record (default: `RecordTerminator::CRLF`).
    ///
    /// `CRLF` also accepts bare `\r` and `\n`; use
    /// [`RecordTerminator::Byte`] (or the `LF`/`CR` constants) to pin a
    /// single terminator byte.
    pub fn terminator(mut self, terminator: RecordTerminator) -> Self {
        self.config.terminator = terminator;
        self
    }

    /// Byte that quotes a field (default: `Some(b'"')`).
    ///
    /// `None` disables quoting entirely, so delimiters and terminators
    /// are always structural. Setting this to `None` while an escape byte
    /// is configured fails at `try_build`.
    pub fn quote(mut self, quote: Option<u8>) -> Self {
        self.config.quote = quote;
        self
    }

    /// Byte that escapes the next character inside a quoted field
    /// (default: `Some(b'"')`, i.e. RFC 4180 doubled quotes).
    ///
    /// Escape bytes are removed from the field in place. Requires
    /// [`Self::quote`] to be set; `None` means no escaping.
    pub fn escape(mut self, escape: Option<u8>) -> Self {
        self.config.escape = escape;
        self
    }

    /// Set how quote characters are handled (default: `QuoteHandling::Toggle`).
    ///
    /// See [`QuoteHandling`] for the available modes.
    ///
    /// **SIMD compatibility:** the SIMD parser backend supports only
    /// `QuoteHandling::Toggle`. Pairing `Literal` or `Strict` with an
    /// explicit `parser_backend(ParserBackend::Simd)` will fail at
    /// `try_build` with `Error::InvalidConfig`.
    pub fn quote_handling(mut self, handling: QuoteHandling) -> Self {
        self.config.quote_handling = handling;
        self
    }

    /// Byte that marks a comment line (default: `None`).
    ///
    /// When set, a line starting with this byte is ignored.
    pub fn comment(mut self, comment: Option<u8>) -> Self {
        self.config.comment = comment;
        self
    }
}

/// Interpretation: what the parser does with the records it finds.
///
/// Blank lines are skipped rather than emitted: a run of record
/// terminators yields no records, as in Go's `encoding/csv` and the
/// `csv` crate.
impl<Mode, Encoding: ?Sized> ParserBuilder<Mode, Encoding> {
    /// Whether the first record is a header row (default: `true`).
    ///
    /// Headers are skipped just like comment lines if enabled.
    pub fn has_headers(mut self, has_headers: bool) -> Self {
        self.config.has_headers = has_headers;
        self
    }

    /// Strip leading/trailing ASCII whitespace (`\t\n\v\f\r `) from each
    /// field before it is handed to the accumulator (default: `false`).
    ///
    /// **SIMD compatibility:** the SIMD parser backend does not support
    /// trimming. Pairing `trim(true)` with an explicit
    /// `parser_backend(ParserBackend::Simd)` will fail at `try_build`
    /// with `Error::InvalidConfig`.
    pub fn trim(mut self, trim: bool) -> Self {
        self.config.trim = trim;
        self
    }
}

/// Backends: which chunk parser and I/O strategy run the parse.
impl<Mode, Encoding: ?Sized> ParserBuilder<Mode, Encoding> {
    /// Set the chunk parser backend (default: `ParserBackend::Auto`).
    ///
    /// `Auto` selects the fastest parser that supports the current config.
    /// `Dfa` forces the DFA parser (works with any config). `Simd` forces
    /// the vectorized parser (errors if the config is unsupported).
    ///
    /// **SIMD restrictions:** `Simd` rejects `trim(true)`,
    /// `QuoteHandling::{Literal, Strict}`, comments, variadic/flexible
    /// mode, an escape byte distinct from the quote, and records wider
    /// than 63 fields. The reason is reported via `Error::InvalidConfig`
    /// at `try_build`. Use `Auto` to fall back to the DFA parser
    /// silently when the config rules SIMD out.
    pub fn parser_backend(mut self, backend: ParserBackend) -> Self {
        self.config.parser_backend = backend;
        self
    }

    /// Set the I/O backend used to read chunks (default: `IoBackend::Auto`).
    ///
    /// `Auto` picks `InMemory` for files up to 1 MiB and `RingBuf` above
    /// that. See [`IoBackend`] for the trade-offs of each backend.
    pub fn io_backend(mut self, backend: IoBackend) -> Self {
        self.config.io_backend = backend;
        self
    }

    /// Check whether a specific chunk-parser backend can handle the
    /// current configuration.
    pub fn supports_backend(&self, backend: ParserBackend) -> Result<(), &'static str> {
        match backend {
            ParserBackend::Auto => Ok(()),
            ParserBackend::Dfa => DfaChunkParser::supports(&self.config),
            ParserBackend::Simd => {
                #[cfg(feature = "simd")]
                {
                    // `field_count` is only known at `parse::<N>` time
                    let mut probe = self.config.clone();
                    probe.field_count.get_or_insert(1);
                    SimdChunkParser::supports(&probe)
                }
                #[cfg(not(feature = "simd"))]
                {
                    Err("SIMD parser not compiled in (build with `--features simd` on nightly)")
                }
            }
        }
    }
}

/// Resources: how much CPU and memory the parse may use.
impl<Mode, Encoding: ?Sized> ParserBuilder<Mode, Encoding> {
    /// Upper bound on worker threads (default: auto).
    ///
    /// `0` means auto-detect from available parallelism at parse time.
    pub fn concurrency(mut self, concurrency: usize) -> Self {
        self.config.concurrency = concurrency;
        self
    }

    /// Size of work-stealing units handed to worker threads (default: auto).
    ///
    /// `0` means auto-detect from the parser backend at parse time
    /// (currently 256 KiB for the DFA parser, 1 MiB for SIMD).
    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.config.chunk_size = chunk_size;
        self
    }

    /// Total I/O buffer memory the parse may hold across all worker
    /// threads (default: auto).
    ///
    /// - `Some(0)` resolves at parse time to 50% of physical RAM.
    /// - `Some(n)` with `n > 0` caps the parse at `n` bytes in total.
    /// - `None` disables the cap entirely. Use only with trusted input.
    ///
    /// Bounds the ring buffer only. Mmap is governed by the OS page
    /// cache, and the in-memory backend's footprint is the file size
    /// rather than a buffer that grows.
    pub fn io_buffer_limit(mut self, limit: Option<usize>) -> Self {
        self.config.io_buffer_limit = limit;
        self
    }
}
