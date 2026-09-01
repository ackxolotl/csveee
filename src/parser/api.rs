use std::io::Read;
use std::path::Path;

use crate::config::Config;
use crate::parser::output::Output;
use crate::scheduler::Scheduler;

/// Marker type: fields are handed to the callback as a fixed-size array `[&mut T; N]` (default).
pub struct Arity;

/// Marker type: fields are handed to the callback as a slice `&mut [&mut T]`.
pub struct Variadic;

/// Fields are validated as UTF-8 and handed out as `&mut str` (default).
pub type Text = str;

/// Fields are raw bytes, no UTF-8 validation. For non-UTF-8 encodings.
pub type Bytes = [u8];

/// The CSV parser, parameterized by field count mode and encoding.
pub struct Parser<Mode = Arity, Encoding: ?Sized = Text> {
    config: Config,
    _marker: std::marker::PhantomData<(Mode, Encoding)>,
}

impl Default for Parser<Arity, Text> {
    fn default() -> Self {
        Self::new()
    }
}

/// Construct a parser from a `Config`, validating it first.
pub(crate) fn try_from_config<Mode, Encoding: Output + ?Sized>(
    config: Config,
) -> crate::Result<Parser<Mode, Encoding>> {
    config.validate()?;
    if Encoding::REQUIRES_ASCII_CONTROL_BYTES {
        config.validate_ascii_control_bytes()?;
    }
    Ok(Parser {
        config,
        _marker: std::marker::PhantomData,
    })
}

impl Parser<Arity, Text> {
    /// Construct a parser with default settings.
    ///
    /// The default config is statically known to be valid, so this
    /// is infallible. Use `ParserBuilder` for custom settings.
    pub fn new() -> Self {
        try_from_config(Config::default()).expect("default config must validate")
    }

    /// Parse a CSV file with a fixed number of fields per record.
    ///
    /// The file is split into chunks and parsed in parallel. `N` is
    /// inferred from the closure and validated at runtime.
    ///
    /// ```ignore
    /// parser.parse("data.csv", init, acc, merge)
    /// ```
    pub fn parse<P, S, I, A, M, R, const N: usize>(
        &mut self,
        path: P,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        P: AsRef<Path>,
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, [&mut str; N]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Sync,
        M: FnOnce(&mut [S]) -> R,
    {
        let acc_slice = move |state: &mut S, fields: &mut [&mut str]| {
            acc(state, fields_as_array::<_, N>(fields))
        };
        parse(
            self.arity_config::<N>(),
            path.as_ref(),
            init,
            acc_slice,
            merge,
        )
    }

    /// Parse CSV from any reader with a fixed number of fields per record.
    ///
    /// Sequential and single-threaded — for inputs without random access
    /// (stdin, pipes, sockets, decompressors). Prefer [`Self::parse`] for
    /// files on disk, which parses in parallel.
    ///
    /// Do not wrap `reader` in a `BufReader`: the parser reads into its own
    /// ring buffer in large blocks, so an intermediate buffer only adds a copy.
    ///
    /// ```ignore
    /// parser.parse_stream(std::io::stdin().lock(), init, acc, merge)
    /// ```
    pub fn parse_stream<Rd, S, I, A, M, R, const N: usize>(
        &mut self,
        reader: Rd,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        Rd: Read,
        I: FnOnce() -> S,
        A: Fn(&mut S, [&mut str; N]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
        M: FnOnce(&mut [S]) -> R,
    {
        let acc_slice = move |state: &mut S, fields: &mut [&mut str]| {
            acc(state, fields_as_array::<_, N>(fields))
        };
        parse_stream(self.arity_config::<N>(), reader, init, acc_slice, merge)
    }

    /// Parse CSV already in memory, with a fixed number of fields per record.
    ///
    /// Chunked and parsed in parallel like [`Self::parse`]. Accepts anything
    /// that derefs to bytes — `&[u8]`, `Vec<u8>`, `&str`, `String`:
    ///
    /// ```ignore
    /// parser.parse_slice("name,age\nada,36\n", init, acc, merge)
    /// ```
    ///
    /// The parser writes into the buffer it parses, so the bytes are copied
    /// into thread-private buffers rather than parsed in place; `data` itself
    /// is never modified.
    pub fn parse_slice<D, S, I, A, M, R, const N: usize>(
        &mut self,
        data: D,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        D: AsRef<[u8]>,
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, [&mut str; N]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Sync,
        M: FnOnce(&mut [S]) -> R,
    {
        let acc_slice = move |state: &mut S, fields: &mut [&mut str]| {
            acc(state, fields_as_array::<_, N>(fields))
        };
        parse_slice(
            self.arity_config::<N>(),
            data.as_ref(),
            init,
            acc_slice,
            merge,
        )
    }

    /// Clone the config, pinning the field count to the callback's arity.
    fn arity_config<const N: usize>(&self) -> Config {
        let mut config = self.config.clone();
        config.field_count = Some(N);
        config
    }
}

impl Parser<Variadic, Text> {
    /// Parse a CSV file with a variable number of fields per record.
    ///
    /// The file is split into chunks and parsed in parallel.
    ///
    /// ```ignore
    /// parser.parse("data.csv", init, acc, merge)
    /// ```
    pub fn parse<P, S, I, A, M, R>(
        &mut self,
        path: P,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        P: AsRef<Path>,
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, &mut [&mut str]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Sync,
        M: FnOnce(&mut [S]) -> R,
    {
        parse(self.config.clone(), path.as_ref(), init, acc, merge)
    }

    /// Parse CSV from any reader with a variable number of fields per record.
    ///
    /// Sequential and single-threaded — see [`Parser::parse_stream`] for when
    /// to reach for this over [`Self::parse`].
    pub fn parse_stream<Rd, S, I, A, M, R>(
        &mut self,
        reader: Rd,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        Rd: Read,
        I: FnOnce() -> S,
        A: Fn(&mut S, &mut [&mut str]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
        M: FnOnce(&mut [S]) -> R,
    {
        parse_stream(self.config.clone(), reader, init, acc, merge)
    }

    /// Parse CSV already in memory, with a variable number of fields per
    /// record. Chunked and parsed in parallel like [`Self::parse`]; see
    /// [`Parser::parse_slice`] for what `data` may be.
    pub fn parse_slice<D, S, I, A, M, R>(
        &mut self,
        data: D,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        D: AsRef<[u8]>,
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, &mut [&mut str]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Sync,
        M: FnOnce(&mut [S]) -> R,
    {
        parse_slice(self.config.clone(), data.as_ref(), init, acc, merge)
    }
}

impl Parser<Arity, Bytes> {
    /// Parse a CSV file with a fixed number of raw byte fields per record.
    ///
    /// The file is split into chunks and parsed in parallel.
    pub fn parse<P, S, I, A, M, R, const N: usize>(
        &mut self,
        path: P,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        P: AsRef<Path>,
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, [&mut [u8]; N]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Sync,
        M: FnOnce(&mut [S]) -> R,
    {
        let acc_slice = move |state: &mut S, fields: &mut [&mut [u8]]| {
            acc(state, fields_as_array::<_, N>(fields))
        };
        parse(
            self.arity_config::<N>(),
            path.as_ref(),
            init,
            acc_slice,
            merge,
        )
    }

    /// Parse raw byte fields from any reader, fixed field count per record.
    ///
    /// Sequential and single-threaded — see [`Parser::parse_stream`] for when
    /// to reach for this over [`Self::parse`].
    pub fn parse_stream<Rd, S, I, A, M, R, const N: usize>(
        &mut self,
        reader: Rd,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        Rd: Read,
        I: FnOnce() -> S,
        A: Fn(&mut S, [&mut [u8]; N]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
        M: FnOnce(&mut [S]) -> R,
    {
        let acc_slice = move |state: &mut S, fields: &mut [&mut [u8]]| {
            acc(state, fields_as_array::<_, N>(fields))
        };
        parse_stream(self.arity_config::<N>(), reader, init, acc_slice, merge)
    }

    /// Parse in-memory CSV as raw byte fields, fixed field count per record.
    ///
    /// Chunked and parsed in parallel like [`Self::parse`]; see
    /// [`Parser::parse_slice`] for what `data` may be.
    pub fn parse_slice<D, S, I, A, M, R, const N: usize>(
        &mut self,
        data: D,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        D: AsRef<[u8]>,
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, [&mut [u8]; N]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Sync,
        M: FnOnce(&mut [S]) -> R,
    {
        let acc_slice = move |state: &mut S, fields: &mut [&mut [u8]]| {
            acc(state, fields_as_array::<_, N>(fields))
        };
        parse_slice(
            self.arity_config::<N>(),
            data.as_ref(),
            init,
            acc_slice,
            merge,
        )
    }

    /// Clone the config, pinning the field count to the callback's arity.
    fn arity_config<const N: usize>(&self) -> Config {
        let mut config = self.config.clone();
        config.field_count = Some(N);
        config
    }
}

impl Parser<Variadic, Bytes> {
    /// Parse a CSV file with a variable number of raw byte fields per record.
    ///
    /// The file is split into chunks and parsed in parallel.
    pub fn parse<P, S, I, A, M, R>(
        &mut self,
        path: P,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        P: AsRef<Path>,
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, &mut [&mut [u8]]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Sync,
        M: FnOnce(&mut [S]) -> R,
    {
        parse(self.config.clone(), path.as_ref(), init, acc, merge)
    }

    /// Parse raw byte fields from any reader, variable field count per record.
    ///
    /// Sequential and single-threaded — see [`Parser::parse_stream`] for when
    /// to reach for this over [`Self::parse`].
    pub fn parse_stream<Rd, S, I, A, M, R>(
        &mut self,
        reader: Rd,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        Rd: Read,
        I: FnOnce() -> S,
        A: Fn(&mut S, &mut [&mut [u8]]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
        M: FnOnce(&mut [S]) -> R,
    {
        parse_stream(self.config.clone(), reader, init, acc, merge)
    }

    /// Parse in-memory CSV as raw byte fields, variable field count per
    /// record. Chunked and parsed in parallel like [`Self::parse`]; see
    /// [`Parser::parse_slice`] for what `data` may be.
    pub fn parse_slice<D, S, I, A, M, R>(
        &mut self,
        data: D,
        init: I,
        acc: A,
        merge: M,
    ) -> crate::Result<R>
    where
        D: AsRef<[u8]>,
        S: Send,
        I: Fn() -> S + Sync,
        A: Fn(&mut S, &mut [&mut [u8]]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Sync,
        M: FnOnce(&mut [S]) -> R,
    {
        parse_slice(self.config.clone(), data.as_ref(), init, acc, merge)
    }
}

/// Run the parallel (chunked, file-backed) pipeline against `path`.
#[cfg_attr(feature = "trace", tracing::instrument(skip(init, acc, merge)))]
fn parse<S, I, A, M, R, O>(
    config: Config,
    path: &Path,
    init: I,
    acc: A,
    merge: M,
) -> crate::Result<R>
where
    S: Send,
    I: Fn() -> S + Sync,
    A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Sync,
    M: FnOnce(&mut [S]) -> R,
    O: Output + ?Sized,
{
    Scheduler::new(config).run(path, init, acc, merge)
}

/// Run the parallel (chunked, in-memory) pipeline against `data`.
#[cfg_attr(feature = "trace", tracing::instrument(skip(init, acc, merge)))]
fn parse_slice<S, I, A, M, R, O>(
    config: Config,
    data: &[u8],
    init: I,
    acc: A,
    merge: M,
) -> crate::Result<R>
where
    S: Send,
    I: Fn() -> S + Sync,
    A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Sync,
    M: FnOnce(&mut [S]) -> R,
    O: Output + ?Sized,
{
    Scheduler::new(config).run_slice(data, init, acc, merge)
}

/// Run the sequential single-threaded pipeline against `reader`.
///
/// Unlike [`parse`] this never crosses a thread boundary, so the
/// callbacks need neither `Send` nor `Sync`.
#[cfg_attr(feature = "trace", tracing::instrument(skip(reader, init, acc, merge)))]
fn parse_stream<Rd, S, I, A, M, R, O>(
    config: Config,
    reader: Rd,
    init: I,
    acc: A,
    merge: M,
) -> crate::Result<R>
where
    Rd: Read,
    I: FnOnce() -> S,
    A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    M: FnOnce(&mut [S]) -> R,
    O: Output + ?Sized,
{
    Scheduler::new(config).run_stream(reader, init, acc, merge)
}

/// Reborrow a slice of fields as a fixed-size array of reborrowed fields.
fn fields_as_array<'a, T: ?Sized, const N: usize>(fields: &'a mut [&mut T]) -> [&'a mut T; N] {
    assert_eq!(fields.len(), N, "record arity not verified by the parser");
    let mut fields = fields.iter_mut();
    std::array::from_fn(|_| &mut **fields.next().unwrap())
}
