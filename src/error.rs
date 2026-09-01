use std::fmt::{Display, Formatter};

/// The location of a parser error in the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    /// Absolute byte offset into the input where the error was detected.
    pub byte_offset: usize,
}

impl Position {
    /// Shift the position by `delta` bytes. Used by the scheduler to
    /// translate chunk-local offsets returned by chunk parsers into
    /// absolute file offsets.
    fn shift(self, delta: usize) -> Self {
        Self {
            byte_offset: self.byte_offset + delta,
        }
    }
}

/// A CSV parser error.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An I/O error.
    IO(std::io::Error),
    /// The parser configuration is invalid.
    InvalidConfig(&'static str),
    /// Wrong number of fields.
    NumberOfFields {
        /// Field count the callback's arity declares.
        expected: usize,
        /// Field count seen so far. A lower bound if the
        /// record runs past the end of the parser's current buffer.
        found: usize,
        /// Start of the record whose field count was wrong.
        position: Position,
    },
    /// A UTF-8 error.
    Utf8 {
        /// Offset of the invalid byte.
        position: Position,
    },
    /// Quote not at a field boundary.
    InvalidQuote {
        /// Offset of the offending quote byte.
        position: Position,
    },
    /// Unterminated quoted field.
    UnclosedQuote {
        /// Start of the field whose opening quote was never closed.
        position: Position,
    },
    /// Error returned by the user's accumulator function.
    User(Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    /// Wrap an error returned by the accumulator, unnesting a crate
    /// [`Error`] a callback handed back rather than boxing it in
    /// [`Error::User`].
    pub(crate) fn from_user(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        match err.downcast::<Error>() {
            Ok(inner) => *inner,
            Err(other) => Error::User(other),
        }
    }

    /// The position in the input where the error occurred, if available.
    pub fn position(&self) -> Option<&Position> {
        match self {
            Error::Utf8 { position }
            | Error::InvalidQuote { position }
            | Error::UnclosedQuote { position }
            | Error::NumberOfFields { position, .. } => Some(position),
            _ => None,
        }
    }

    /// Translate a chunk-local error position into an absolute file
    /// position by adding `base` to its `byte_offset`.
    pub(crate) fn with_base(self, base: usize) -> Self {
        match self {
            Error::Utf8 { position } => Error::Utf8 {
                position: position.shift(base),
            },
            Error::InvalidQuote { position } => Error::InvalidQuote {
                position: position.shift(base),
            },
            Error::UnclosedQuote { position } => Error::UnclosedQuote {
                position: position.shift(base),
            },
            Error::NumberOfFields {
                expected,
                found,
                position,
            } => Error::NumberOfFields {
                expected,
                found,
                position: position.shift(base),
            },
            other => other,
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::IO(err) => err.fmt(f),
            Error::InvalidConfig(msg) => write!(f, "invalid parser configuration: {msg}"),
            Error::NumberOfFields {
                expected,
                found,
                position,
            } => {
                write!(
                    f,
                    "expected {expected} fields, found {found} in the record at byte {}",
                    position.byte_offset
                )
            }
            Error::Utf8 { position } => {
                write!(f, "invalid UTF-8 at byte {}", position.byte_offset)
            }
            Error::InvalidQuote { position } => {
                write!(
                    f,
                    "quote not at field boundary at byte {}",
                    position.byte_offset
                )
            }
            Error::UnclosedQuote { position } => {
                write!(
                    f,
                    "unterminated quoted field at byte {}",
                    position.byte_offset
                )
            }
            Error::User(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IO(err)
    }
}
