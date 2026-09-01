//! A parallel CSV parser.
//!
//! An input is split into chunks that are parsed concurrently. Each
//! worker accumulates into its own state and a merge function folds the
//! per-worker states into the result. Fields are handed to the callback
//! as mutable slices into the parser's own buffer, so no per-record
//! allocation happens.
//!
//! ```no_run
//! use csveee::Parser;
//!
//! let mut parser = Parser::new();
//! let cities = parser.parse(
//!     "data.csv",
//!     Vec::new,
//!     |acc, [_name, _age, city]| {
//!         acc.push(city.to_string());
//!         Ok(())
//!     },
//!     |states| states.concat(),
//! )?;
//! # Ok::<(), csveee::Error>(())
//! ```
//!
//! [`Parser::parse_slice`] does the same for bytes already in memory,
//! and [`Parser::parse_stream`] is the sequential fallback for sources
//! without random access. [`ParserBuilder`] configures the dialect
//! (delimiter, quoting, comments), the output mode ([`Variadic`] for
//! slices instead of fixed-size arrays, [`Bytes`] to skip UTF-8
//! validation) and the execution and I/O backends.

#![cfg_attr(test, allow(clippy::field_reassign_with_default))]
#![cfg_attr(feature = "simd", feature(portable_simd))]

mod builder;
mod config;
mod error;
mod io;
mod parser;
mod scheduler;
mod sys;
#[cfg(test)]
mod test_support;
mod trace;

pub use self::builder::ParserBuilder;
pub use self::config::{IoBackend, ParserBackend, QuoteHandling, RecordTerminator};
pub use self::error::{Error, Position};
pub use self::io::ringbuf::RingBufSettings;
pub use self::parser::{Arity, Bytes, Parser, Text, Variadic};

/// A `Result` with this crate's [`Error`] type.
pub type Result<T> = std::result::Result<T, Error>;
