mod api;
pub mod chunk;
pub mod dfa;
mod driver;
mod output;
#[cfg(feature = "simd")]
pub mod simd;

pub(crate) use self::api::try_from_config;
pub use self::api::{Arity, Bytes, Parser, Text, Variadic};
pub use self::chunk::{Assumption, ChunkParser, FindRecordStart, assumptions_for_config};
pub(crate) use self::output::Output;
