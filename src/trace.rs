//! Internal instrumentation shim.
//!
//! See `examples/trace.rs` for a working subscriber setup.

#[cfg(feature = "trace")]
pub(crate) use tracing::Level;

#[cfg(feature = "trace")]
macro_rules! debug {
    ($($arg:tt)*) => { ::tracing::debug!($($arg)*) };
}

#[cfg(feature = "trace")]
macro_rules! enabled {
    ($($arg:tt)*) => { ::tracing::enabled!($($arg)*) };
}

#[cfg(not(feature = "trace"))]
macro_rules! debug {
    ($($arg:tt)*) => {
        ()
    };
}

#[cfg(not(feature = "trace"))]
#[allow(unused_macros)]
macro_rules! enabled {
    ($($arg:tt)*) => {
        false
    };
}

#[allow(unused_imports)]
pub(crate) use {debug, enabled};
