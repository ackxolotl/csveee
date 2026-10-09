//! The byte buffer behind the ring-buffer backend.
//!
//! Two implementations share one interface, selected per target:
//!
//! - `MirroredBuf` maps its memory twice back to back, so wrapped data is
//!   contiguous without copying. Needs OS support for double mapping
//!   (Linux, macOS, Windows).
//! - `LinearBuf` is a plain heap buffer that moves the leftover bytes to
//!   the front before each write. Used everywhere else, e.g. on WebAssembly.
//!
//! Both keep `data()` contiguous and `unused_mut().len() == writable()`,
//! which is all the readers rely on. The test suite below runs against both.

use std::io;

use crate::sys::allocation_granularity;

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
mod file;
#[cfg(any(test, not(any(target_os = "linux", target_os = "macos", windows))))]
mod linear;
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
mod mirrored;

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
pub use self::linear::LinearBuf as RingBuf;
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
pub use self::mirrored::MirroredBuf as RingBuf;

/// Round a requested capacity up to a power of two of at least the
/// allocation granularity, and check it against `limit`.
fn checked_capacity(capacity: usize, limit: Option<usize>) -> io::Result<usize> {
    let granularity = allocation_granularity();
    let capacity = capacity.next_power_of_two().max(granularity);

    if let Some(limit) = limit
        && capacity > limit
    {
        return Err(io::Error::other(format!(
            "ring buffer capacity ({capacity} bytes, rounded up from the requested size to a \
             power of two and at least the {granularity}-byte allocation granularity) would \
             exceed limit ({limit} bytes)",
        )));
    }
    Ok(capacity)
}

/// The capacity after doubling, checked against `limit`.
fn grown_capacity(capacity: usize, limit: Option<usize>) -> io::Result<usize> {
    let new_capacity = capacity * 2;
    if let Some(limit) = limit
        && new_capacity > limit
    {
        return Err(io::Error::other(format!(
            "ring buffer growth ({new_capacity} bytes) would exceed limit ({limit} bytes)",
        )));
    }
    Ok(new_capacity)
}

/// The same suite for every buffer type, so they cannot drift apart.
#[cfg(test)]
macro_rules! buffer_tests {
    ($name:ident, $buf:ty) => {
        mod $name {
            use std::collections::VecDeque;

            use crate::sys::allocation_granularity;

            type Buf = $buf;

            #[test]
            fn basic() {
                let mut rb = Buf::new(allocation_granularity(), None).unwrap();
                let cap = rb.capacity();

                assert_eq!(rb.readable(), 0);
                assert_eq!(rb.writable(), cap);

                rb.unused_mut()[..5].copy_from_slice(b"hello");
                rb.produce(5);

                assert_eq!(rb.readable(), 5);
                assert_eq!(rb.data(), b"hello");

                rb.consume(5);
                assert_eq!(rb.readable(), 0);
                assert_eq!(rb.writable(), cap);
            }

            #[test]
            fn wrap_around() {
                let mut rb = Buf::new(allocation_granularity(), None).unwrap();
                let cap = rb.capacity();

                // Advance consumer/producer near the end.
                rb.unused_mut()[..cap - 2].copy_from_slice(&vec![0u8; cap - 2]);
                rb.produce(cap - 2);
                rb.consume(cap - 2);

                // Write across the wrap boundary.
                rb.unused_mut()[..5].copy_from_slice(b"abcde");
                rb.produce(5);

                // Data should appear contiguous despite wrapping.
                assert_eq!(rb.data(), b"abcde");
            }

            /// Leftover bytes near the end must stay intact, in order,
            /// when the next write needs the space before them.
            #[test]
            fn leftover_survives_a_write_past_the_end() {
                let mut rb = Buf::new(allocation_granularity(), None).unwrap();
                let cap = rb.capacity();

                rb.unused_mut().fill(b'x');
                rb.produce(cap);
                rb.consume(cap - 3);
                rb.data_mut().copy_from_slice(b"abc");

                let unused = rb.unused_mut();
                assert_eq!(unused.len(), cap - 3);
                unused.fill(b'y');
                rb.produce(cap - 3);

                assert_eq!(rb.readable(), cap);
                assert_eq!(&rb.data()[..3], b"abc");
                assert!(rb.data()[3..].iter().all(|&b| b == b'y'));
            }

            #[test]
            fn grow() {
                let mut rb = Buf::new(allocation_granularity(), None).unwrap();
                let old_cap = rb.capacity();

                rb.unused_mut()[..3].copy_from_slice(b"abc");
                rb.produce(3);

                rb.grow().unwrap();
                assert_eq!(rb.capacity(), old_cap * 2);
                assert_eq!(rb.readable(), 3);
                assert_eq!(&rb.data()[..3], b"abc");
            }

            #[test]
            fn grow_keeps_wrapped_data() {
                let mut rb = Buf::new(allocation_granularity(), None).unwrap();
                let cap = rb.capacity();

                rb.unused_mut().fill(0);
                rb.produce(cap);
                rb.consume(cap - 2);
                let unused = rb.unused_mut();
                unused[..2].copy_from_slice(b"cd");
                rb.produce(2);
                rb.data_mut()[..2].copy_from_slice(b"ab");

                rb.grow().unwrap();
                assert_eq!(rb.data(), b"abcd");
                assert_eq!(rb.writable(), cap * 2 - 4);
            }

            #[test]
            fn reset_empties() {
                let mut rb = Buf::new(allocation_granularity(), None).unwrap();
                let cap = rb.capacity();
                rb.unused_mut()[..7].copy_from_slice(b"discard");
                rb.produce(7);
                rb.consume(2);

                rb.reset();
                assert_eq!(rb.readable(), 0);
                assert_eq!(rb.unused_mut().len(), cap);
            }

            /// Random produce/consume/grow against a `VecDeque` model.
            #[test]
            fn matches_a_model() {
                let mut rb = Buf::new(allocation_granularity(), None).unwrap();
                let mut model = VecDeque::new();
                let mut state = 0x2545_f491_4f6c_dd1du64;
                let mut next = |bound: usize| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state % bound as u64) as usize
                };
                let mut byte = 0u8;

                for _ in 0..2000 {
                    match next(10) {
                        0 if rb.capacity() < 8 * allocation_granularity() => {
                            rb.grow().unwrap();
                        }
                        1..=5 => {
                            let writable = rb.writable();
                            let unused = rb.unused_mut();
                            assert_eq!(unused.len(), writable);
                            let n = next(unused.len() + 1);
                            for b in &mut unused[..n] {
                                *b = byte;
                                model.push_back(byte);
                                byte = byte.wrapping_add(1);
                            }
                            rb.produce(n);
                        }
                        _ => {
                            let n = next(rb.readable() + 1);
                            rb.consume(n);
                            model.drain(..n);
                        }
                    }
                    assert_eq!(rb.data(), model.make_contiguous());
                }
            }

            #[test]
            fn new_rejects_limit_below_granularity() {
                let Err(err) = Buf::new(50, Some(200)) else {
                    panic!("expected a ring buffer larger than its limit to be rejected");
                };
                assert!(
                    err.to_string().contains("would exceed limit"),
                    "unexpected error: {err}"
                );
            }

            #[test]
            fn new_accepts_limit_at_granularity() {
                let granularity = allocation_granularity();
                let rb = Buf::new(1, Some(granularity)).unwrap();
                assert_eq!(rb.capacity(), granularity);
            }

            #[test]
            fn grow_rejects_exceeding_the_limit() {
                let granularity = allocation_granularity();
                let mut rb = Buf::new(1, Some(granularity)).unwrap();
                let err = rb.grow().unwrap_err();
                assert!(
                    err.to_string().contains("would exceed limit"),
                    "unexpected error: {err}"
                );
            }
        }
    };
}

#[cfg(test)]
buffer_tests!(linear_tests, super::linear::LinearBuf);
#[cfg(all(test, any(target_os = "linux", target_os = "macos", windows)))]
buffer_tests!(mirrored_tests, super::mirrored::MirroredBuf);
