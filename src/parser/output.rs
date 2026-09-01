/// Trait abstracting over field output types — the form in which a
/// parsed field is handed to the user's callback.
///
/// Implemented for `str` (UTF-8 validated) and `[u8]` (raw bytes).
///
/// # Safety
///
/// The steppers build each record as `(*mut u8, usize)` pairs in their
/// scratch and reinterpret it as `&mut [&mut O]` before calling the
/// accumulator. Implementors must uphold all of:
///
/// 1. `&mut O` is a fat pointer laid out exactly as `(*mut u8, usize)`,
///    with the second word being a **byte** length. True for `str` and
///    `[u8]`.
/// 2. `validate_chunk(bytes)` returning `Ok(n)` means every prefix
///    `&bytes[..k]` for `k <= n` that the parser hands out as an `O`
///    is a valid `O`.
/// 3. `REQUIRES_ASCII_CONTROL_BYTES` is `true` unless *every* sub-range
///    of a validated prefix is a valid `O`.
pub(crate) unsafe trait Output {
    /// Whether the parser must reject non-ASCII control bytes.
    ///
    /// `true` for `str`: a delimiter set to a UTF-8 continuation byte
    /// would cut a multi-byte sequence in half. `false` for `[u8]`.
    const REQUIRES_ASCII_CONTROL_BYTES: bool;

    /// Validate a raw chunk of bytes.
    fn validate_chunk(bytes: &[u8]) -> Result<usize, usize>;
}

unsafe impl Output for str {
    const REQUIRES_ASCII_CONTROL_BYTES: bool = true;

    /// Uses `simdutf8` when that feature is on; both report the same
    /// `valid_up_to` / `error_len` pair.
    fn validate_chunk(bytes: &[u8]) -> Result<usize, usize> {
        #[cfg(feature = "simdutf8")]
        let result = simdutf8::compat::from_utf8(bytes);
        #[cfg(not(feature = "simdutf8"))]
        let result = core::str::from_utf8(bytes);

        match result {
            Ok(_) => Ok(bytes.len()),
            // ended mid-sequence: fine during incremental parsing
            Err(e) if e.error_len().is_none() => Ok(e.valid_up_to()),
            Err(e) => Err(e.valid_up_to()),
        }
    }
}

unsafe impl Output for [u8] {
    const REQUIRES_ASCII_CONTROL_BYTES: bool = false;

    /// Raw bytes need no validation; every byte sequence is valid.
    fn validate_chunk(bytes: &[u8]) -> Result<usize, usize> {
        Ok(bytes.len())
    }
}

/// Hand a record's `(*mut u8, usize)` field pairs to the accumulator as
/// `&mut [&mut O]`.
///
/// # Safety
///
/// Every pair must delimit an initialized range of the reader's buffer
/// that is a valid `O` (for `str`, inside the driver's validated
/// prefix).
#[inline]
pub(crate) unsafe fn emit_record<O, E>(
    fields: &mut [(*mut u8, usize)],
    emit: &mut E,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    O: Output + ?Sized,
    E: FnMut(&mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
{
    // SAFETY: `&mut O` is laid out as `(*mut u8, usize)` with a byte
    // length (`Output` clause 1); the caller covers the rest.
    let refs: &mut [&mut O] =
        unsafe { std::slice::from_raw_parts_mut(fields.as_mut_ptr() as *mut &mut O, fields.len()) };
    emit(refs)
}
