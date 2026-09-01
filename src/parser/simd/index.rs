//! Index extraction and record assembly: turns per-vector `B`/`E`/`R`
//! bitmasks into buffer offsets and compacts fields around `R` in place.
//! Extraction is a scalar `tzcnt` loop — CSV bitmasks are sparse.

use crate::parser::output::Output;

/// Largest arity the SIMD path serves; wider records route to the DFA.
pub(super) const FIXED_MAX_ARITY: usize = 63;

/// Emit `num_records` records of `n` fields by striding `b_offs`/`e_offs`
/// into the caller's persistent `scratch`, applying R-removal inline.
/// Validation is the verifier's job.
#[allow(clippy::too_many_arguments)]
pub(super) fn assemble_records_fixed<O, E>(
    buf: &mut [u8],
    n: usize,
    b_offs: &[usize],
    e_offs: &[usize],
    r_offs: &[usize],
    num_records: usize,
    scratch: &mut [(*mut u8, usize); FIXED_MAX_ARITY],
    emit: &mut E,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    O: Output + ?Sized,
    E: FnMut(&mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
{
    // Once per call, so it costs nothing against the per-record work, and
    // it is what makes the `get_unchecked`es below sound.
    let total = num_records * n;
    assert!((1..=FIXED_MAX_ARITY).contains(&n) && b_offs.len() >= total && e_offs.len() >= total);

    // Read before the derivation below: reborrowing `buf` afterwards is
    // exactly what invalidates it.
    let buf_len = buf.len();
    // One derivation for the whole call: the removal path compacts and
    // hands out pointers, so they must share a provenance root.
    // Compacting through `buf` would invalidate what `scratch` holds.
    let buf_ptr = buf.as_mut_ptr();

    let mut r_cursor = 0usize;
    let mut field_idx = 0usize;

    for _ in 0..num_records {
        // A record with no removals — the common case even in files with
        // escapes — costs two loads, a subtract and a store per field.
        // SAFETY: `field_idx + n - 1 < total <= b/e_offs.len()`.
        let rec_first_b = unsafe { *b_offs.get_unchecked(field_idx) };
        let rec_last_e = unsafe { *e_offs.get_unchecked(field_idx + n - 1) };
        while r_cursor < r_offs.len() && r_offs[r_cursor] < rec_first_b {
            r_cursor += 1;
        }
        let record_clean = r_cursor >= r_offs.len() || r_offs[r_cursor] >= rec_last_e;

        if record_clean {
            for i in 0..n {
                // SAFETY: `field_idx < total <= b/e_offs.len()`,
                // `i < n <= scratch.len()`, and `b`/`e` index `buf`.
                unsafe {
                    let b = *b_offs.get_unchecked(field_idx);
                    let e = *e_offs.get_unchecked(field_idx);
                    debug_assert!(e >= b, "field end precedes its begin");
                    debug_assert!(e <= buf_len, "field end past the buffer");
                    *scratch.get_unchecked_mut(i) = (buf_ptr.add(b), e - b);
                }
                field_idx += 1;
            }
        } else {
            for slot in scratch.iter_mut().take(n) {
                let b = b_offs[field_idx];
                let e = e_offs[field_idx];
                debug_assert!(e <= buf_len, "field end past the buffer");
                while r_cursor < r_offs.len() && r_offs[r_cursor] < b {
                    r_cursor += 1;
                }
                let r_start = r_cursor;
                while r_cursor < r_offs.len() && r_offs[r_cursor] < e {
                    r_cursor += 1;
                }
                debug_assert!(
                    e >= b + (r_cursor - r_start),
                    "more removals than field bytes"
                );
                let final_len = e - b - (r_cursor - r_start);
                if r_cursor != r_start {
                    let mut write = b;
                    let mut prev_end = b;
                    for &r in &r_offs[r_start..r_cursor] {
                        let len = r - prev_end;
                        if len != 0 && write != prev_end {
                            // SAFETY: a left shift (`write < prev_end`) with
                            // `prev_end + len == r <= e`, all inside `buf`.
                            unsafe {
                                std::ptr::copy(buf_ptr.add(prev_end), buf_ptr.add(write), len)
                            };
                        }
                        write += len;
                        prev_end = r + 1;
                    }
                    if e != prev_end && write != prev_end {
                        // SAFETY: as above, for the trailing run.
                        unsafe {
                            std::ptr::copy(buf_ptr.add(prev_end), buf_ptr.add(write), e - prev_end)
                        };
                    }
                }
                // SAFETY: `b` is an offset into `buf` by construction.
                *slot = (unsafe { buf_ptr.add(b) }, final_len);
                field_idx += 1;
            }
        }
        // SAFETY: slots `0..n` hold ascending, non-overlapping ranges
        // derived from `buf_ptr`, which the removal path compacts through.
        unsafe { crate::parser::output::emit_record::<O, _>(&mut scratch[..n], emit) }?;
    }
    Ok(())
}

/// Append the offset of every set bit in `mask` to `out`, shifted by
/// `base` and sorted ascending. Runs three times per vector, so it
/// reserves up front and writes unchecked instead of `push`ing.
#[inline]
pub(super) fn extend_offsets(out: &mut Vec<usize>, mask: u64, base: usize) {
    let count = mask.count_ones() as usize;
    out.reserve(count);
    let mut len = out.len();
    // SAFETY: `reserve(count)` covers the exactly `count` elements the
    // loop writes, and `set_len` publishes only initialized slots.
    unsafe {
        let ptr = out.as_mut_ptr();
        let mut m = mask;
        while m != 0 {
            // `wrapping_add`: the prologue's base is a wrapped `-misalign`,
            // but its low lanes are pad, so every offset is non-negative.
            *ptr.add(len) = base.wrapping_add(m.trailing_zeros() as usize);
            len += 1;
            m &= m - 1;
        }
        out.set_len(len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── extend_offsets ───────────────────────────────────────────

    #[test]
    fn extend_offsets_empty_mask_pushes_nothing() {
        let mut out = vec![];
        extend_offsets(&mut out, 0, 100);
        assert!(out.is_empty());
    }

    #[test]
    fn extend_offsets_extracts_in_ascending_order() {
        let mut out = vec![];
        // Bits at 0, 5, 17, 63.
        let mask = 1u64 | (1 << 5) | (1 << 17) | (1 << 63);
        extend_offsets(&mut out, mask, 0);
        assert_eq!(out, vec![0, 5, 17, 63]);
    }

    #[test]
    fn extend_offsets_applies_base() {
        let mut out = vec![];
        extend_offsets(&mut out, 0b1011, 1000);
        assert_eq!(out, vec![1000, 1001, 1003]);
    }

    #[test]
    fn extend_offsets_appends() {
        let mut out = vec![42];
        extend_offsets(&mut out, 0b10, 0);
        assert_eq!(out, vec![42, 1]);
    }

    #[test]
    fn extend_offsets_full_mask() {
        let mut out = vec![];
        extend_offsets(&mut out, !0u64, 0);
        assert_eq!(out.len(), 64);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, i);
        }
    }

    // ── helpers ──────────────────────────────────────────────────

    /// Drive the fixed-arity path (`n` fields × `num_records`).
    fn collect_fixed(
        buf: &mut [u8],
        n: usize,
        b_offs: &[usize],
        e_offs: &[usize],
        r_offs: &[usize],
        num_records: usize,
    ) -> Vec<Vec<Vec<u8>>> {
        let mut out: Vec<Vec<Vec<u8>>> = Vec::new();
        let mut emit =
            |fields: &mut [&mut [u8]]| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                out.push(fields.iter().map(|f| f.to_vec()).collect());
                Ok(())
            };
        let mut scratch = [(std::ptr::null_mut(), 0); FIXED_MAX_ARITY];
        assemble_records_fixed::<[u8], _>(
            buf,
            n,
            b_offs,
            e_offs,
            r_offs,
            num_records,
            &mut scratch,
            &mut emit,
        )
        .unwrap();
        out
    }

    // ── assemble_records_fixed (fast path) ───────────────────────

    #[test]
    fn fixed_strides_records_by_n() {
        // "aa,bb\ncc,dd\n", N=2, 2 records.
        let mut buf = b"aa,bb\ncc,dd\n".to_vec();
        let recs = collect_fixed(&mut buf, 2, &[0, 3, 6, 9], &[2, 5, 8, 11], &[], 2);
        assert_eq!(
            recs,
            vec![
                vec![b"aa".to_vec(), b"bb".to_vec()],
                vec![b"cc".to_vec(), b"dd".to_vec()],
            ]
        );
    }

    #[test]
    fn fixed_applies_inline_removals() {
        // `"a""b",c\n`: field 0 spans [1, 5) with R={2} dropping one inner
        // quote → `a"b`; field 1 is `c` at [7, 8).
        let mut buf = b"\"a\"\"b\",c\n".to_vec();
        let recs = collect_fixed(&mut buf, 2, &[1, 7], &[5, 8], &[2], 1);
        assert_eq!(recs.len(), 1);
        assert_eq!(&recs[0][0][..], b"a\"b");
        assert_eq!(&recs[0][1][..], b"c");
    }
}
