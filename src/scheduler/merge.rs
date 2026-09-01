#[cfg(feature = "trace")]
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::slot::WriteOnceSlot;
use crate::config::Config;

/// Result of parsing a chunk under one assumption.
pub(super) struct ResolvedPass<S> {
    /// Absolute file offset of the first record's start.
    pub record_start: usize,
    /// The accumulated state, or the parse error.
    pub result: crate::Result<S>,
}

/// All speculative results for one chunk.
pub(super) struct ChunkOutput<S> {
    /// Results per assumption, in priority order.
    pub passes: Vec<ResolvedPass<S>>,
    /// Absolute file offset after the last complete record.
    pub record_end: Option<usize>,
}

/// Sequential merge: align record boundaries across chunks,
/// select the correct assumption for each, call merge.
#[cfg_attr(not(feature = "trace"), allow(unused_variables))]
#[cfg_attr(feature = "trace", tracing::instrument(skip(results, merge, reparse)))]
pub(super) fn collect_and_merge<S, M, R, F>(
    config: &Config,
    results: Vec<WriteOnceSlot<ChunkOutput<S>>>,
    merge: M,
    file_path: Option<&Path>,
    reparse: F,
) -> crate::Result<R>
where
    M: FnOnce(&mut [S]) -> R,
    F: Fn(usize, usize) -> crate::Result<(S, Option<usize>)>,
{
    let mut states = Vec::with_capacity(results.len());
    let mut next_record: Option<usize> = None;
    let mut chunk_end = 0;

    for (idx, slot) in results.into_iter().enumerate() {
        let output = slot.into_inner().expect("chunk was not parsed");

        // Skip if the next record does not begin in this chunk.
        chunk_end += config.chunk_size;
        if let Some(next) = next_record
            && next > chunk_end
        {
            continue;
        }

        // If no passes were produced, the chunk contained no records
        // at all (e.g., the entire chunk was consumed by headers, or
        // the file is too small to contain any data). Skip it.
        if output.passes.is_empty() {
            continue;
        }

        // Find the pass whose record_start matches where we expect
        // the next record. For the first chunk (next_record == None),
        // accept any successful pass.
        #[cfg(feature = "trace")]
        let chunk_idx = idx;
        let selected_idx = output
            .passes
            .iter()
            .position(|p| next_record.is_none_or(|expected| expected == p.record_start));

        if let Some(pos) = selected_idx {
            let pass = output.passes.into_iter().nth(pos).unwrap();
            crate::trace::debug!(
                chunk_idx,
                record_start = pass.record_start,
                record_end = ?output.record_end,
                "merge matched",
            );
            // Propagate errors: if the selected assumption's parse
            // failed, that's the authoritative error for this chunk.
            states.push(pass.result?);
            next_record = output.record_end;
            continue;
        }

        // No assumption's record_start aligns. Speculation produced
        // a false positive — reparse the chunk from the known-correct
        // record boundary.
        let expected = next_record.expect("mismatch on first chunk (no previous record_end)");

        #[cfg(feature = "trace")]
        {
            let chunk_boundary = chunk_idx * config.chunk_size;
            let pass_starts: Vec<_> = output
                .passes
                .iter()
                .map(|p| (p.record_start, p.result.is_ok()))
                .collect();

            crate::trace::debug!(
                chunk_idx,
                expected,
                ?pass_starts,
                "speculation mismatch, reparsing chunk",
            );
            if let Some(path) = file_path
                && crate::trace::enabled!(crate::trace::Level::DEBUG)
            {
                dump_mismatch_context(path, expected, &pass_starts, chunk_boundary);
            }
        }

        let (state, record_end) = reparse(expected, chunk_end)?;
        let record_end = record_end.map(|o| expected + o);
        crate::trace::debug!(
            chunk_idx,
            record_start = expected,
            ?record_end,
            "reparse complete",
        );
        states.push(state);
        next_record = record_end;
    }

    Ok(merge(&mut states))
}

/// Dump CSV context around the mismatch for debugging.
#[cfg(feature = "trace")]
fn dump_mismatch_context(
    file_path: &Path,
    expected: usize,
    pass_starts: &[(usize, bool)],
    chunk_boundary: usize,
) {
    if let Some((escaped, marker)) = dump_offset_context(file_path, expected, Some(chunk_boundary))
    {
        crate::trace::debug!(
            offset = expected,
            chunk_boundary,
            "expected next_record context:\n{}\n{}",
            escaped,
            marker,
        );
    }
    for &(offered, ok) in pass_starts {
        if let Some((escaped, marker)) =
            dump_offset_context(file_path, offered, Some(chunk_boundary))
        {
            crate::trace::debug!(
                offset = offered,
                ok,
                chunk_boundary,
                "offered record_start context:\n{}\n{}",
                escaped,
                marker,
            );
        }
    }
}

/// Read ~80 bytes on either side of `offset` from `file_path` and return
/// an escape-debug rendering alongside a marker line that places a `^`
/// at `offset`. If `boundary` is provided and falls in range it also
/// gets a `|` marker.
#[cfg(feature = "trace")]
pub(super) fn dump_offset_context(
    file_path: &Path,
    offset: usize,
    boundary: Option<usize>,
) -> Option<(String, String)> {
    let mut f = std::fs::File::open(file_path).ok()?;
    let context = 80;
    let start = offset.saturating_sub(context);
    let mut buf = vec![0u8; context * 2];
    f.seek(SeekFrom::Start(start as u64)).ok()?;
    let n = f.read(&mut buf).unwrap_or(0);
    buf.truncate(n);

    let escaped = String::from_utf8_lossy(&buf).escape_debug().to_string();

    let esc_pos = |byte_off: usize| -> usize {
        if byte_off <= start {
            return 0;
        }
        let rel = (byte_off - start).min(buf.len());
        String::from_utf8_lossy(&buf[..rel])
            .escape_debug()
            .to_string()
            .len()
    };

    let off_col = esc_pos(offset);
    let bnd_col = boundary.map(esc_pos);

    let max_col = bnd_col.map_or(off_col, |b| off_col.max(b));
    let mut marker_line = vec![b' '; max_col + 1];
    if let (Some(b), Some(col)) = (boundary, bnd_col)
        && b >= start
        && b - start < buf.len()
    {
        marker_line[col] = b'|';
    }
    marker_line[off_col] = b'^';
    let marker = String::from_utf8_lossy(&marker_line).to_string();

    Some((escaped, marker))
}
