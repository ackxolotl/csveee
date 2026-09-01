//! Quote-free SIMD stepper, the single-stream `cell_start` path: a
//! vector's boundaries are just `delim | term`, collapsing the
//! [`super::scan::Scanner`] pipeline to one stream, held back a vector.

use super::bitmask::{
    Prologue, VECTOR_BYTES, head_padded_prologue, match_structural, neutral_pad,
    terminator_run_ends,
};
use super::fieldcount::{FieldCounter, locate_bad_record};
use super::handoff;
use super::index::{FIXED_MAX_ARITY, extend_offsets};
use crate::config::Config;
use crate::parser::driver::{ChunkStepper, StepCtx, StepResult};
use crate::parser::output::Output;

pub(super) struct SimdCursorStepper {
    /// Field delimiter byte.
    delim: u8,
    /// Primary terminator byte; a maximal run over both is one boundary.
    term_a: u8,
    /// Second terminator byte (the `\r` of a CRLF dialect), if any.
    term_b: Option<u8>,
    /// Per-vector field-count verifier.
    field_counter: FieldCounter,
    /// True until [`head_padded_prologue`] has run, or been given up on.
    needs_align_prologue: bool,
    /// The held vector, resolved against the next one's terminator LSB.
    pending: Option<(u64, u64, usize)>,
    /// Boundary offsets of the published vectors; records stride by `n`.
    bounds: Vec<usize>,
    /// Buffer offset where the in-flight field starts; the consume limit.
    cell_start: usize,
    /// Set once a terminator *run* is seen, taking the emit off its tight loop.
    has_runs: bool,
    /// Absolute file offset just past the last completed record.
    last_record_end: Option<usize>,
    /// Field-pointer scratch, live only within one `step` / `finalize`.
    refs_scratch: [(*mut u8, usize); FIXED_MAX_ARITY],
}

impl SimdCursorStepper {
    pub(super) fn new(config: &Config) -> Self {
        let (term_a, term_b) = config.terminator.bytes();
        debug_assert!(
            config.quote.is_none(),
            "SimdCursorStepper is the quote-free path; parse_from must gate on quote.is_none()",
        );
        // `SimdChunkParser::supports` gates the arity; `new` rechecks the range.
        let field_counter =
            FieldCounter::new(config.field_count.expect("fixed field count") as u32);
        Self {
            delim: config.delimiter,
            term_a,
            term_b,
            field_counter,
            needs_align_prologue: true,
            pending: None,
            bounds: Vec::with_capacity(64),
            cell_start: 0,
            has_runs: false,
            last_record_end: None,
            refs_scratch: [(std::ptr::null_mut(), 0); FIXED_MAX_ARITY],
        }
    }

    /// Is `c` a terminator byte (`term_a`, or `term_b` when set)?
    #[inline]
    fn is_term(&self, c: u8) -> bool {
        c == self.term_a || self.term_b == Some(c)
    }

    /// Run the verifier on a vector resolved during [`Self::finalize`].
    /// The lag leaves the EOF vectors unresolved until here, so without
    /// this the file's last record would escape arity checking.
    fn verify_tail(
        &mut self,
        d_struct: u64,
        run_end: u64,
        buf: &[u8],
        total_consumed: usize,
    ) -> Result<(), crate::Error> {
        if !self.field_counter.verify(d_struct, run_end) {
            return Err(self.arity_error(buf, total_consumed));
        }
        Ok(())
    }

    /// Turn a verifier rejection into a positioned `NumberOfFields` by
    /// re-walking from the oldest record pending emission.
    #[cold]
    fn arity_error(&self, buf: &[u8], total_consumed: usize) -> crate::Error {
        let n = self.field_counter.n();
        let (start, found) = locate_bad_record(
            buf,
            self.cell_start,
            n,
            self.delim,
            self.term_a,
            self.term_b,
            None,
        );
        crate::Error::NumberOfFields {
            expected: n,
            found,
            position: crate::error::Position {
                byte_offset: total_consumed + start,
            },
        }
    }

    /// Emit `count` records from `bounds[base0..]`, striding by `n` from
    /// the running `cs`. Returns the consume cursor `run_end + 1` and the
    /// last record's run start.
    fn emit_records<O, E>(
        &mut self,
        buf: &mut [u8],
        base0: usize,
        count: usize,
        n: usize,
        mut cs: usize,
        emit: &mut E,
    ) -> Result<(usize, usize), crate::Error>
    where
        O: Output + ?Sized,
        E: FnMut(&mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        let buf_ptr = buf.as_mut_ptr();
        // No runs yet, so each boundary *is* a field end: no strip needed.
        debug_assert!(count > 0, "emit_records needs at least one record");
        if !self.has_runs {
            for rec in 0..count {
                let base = base0 + rec * n;
                // SAFETY: `base + n - 1 < base0 + count*n ≤ bounds.len()`;
                // each span lies in `buf`; `i < n ≤ scratch.len()`.
                unsafe {
                    for i in 0..n {
                        let bound = *self.bounds.get_unchecked(base + i);
                        debug_assert!(bound >= cs, "bounds not ascending");
                        *self.refs_scratch.get_unchecked_mut(i) = (buf_ptr.add(cs), bound - cs);
                        cs = bound + 1;
                    }
                    // SAFETY: slots `0..n` hold ascending, non-overlapping
                    // ranges derived from `buf_ptr`; nothing writes `buf`.
                    crate::parser::output::emit_record::<O, _>(&mut self.refs_scratch[..n], emit)
                        .map_err(crate::Error::from_user)?;
                }
            }
            // One-byte terminators, so the last run starts at `cs - 1`.
            return Ok((cs, cs - 1));
        }

        // With runs, the boundary is a run's last terminator byte, so the
        // last field ends at its first: strip back within `[cs, bound)`.
        let ta = self.term_a;
        let tb = self.term_b.unwrap_or(self.term_a);
        let mut last_run_start = cs;
        for rec in 0..count {
            let base = base0 + rec * n;
            for i in 0..n - 1 {
                let bound = unsafe { *self.bounds.get_unchecked(base + i) };
                debug_assert!(bound >= cs, "bounds not ascending");
                unsafe {
                    *self.refs_scratch.get_unchecked_mut(i) = (buf_ptr.add(cs), bound - cs);
                }
                cs = bound + 1;
            }
            let bound = unsafe { *self.bounds.get_unchecked(base + n - 1) };
            debug_assert!(bound >= cs, "bounds not ascending");
            let mut end = bound;
            // Through `buf_ptr`, not `buf`: a reborrow of the slice would
            // be a foreign access to the pointers already in `scratch`.
            // SAFETY: `cs <= end <= bound`, an offset into `buf`.
            while end > cs && {
                let c = unsafe { *buf_ptr.add(end - 1) };
                c == ta || c == tb
            } {
                end -= 1;
            }
            unsafe {
                *self.refs_scratch.get_unchecked_mut(n - 1) = (buf_ptr.add(cs), end - cs);
            }
            last_run_start = end;
            cs = bound + 1;

            // SAFETY: slots `0..n` hold ascending, non-overlapping ranges
            // derived from `buf_ptr`; nothing writes `buf`.
            unsafe {
                crate::parser::output::emit_record::<O, _>(&mut self.refs_scratch[..n], emit)
            }
            .map_err(crate::Error::from_user)?;
        }
        Ok((cs, last_run_start))
    }

    /// Emit the records pending in `bounds[*head..]`, stopping after the
    /// first whose terminator crosses the chunk boundary — a record's
    /// terminator is its `n`-th bound. `Some(end)` once it is crossed.
    fn flush<O, E>(
        &mut self,
        buf: &mut [u8],
        head: &mut usize,
        remaining_in_chunk: usize,
        total_consumed: usize,
        n: usize,
        emit: &mut E,
    ) -> Result<Option<usize>, crate::Error>
    where
        O: Output + ?Sized,
        E: FnMut(&mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        let avail = (self.bounds.len() - *head) / n;
        if avail == 0 {
            return Ok(None);
        }
        // Past the boundary the scans below can only agree, so skip them.
        let (count, stop) = if remaining_in_chunk == 0 {
            (1, true)
        } else if self.bounds[*head + avail * n - 1] < remaining_in_chunk {
            (avail, false)
        } else {
            let mut k = 1;
            while self.bounds[*head + k * n - 1] < remaining_in_chunk {
                k += 1;
            }
            (k, true)
        };

        let (cs, last_run_start) =
            self.emit_records::<O, E>(buf, *head, count, n, self.cell_start, emit)?;
        self.cell_start = cs;
        let end = total_consumed + handoff(last_run_start, remaining_in_chunk, stop);
        self.last_record_end = Some(end);
        *head += count * n;
        Ok(stop.then_some(end))
    }

    /// Scan one 64-byte vector → `(delim_mask, raw term_mask)`. The
    /// run-end collapse is the caller's job; it holds the lookahead.
    #[inline]
    fn scan_vector(&self, buf: &[u8], at: usize) -> (u64, u64) {
        let v: &[u8; VECTOR_BYTES] = (&buf[at..at + VECTOR_BYTES])
            .try_into()
            .expect("64-byte slice");
        let s = match_structural(v, self.delim, self.term_a, self.term_b, None);
        (s.delim, s.term)
    }
}

impl<O: Output + ?Sized> ChunkStepper<O> for SimdCursorStepper {
    fn step<E>(&mut self, ctx: StepCtx<'_>, emit: &mut E) -> StepResult
    where
        E: FnMut(&mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        let StepCtx {
            buf,
            scan_start,
            scan_end,
            remaining_in_chunk,
            total_consumed,
        } = ctx;

        let mut cursor = scan_start;

        // Put the scan on a 64-byte boundary before the first real vector;
        // the stride fixes every later load.
        if self.needs_align_prologue && cursor == 0 && self.pending.is_none() {
            let pad = neutral_pad(self.delim, self.term_a, self.term_b, None);
            match head_padded_prologue(buf, scan_end, pad) {
                Prologue::Vector(v, misalign) => {
                    let s = match_structural(&v, self.delim, self.term_a, self.term_b, None);
                    // Wrapped-negative base: bit `i` is offset `i - misalign`.
                    // The first loop iteration is guaranteed to run, so this
                    // never survives to a suspension.
                    self.pending = Some((s.delim, s.term, 0usize.wrapping_sub(misalign)));
                    cursor = VECTOR_BYTES - misalign;
                    self.needs_align_prologue = false;
                }
                Prologue::Aligned => self.needs_align_prologue = false,
                // Retrying is only possible while the loop below scans
                // nothing: one vector from 0 and `cursor` never returns.
                Prologue::TooShort => self.needs_align_prologue = scan_end < VECTOR_BYTES,
            }
        }

        const COMPACT_THRESHOLD: usize = 2048;
        const EMIT_BATCH_FIELDS: usize = 128;
        let mut head = 0usize;
        let n = self.field_counter.n();

        let mut pending = self.pending.take();

        while cursor + VECTOR_BYTES <= scan_end {
            let (delim_mask, term_mask) = self.scan_vector(buf, cursor);
            let base = cursor;
            cursor += VECTOR_BYTES;

            // The current vector's terminator LSB resolves the held one:
            // collapse its runs, verify, publish, then hold this one.
            if let Some((p_delim, p_term, pbase)) = pending {
                let run_end = terminator_run_ends(p_term, (term_mask & 1) != 0);
                self.has_runs |= run_end != p_term;
                let d_struct = p_delim | run_end;
                if !self.field_counter.verify(d_struct, run_end) {
                    return StepResult::Error(self.arity_error(&buf[..scan_end], total_consumed));
                }
                extend_offsets(&mut self.bounds, d_struct, pbase);
            }
            pending = Some((delim_mask, term_mask, base));

            if self.bounds.len() - head < EMIT_BATCH_FIELDS {
                continue;
            }
            match self.flush::<O, E>(buf, &mut head, remaining_in_chunk, total_consumed, n, emit) {
                Ok(None) => {}
                Ok(Some(end)) => {
                    self.pending = pending;
                    return StepResult::Done(end);
                }
                Err(e) => {
                    self.pending = pending;
                    return StepResult::Error(e);
                }
            }

            if head >= COMPACT_THRESHOLD {
                self.bounds.drain(..head);
                head = 0;
            }
        }
        self.pending = pending;

        // Flush records completed by the boundaries published this step.
        if self.bounds.len() - head >= n {
            match self.flush::<O, E>(buf, &mut head, remaining_in_chunk, total_consumed, n, emit) {
                Ok(None) => {}
                Ok(Some(end)) => return StepResult::Done(end),
                Err(e) => return StepResult::Error(e),
            }
        }

        if head != 0 {
            self.bounds.drain(..head);
        }

        StepResult::Suspended {
            resume_at: cursor,
            consumable: self.cell_start,
            last_record_end: self.last_record_end,
        }
    }

    fn consume(&mut self, n: usize) {
        self.cell_start = self.cell_start.saturating_sub(n);
        if let Some((_, _, ref mut pbase)) = self.pending {
            *pbase = pbase.saturating_sub(n);
        }
        for x in &mut self.bounds {
            debug_assert!(*x >= n, "consumed past a published bound");
            *x = x.saturating_sub(n);
        }
    }

    fn finalize<E>(&mut self, ctx: StepCtx<'_>, emit: &mut E) -> Result<Option<usize>, crate::Error>
    where
        E: FnMut(&mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        let StepCtx {
            buf,
            scan_start,
            scan_end,
            remaining_in_chunk,
            total_consumed,
        } = ctx;
        let cursor = scan_start;
        debug_assert!(cursor <= scan_end);
        if scan_end == 0 && cursor == 0 && self.pending.is_none() {
            return Ok(self.last_record_end);
        }

        // 1. Build the tail vector's raw masks (< 64 B), splicing a
        //    synthetic terminator in when the last real byte isn't one.
        let real_tail_len = scan_end.saturating_sub(cursor);
        let needs_virtual_term = scan_end > 0 && !self.is_term(buf[scan_end - 1]);
        let tail: Option<(u64, u64, usize)> = if real_tail_len > 0 || needs_virtual_term {
            debug_assert!(real_tail_len < VECTOR_BYTES);
            let pad = neutral_pad(self.delim, self.term_a, self.term_b, None);
            let mut v = [pad; VECTOR_BYTES];
            v[..real_tail_len].copy_from_slice(&buf[cursor..scan_end]);
            if needs_virtual_term {
                v[real_tail_len] = self.term_a;
            }
            let s = match_structural(&v, self.delim, self.term_a, self.term_b, None);
            Some((s.delim, s.term, cursor))
        } else {
            None
        };

        // 2. Resolve the held vector against the tail's first-terminator
        //    bit, and publish it first to keep `bounds` sorted.
        if let Some((p_delim, p_term, pbase)) = self.pending.take() {
            let next_lsb = tail.is_some_and(|(_, t, _)| (t & 1) != 0);
            let run_end = terminator_run_ends(p_term, next_lsb);
            self.has_runs |= run_end != p_term;
            self.verify_tail(p_delim | run_end, run_end, &buf[..scan_end], total_consumed)?;
            extend_offsets(&mut self.bounds, p_delim | run_end, pbase);
        }

        // 3. Publish the tail's boundaries (no continuation past it).
        if let Some((t_delim, t_term, tbase)) = tail {
            let run_end = terminator_run_ends(t_term, false);
            self.has_runs |= run_end != t_term;
            self.verify_tail(t_delim | run_end, run_end, &buf[..scan_end], total_consumed)?;
            extend_offsets(&mut self.bounds, t_delim | run_end, tbase);
        }

        // 4. Neutral padding leaves no bound past real data; the
        //    synthetic terminator sits at `scan_end`, hence `<=`.
        debug_assert!(
            self.bounds.last().is_none_or(|&x| x <= scan_end),
            "bound past the scanned bytes",
        );

        // 5. One record at a time, so the per-record boundary stop applies
        //    (a chunk under 64 B never iterates in `step`).
        let n = self.field_counter.n();
        let mut head = 0usize;
        while self.bounds.len() - head >= n {
            let (cs, last_run_start) =
                self.emit_records::<O, E>(buf, head, 1, n, self.cell_start, emit)?;
            self.cell_start = cs;
            let stop = self.cell_start > remaining_in_chunk;
            self.last_record_end =
                Some(total_consumed + handoff(last_run_start, remaining_in_chunk, stop));
            head += n;

            if stop {
                return Ok(self.last_record_end);
            }
        }

        Ok(self.last_record_end)
    }
}
