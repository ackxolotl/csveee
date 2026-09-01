//! SIMD-side of `parse_from` for the quoted path: drives [`Scanner`] in
//! 64-byte vectors, extracts B/E/R offsets, and emits records by striding
//! them in groups of `N` via [`assemble_records_fixed`].

use super::bitmask::{Prologue, VECTOR_BYTES, head_padded_prologue, neutral_pad};
use super::fieldcount::{FieldCounter, locate_bad_record};
use super::handoff;
use super::index::{FIXED_MAX_ARITY, assemble_records_fixed, extend_offsets};
use super::scan::{Scanner, ScannerConfig};
use crate::config::Config;
use crate::parser::driver::{ChunkStepper, StepCtx, StepResult};
use crate::parser::output::Output;

/// SIMD chunk stepper for the general / quoted path: drives the full
/// cross-vector [`Scanner`] and assembles records from begin/end indices.
/// [`super::cursor_stepper::SimdCursorStepper`] is the quote-free one.
pub(super) struct SimdIndexStepper {
    /// Dialect bytes handed to the scanner and the error paths.
    scanner_config: ScannerConfig,
    /// The cross-vector bitmask pipeline, with its own lag and carries.
    scanner: Scanner,

    /// Per-vector field-count verifier.
    field_counter: FieldCounter,

    /// True until [`head_padded_prologue`] has run, or been given up on.
    needs_align_prologue: bool,
    /// True until the stepper has pushed the `B[0]` the scanner omits.
    needs_chunk_start_b: bool,

    /// Field begins; records stride this and `e_offs` by `N`.
    b_offs: Vec<usize>,
    /// Field ends.
    e_offs: Vec<usize>,
    /// Chars to remove, applied during assembly.
    r_offs: Vec<usize>,

    /// Buffer offset where the next record begins; the consume limit.
    record_buf_start: usize,
    /// Absolute file offset just past the last completed record.
    last_record_end: Option<usize>,
    /// Field-pointer scratch, live only within one `step` / `finalize`.
    refs_scratch: [(*mut u8, usize); FIXED_MAX_ARITY],
}

impl SimdIndexStepper {
    pub(super) fn new(config: &Config) -> Self {
        // A run over `{term, term_b}` collapses to one record boundary.
        let (term, term_b) = config.terminator.bytes();
        let doubled_quotes = matches!(
            (config.quote, config.escape),
            (Some(q), Some(e)) if q == e
        );
        let scanner_config = ScannerConfig {
            delim: config.delimiter,
            term,
            term_b,
            quote: config.quote,
            doubled_quotes,
        };
        // `SimdChunkParser::supports` gates the arity; `new` rechecks the range.
        let field_counter =
            FieldCounter::new(config.field_count.expect("fixed field count") as u32);
        Self {
            scanner_config,
            field_counter,
            scanner: Scanner::new(scanner_config),
            needs_align_prologue: true,
            needs_chunk_start_b: true,
            b_offs: Vec::with_capacity(64),
            e_offs: Vec::with_capacity(64),
            r_offs: Vec::with_capacity(16),
            record_buf_start: 0,
            last_record_end: None,
            refs_scratch: [(std::ptr::null_mut(), 0); FIXED_MAX_ARITY],
        }
    }

    /// Is `c` a terminator byte (`term`, or `term_b` when set)?
    #[inline]
    fn is_term(&self, c: u8) -> bool {
        c == self.scanner_config.term || self.scanner_config.term_b == Some(c)
    }

    /// Run the verifier on a vector drained during [`Self::finalize`].
    /// The scanner's lag leaves the EOF vectors unresolved until here, so
    /// without this the file's last record would escape arity checking.
    fn verify_tail(
        &mut self,
        out: &super::scan::VectorOutput,
        buf: &[u8],
        total_consumed: usize,
    ) -> Result<(), crate::Error> {
        if !self.field_counter.verify(out.d_struct, out.term) {
            return Err(self.arity_error(buf, total_consumed));
        }
        Ok(())
    }

    /// Turn a verifier rejection into a positioned `NumberOfFields` by
    /// re-walking from the oldest record pending emission.
    #[cold]
    fn arity_error(&self, buf: &[u8], total_consumed: usize) -> crate::Error {
        let cfg = self.scanner_config;
        let n = self.field_counter.n();
        let (start, found) = locate_bad_record(
            buf,
            self.record_buf_start,
            n,
            cfg.delim,
            cfg.term,
            cfg.term_b,
            cfg.quote,
        );
        crate::Error::NumberOfFields {
            expected: n,
            found,
            position: crate::error::Position {
                byte_offset: total_consumed + start,
            },
        }
    }

    /// Drop the emitted prefix; only the live suffix moves, so the cost is
    /// independent of the prefix length.
    fn drop_emitted_prefix(&mut self, fields: usize, removals: usize) {
        self.b_offs.drain(..fields);
        self.e_offs.drain(..fields);
        self.r_offs.drain(..removals);
    }

    /// Emit the records pending in `b/e_offs[*head..]`, stopping after the
    /// first whose end crosses the chunk boundary. `Some(end)` once it is
    /// crossed; a record terminating past `resolved_end` defers.
    #[allow(clippy::too_many_arguments)]
    fn flush_records<O, E>(
        &mut self,
        buf: &mut [u8],
        head: &mut usize,
        r_head: &mut usize,
        resolved_end: usize,
        remaining_in_chunk: usize,
        total_consumed: usize,
        n: usize,
        emit: &mut E,
    ) -> Result<Option<usize>, crate::Error>
    where
        O: Output + ?Sized,
        E: FnMut(&mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        // A run reaching the boundary needs no end, and the cap stops a
        // long run being re-walked on every refill.
        let walk_cap = resolved_end.min(remaining_in_chunk.saturating_add(1));
        // Terminator run of the record whose last field is `e_idx`: outer
        // `None` while unresolved, inner `None` at the boundary. Must run
        // before `assemble` mutates buf.
        let term_of = |e_idx: usize| -> Option<(usize, Option<usize>)> {
            let mut p = self.e_offs[e_idx];
            while p < resolved_end && !self.is_term(buf[p]) {
                p += 1;
            }
            if p >= resolved_end {
                return None;
            }
            let run_start = p;
            while p + 1 < walk_cap && self.is_term(buf[p + 1]) {
                p += 1;
            }
            if p + 1 < walk_cap {
                Some((run_start, Some(p))) // ended inside the window
            } else if walk_cap < resolved_end {
                Some((run_start, None)) // hit the boundary cap
            } else {
                // Hit `resolved_end`: the run may continue into the
                // pending vector, so defer rather than guess its end.
                None
            }
        };
        // Trim trailing records whose boundary isn't resolved yet. A loop,
        // not one decrement: deferring can leave the new last record
        // unresolved too.
        let mut avail = (self.e_offs.len() - *head) / n;
        let mut last = loop {
            if avail == 0 {
                return Ok(None);
            }
            if let Some(t) = term_of(*head + avail * n - 1) {
                break t;
            }
            avail -= 1;
        };
        // Past the boundary the walk below can only agree, so skip it.
        let ends_before_boundary =
            |i: usize| matches!(term_of(i), Some((_, Some(end))) if end < remaining_in_chunk);
        let (count, stop) = if remaining_in_chunk == 0 {
            (1, true)
        } else if last.1.is_some_and(|end| end < remaining_in_chunk) {
            (avail, false)
        } else {
            let mut k = 0;
            while ends_before_boundary(*head + (k + 1) * n - 1) {
                k += 1;
            }
            (k + 1, true)
        };
        // Only the stopping paths shorten the batch; a record earlier than
        // `avail` resolves whenever `avail` does.
        if count != avail {
            last = term_of(*head + count * n - 1).expect("earlier records resolve too");
        }
        let (last_run_start, last_run_end) = last;
        assemble_records_fixed::<O, _>(
            buf,
            n,
            &self.b_offs[*head..],
            &self.e_offs[*head..],
            &self.r_offs[*r_head..],
            count,
            &mut self.refs_scratch,
            emit,
        )
        .map_err(crate::Error::from_user)?;
        let consumed = count * n;

        // No end means this flush stops, so the cursor is never read again.
        if let Some(end) = last_run_end {
            self.record_buf_start = end + 1;
        }
        let end = total_consumed + handoff(last_run_start, remaining_in_chunk, stop);
        self.last_record_end = Some(end);

        // Advance the R cursor past the consumed fields.
        let last_consumed_e = self.e_offs[*head + consumed - 1];
        while *r_head < self.r_offs.len() && self.r_offs[*r_head] <= last_consumed_e {
            *r_head += 1;
        }
        *head += consumed;

        Ok(stop.then_some(end))
    }
}

impl<O: Output + ?Sized> ChunkStepper<O> for SimdIndexStepper {
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

        if self.needs_chunk_start_b {
            // The chunk's first field begins at byte 0; an opening quote
            // there gets `R[0] = 1`, so the assembly strips it.
            self.b_offs.push(0);
            self.needs_chunk_start_b = false;
        }

        let mut cursor = scan_start;

        // Put the scan on a 64-byte boundary before the first real vector;
        // the stride fixes every later load. The prologue is an ordinary
        // `Scanner::step` whose neutral pad leaves the carries where a
        // scan starting at the record would.
        if self.needs_align_prologue && cursor == 0 {
            let cfg = self.scanner_config;
            let pad = neutral_pad(cfg.delim, cfg.term, cfg.term_b, cfg.quote);
            match head_padded_prologue(buf, scan_end, pad) {
                Prologue::Vector(v, misalign) => {
                    let first = self.scanner.step(&v);
                    debug_assert!(first.is_none(), "prologue is the scanner's first vector");
                    // The next iteration's `pending_base` is exactly the
                    // `-misalign` the wrapped base needs, and it always runs.
                    cursor = VECTOR_BYTES - misalign;
                    self.needs_align_prologue = false;
                }
                Prologue::Aligned => self.needs_align_prologue = false,
                // Retrying is only possible while the loop below scans
                // nothing: one vector from 0 and `cursor` never returns.
                Prologue::TooShort => self.needs_align_prologue = scan_end < VECTOR_BYTES,
            }
        }

        // Assemble against the live suffixes and compact the consumed
        // prefix once per step, not on every vector.
        const COMPACT_THRESHOLD: usize = 2048;
        // In-loop flush threshold, in pending fields. Must be >= the max
        // arity (63) so a full record is always pending when it fires.
        const EMIT_BATCH_FIELDS: usize = 128;
        let mut head = 0usize;
        let mut r_head = 0usize;
        let n = self.field_counter.n();

        // Same call at both sites; a macro keeps the tokens identical
        // rather than adding a branch to the vector loop.
        macro_rules! flush {
            () => {
                match self.flush_records::<O, E>(
                    buf,
                    &mut head,
                    &mut r_head,
                    cursor - VECTOR_BYTES,
                    remaining_in_chunk,
                    total_consumed,
                    n,
                    emit,
                ) {
                    Ok(None) => {}
                    // Stepper is discarded after `Done`; no compaction needed.
                    Ok(Some(end)) => return StepResult::Done(end),
                    Err(e) => return StepResult::Error(e),
                }
            };
        }

        while cursor + VECTOR_BYTES <= scan_end {
            let pending_base = cursor.wrapping_sub(VECTOR_BYTES);
            let out = {
                let v: &[u8; VECTOR_BYTES] = (&buf[cursor..cursor + VECTOR_BYTES])
                    .try_into()
                    .expect("64-byte slice");
                self.scanner.step(v)
            };
            cursor += VECTOR_BYTES;

            if let Some(out) = out {
                // The verifier counts separators and threads its own
                // phase: the emit's E-count diverges for a quoted last
                // field, so it must not feed back in here.
                if !self.field_counter.verify(out.d_struct, out.term) {
                    return StepResult::Error(self.arity_error(&buf[..scan_end], total_consumed));
                }
                extend_offsets(&mut self.b_offs, out.b, pending_base);
                extend_offsets(&mut self.e_offs, out.e, pending_base);
                extend_offsets(&mut self.r_offs, out.r, pending_base);
            }

            // Batched: a per-vector emit costs more in setup than the
            // assembly itself. The flush below still emits every record
            // this step completed, before suspending.
            if self.e_offs.len() - head < EMIT_BATCH_FIELDS {
                continue;
            }
            flush!();

            // Compact periodically to keep the live suffix small.
            if head >= COMPACT_THRESHOLD {
                self.drop_emitted_prefix(head, r_head);
                head = 0;
                r_head = 0;
            }
        }

        // The in-loop flush only fires on the batch threshold, so pick up
        // the records this step's tail vectors completed.
        if self.e_offs.len() - head >= n && cursor >= VECTOR_BYTES {
            flush!();
        }

        // Final compaction so the persisted scratches hold only live entries.
        if head != 0 {
            self.drop_emitted_prefix(head, r_head);
        }

        // The pending vector's bytes must stay, so it pins the consume limit.
        StepResult::Suspended {
            resume_at: cursor,
            consumable: self.record_buf_start,
            last_record_end: self.last_record_end,
        }
    }

    fn consume(&mut self, n: usize) {
        self.record_buf_start = self.record_buf_start.saturating_sub(n);
        for x in &mut self.b_offs {
            *x = x.saturating_sub(n);
        }
        for x in &mut self.e_offs {
            *x = x.saturating_sub(n);
        }
        for x in &mut self.r_offs {
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
        let mut cursor = scan_start;

        if scan_end == 0 && cursor == 0 {
            return Ok(self.last_record_end);
        }

        // 1. Feed the tail bytes as a neutral-padded vector, splicing a
        //    synthetic terminator in at `scan_end` when the last real byte
        //    isn't one, so the trailing record completes. Splicing after a
        //    real terminator would fabricate an empty line the verifier
        //    rejects, hence the guard.
        let real_tail_len = scan_end.saturating_sub(cursor);
        let term_a = self.scanner_config.term;
        let needs_virtual_term = scan_end > 0 && !self.is_term(buf[scan_end - 1]);
        if real_tail_len > 0 || needs_virtual_term {
            debug_assert!(real_tail_len < VECTOR_BYTES);
            let pad = neutral_pad(
                self.scanner_config.delim,
                self.scanner_config.term,
                self.scanner_config.term_b,
                self.scanner_config.quote,
            );
            let mut v = [pad; VECTOR_BYTES];
            v[..real_tail_len].copy_from_slice(&buf[cursor..scan_end]);
            if needs_virtual_term {
                // In-bounds since `real_tail_len < VECTOR_BYTES`.
                v[real_tail_len] = term_a;
            }

            let pending_base = cursor.wrapping_sub(VECTOR_BYTES);
            if let Some(out) = self.scanner.step(&v) {
                self.verify_tail(&out, &buf[..scan_end], total_consumed)?;
                extend_offsets(&mut self.b_offs, out.b, pending_base);
                extend_offsets(&mut self.e_offs, out.e, pending_base);
                extend_offsets(&mut self.r_offs, out.r, pending_base);
            }
            cursor += VECTOR_BYTES;
        }

        // 2. Drain the scanner's last pending vector, based one vector
        //    back from the cursor.
        let final_base = cursor.wrapping_sub(VECTOR_BYTES);
        if let Some(out) = self.scanner.finalize() {
            self.verify_tail(&out, &buf[..scan_end], total_consumed)?;
            extend_offsets(&mut self.b_offs, out.b, final_base);
            extend_offsets(&mut self.e_offs, out.e, final_base);
            extend_offsets(&mut self.r_offs, out.r, final_base);
        }

        // 2b. Ending mid-quoted-field mirrors the DFA's UnclosedQuote.
        if self.scanner.in_quotes() {
            let b = self.b_offs.last().copied().unwrap_or(scan_end);
            let position =
                if b > 0 && b <= scan_end && Some(buf[b - 1]) == self.scanner_config.quote {
                    b - 1
                } else {
                    b
                };
            return Err(crate::Error::UnclosedQuote {
                position: crate::error::Position {
                    byte_offset: total_consumed + position,
                },
            });
        }

        // 3. Trim offsets in the pad region. A trailing empty field's `B`
        //    and the synthetic terminator's `E` both sit at `scan_end`,
        //    hence `<=`; `R` never coincides with an `E`, hence `<`.
        self.b_offs.retain(|&x| x <= scan_end);
        self.e_offs.retain(|&x| x <= scan_end);
        self.r_offs.retain(|&x| x < scan_end);

        // 4. One record at a time, so the per-record boundary stop applies
        //    (a chunk under 64 B never iterates in `step`).
        let n = self.field_counter.n();
        let mut head = 0usize;
        let mut r_head = 0usize;
        while self.e_offs.len() - head >= n {
            // The last field's E is the terminator or the closing quote,
            // so scan forward to the terminator and across its run.
            let last_e = self.e_offs[head + n - 1];
            let mut rec_end = last_e;
            while rec_end < scan_end && !self.is_term(buf[rec_end]) {
                rec_end += 1;
            }
            let run_start = rec_end;
            while rec_end + 1 < scan_end && self.is_term(buf[rec_end + 1]) {
                rec_end += 1;
            }
            if let Err(e) = assemble_records_fixed::<O, _>(
                buf,
                n,
                &self.b_offs[head..],
                &self.e_offs[head..],
                &self.r_offs[r_head..],
                1,
                &mut self.refs_scratch,
                emit,
            ) {
                return Err(crate::Error::from_user(e));
            }
            self.record_buf_start = rec_end + 1;
            let stop = self.record_buf_start > remaining_in_chunk;
            self.last_record_end =
                Some(total_consumed + handoff(run_start, remaining_in_chunk, stop));
            while r_head < self.r_offs.len() && self.r_offs[r_head] <= last_e {
                r_head += 1;
            }
            head += n;

            if stop {
                // Later records belong to the next chunk.
                return Ok(self.last_record_end);
            }
        }

        Ok(self.last_record_end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RecordTerminator;

    fn lf_config(columns: usize) -> Config {
        let mut c = Config::default();
        c.terminator = RecordTerminator::LF;
        c.field_count = Some(columns);
        c
    }

    /// Run one `step` over `buf[..scan_end]`, returning `resume_at`.
    fn step_to(
        stepper: &mut SimdIndexStepper,
        buf: &mut [u8],
        at: usize,
        scan_end: usize,
    ) -> usize {
        let len = buf.len();
        let mut emit =
            |_: &mut [&mut [u8]]| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                Ok(())
            };
        let ctx = StepCtx {
            buf,
            scan_start: at,
            scan_end,
            remaining_in_chunk: len,
            total_consumed: 0,
        };
        match <SimdIndexStepper as ChunkStepper<[u8]>>::step(stepper, ctx, &mut emit) {
            StepResult::Suspended { resume_at, .. } => resume_at,
            _ => panic!("expected Suspended"),
        }
    }

    /// `Prologue::TooShort` is only worth retrying while the vector loop
    /// scans nothing; once it runs, `cursor` never returns to 0.
    #[test]
    fn prologue_retries_only_while_the_vector_loop_stays_idle() {
        let config = lf_config(2);
        let input = b"aa,bb\n".repeat(64);

        for misalign in 1..VECTOR_BYTES {
            let mut backing = vec![0u8; input.len() + 2 * VECTOR_BYTES];
            let head = (VECTOR_BYTES - backing.as_ptr() as usize % VECTOR_BYTES) % VECTOR_BYTES;
            let base = head + misalign;
            let ptr = backing.as_ptr() as usize + base;
            backing[base..base + input.len()].copy_from_slice(&input);
            let buf = &mut backing[base..base + input.len()];
            assert_eq!(ptr % VECTOR_BYTES, misalign);

            // Under a vector: nothing scanned, so the prologue stays live
            // and the retry lands every later load on 64 bytes.
            let mut stepper = SimdIndexStepper::new(&config);
            let resume = step_to(&mut stepper, buf, 0, VECTOR_BYTES - 1);
            assert_eq!(resume, 0);
            assert!(stepper.needs_align_prologue, "misalign={misalign}");
            let resume = step_to(&mut stepper, buf, resume, input.len());
            assert!(!stepper.needs_align_prologue);
            assert_eq!((ptr + resume) % VECTOR_BYTES, 0, "misalign={misalign}");

            // A whole vector, but still short of `skip + 64`: the loop
            // scans it unaligned, so the prologue is given up on.
            let mut stepper = SimdIndexStepper::new(&config);
            step_to(&mut stepper, buf, 0, VECTOR_BYTES);
            assert!(!stepper.needs_align_prologue, "misalign={misalign}");
        }
    }
}
