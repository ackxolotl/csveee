//! DFA-side of `parse_from`: per-byte state machine + record emission.
//!
//! Plugs into [`crate::parser::driver::ChunkDriver`] via [`ChunkStepper`].
//! The driver owns fill/consume, UTF-8 validation, and offset shifts;
//! this file owns the inner hot loop, write-cursor compaction, field
//! accumulation, and EOF finalization for terminal DFA states.
//!
//! # Buffer access discipline
//!
//! Pointers cannot outlive a call (a grow relocates the buffer, a
//! consume can wrap it), so a record still open on return is demoted to
//! offsets in `pending_spans` and promoted back on the next entry.

use std::ops::ControlFlow;

use super::table::{Dfa, ERROR_STATE, FIRST_ACTION, RECORD_END_STATE, State, StructuralNeedles};
use crate::config::Config;
use crate::parser::driver::{ChunkStepper, StepCtx, StepResult};
use crate::parser::output::{self, Output};

/// DFA stepper. Holds all state that must survive between driver iterations.
pub(super) struct DfaStepper<'a> {
    /// Transition table and structural-byte needles for the configured dialect.
    dfa: &'a Dfa,
    /// Strip leading and trailing ASCII whitespace from each field before emitting.
    trim_fields: bool,
    /// Current DFA state, carried across refills so records may span them.
    dfa_state: State,
    /// Buffer offset where the current field started.
    field_start: usize,
    /// In-place compaction write cursor; lags the read pos once escapes are removed.
    write_pos: usize,
    /// Buffer offset of the record in flight; caps what the driver may consume.
    record_buf_start: usize,
    /// Fields of the record in flight; live only within one `step` / `finalize`.
    refs: Vec<(*mut u8, usize)>,
    /// `refs` demoted to offsets across a refill; never both non-empty.
    pending_spans: Vec<(usize, usize)>,
    /// Absolute file offset just past the last completed record.
    last_record_end: Option<usize>,
    /// Absolute file offset of the current record, for error reporting only.
    record_in_start: usize,
    /// The same, for the field in flight.
    field_in_start: usize,
    /// Arity the callback declares; `None` in variadic mode. Checked per record.
    expected_fields: Option<usize>,
}

impl<'a> DfaStepper<'a> {
    pub(super) fn new(dfa: &'a Dfa, config: &Config, start_offset: usize) -> Self {
        Self {
            dfa,
            trim_fields: config.trim,
            dfa_state: State::RecordStart,
            field_start: 0,
            write_pos: 0,
            record_buf_start: 0,
            record_in_start: start_offset,
            field_in_start: start_offset,
            refs: Vec::with_capacity(128),
            pending_spans: Vec::with_capacity(128),
            last_record_end: None,
            expected_fields: config.field_count,
        }
    }

    /// Promote spans carried across a refill to pointers against the
    /// current `base`. No-op in the steady state.
    #[inline]
    fn resume(&mut self, base: *mut u8) {
        if self.pending_spans.is_empty() {
            return;
        }
        self.refs.extend(
            self.pending_spans
                .drain(..)
                // SAFETY: `consume` tracks every consume, so each offset is still in bounds
                .map(|(off, len)| (unsafe { base.add(off) }, len)),
        );
    }

    /// Demote the record in flight to offsets before suspending back to
    /// the driver, whose next `fill` / `consume` may move the buffer.
    #[inline]
    fn suspend(&mut self, base: *mut u8) {
        if self.refs.is_empty() {
            return;
        }
        self.pending_spans.extend(
            self.refs
                .drain(..)
                // SAFETY: both derive from `base`, so the distance is in range and non-negative.
                .map(|(p, len)| (unsafe { p.offset_from(base) } as usize, len)),
        );
    }

    /// Push the current field span (`field_start..write_pos`, optionally
    /// trimmed) onto `refs`. Caller advances the cursors afterward.
    ///
    /// # Safety
    ///
    /// `base` must be the current buffer base and the span lie within it.
    #[inline]
    unsafe fn push_field(&mut self, base: *mut u8) {
        debug_assert!(self.field_start <= self.write_pos);
        let (mut off, mut len) = (self.field_start, self.write_pos - self.field_start);
        if self.trim_fields {
            // SAFETY: the span lies within the buffer and is not held across a write
            let s = unsafe { std::slice::from_raw_parts(base.add(off), len) };
            let t = s.trim_ascii();
            off += t.as_ptr() as usize - s.as_ptr() as usize;
            len = t.len();
        }
        self.refs.push((unsafe { base.add(off) }, len));
    }

    /// Point every field cursor at `pos`: the next field starts here
    /// and nothing has been written for it yet.
    #[inline]
    fn start_field_at(&mut self, total_consumed: usize, pos: usize) {
        self.field_start = pos;
        self.field_in_start = total_consumed + pos;
        self.write_pos = pos;
    }

    /// Scan-and-copy fast path for the two "content" states, where the
    /// DFA self-loops with `has_output = true` on every non-structural
    /// byte. Returns the new read position.
    ///
    /// # Safety
    ///
    /// `pos <= scan_end` must hold and `..scan_end` be readable and
    /// writable through `base`.
    #[inline(always)]
    unsafe fn scan_run(
        &mut self,
        base: *mut u8,
        pos: usize,
        scan_end: usize,
        needles: &StructuralNeedles,
    ) -> usize {
        // SAFETY: `pos <= scan_end`; not held across any write through `base`
        let haystack = unsafe { std::slice::from_raw_parts(base.add(pos), scan_end - pos) };
        let next_pos = match needles.find(haystack) {
            Some(off) => pos + off,
            None => scan_end,
        };
        let run_len = next_pos - pos;
        if run_len != 0 && self.write_pos != pos {
            // SAFETY: both ranges lie within `..scan_end`
            unsafe { std::ptr::copy(base.add(pos), base.add(self.write_pos), run_len) };
        }
        self.write_pos += run_len;
        next_pos
    }

    /// Check the field count of the in-flight record against [`Self::expected_fields`],
    #[inline]
    fn check_arity(&self) -> Result<(), crate::Error> {
        let Some(expected) = self.expected_fields else {
            return Ok(());
        };
        let found = self.refs.len();
        if found == expected {
            return Ok(());
        }
        Err(crate::Error::NumberOfFields {
            expected,
            found,
            position: crate::error::Position {
                byte_offset: self.record_in_start,
            },
        })
    }

    /// Arity-check the record in flight, then hand it to `emit`.
    #[inline]
    fn emit_record<O, E>(&mut self, emit: &mut E) -> Result<(), crate::Error>
    where
        O: Output + ?Sized,
        E: FnMut(&mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        self.check_arity()?;
        // SAFETY: a `&mut str` here is known-good UTF-8
        let res = unsafe { output::emit_record::<O, _>(&mut self.refs, emit) };
        self.refs.clear();
        res.map_err(crate::Error::from_user)
    }

    /// Close out the field the action state `s` landed on, and — for
    /// `RECORD_END_STATE` — the record with it. `pos` is one past the
    /// byte that triggered the action.
    ///
    /// `Break` ends the chunk: a parse error, or a record that closed on
    /// the chunk boundary. Nothing is suspended on the error path — the
    /// driver hands the error straight out of `run` and drops the stepper.
    ///
    /// # Safety
    ///
    /// `field_start <= write_pos <= pos <= scan_end`, all within `base`.
    #[inline(always)]
    unsafe fn apply_action<O, E>(
        &mut self,
        s: u8,
        base: *mut u8,
        pos: usize,
        remaining_in_chunk: usize,
        total_consumed: usize,
        emit: &mut E,
    ) -> ControlFlow<StepResult>
    where
        O: Output + ?Sized,
        E: FnMut(&mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        if s == ERROR_STATE {
            return ControlFlow::Break(StepResult::Error(crate::Error::InvalidQuote {
                position: crate::error::Position {
                    byte_offset: total_consumed + pos - 1,
                },
            }));
        }

        // SAFETY: `field_start <= write_pos <= scan_end`.
        unsafe { self.push_field(base) };
        self.start_field_at(total_consumed, pos);

        if s != RECORD_END_STATE {
            // FieldEnd — next field.
            self.dfa_state = State::FieldStart;
            return ControlFlow::Continue(());
        }

        if let Err(e) = self.emit_record::<O, E>(emit) {
            return ControlFlow::Break(StepResult::Error(e));
        }

        let anchor = pos - 1;

        let end = total_consumed + anchor;
        self.last_record_end = Some(end);
        self.record_buf_start = pos;
        self.start_field_at(total_consumed, pos);
        self.dfa_state = State::RecordStart;

        // The byte `find_record_start` stops on from the other side.
        if anchor >= remaining_in_chunk {
            debug_assert!(self.refs.is_empty(), "record in flight after emit");
            return ControlFlow::Break(StepResult::Done(end));
        }
        ControlFlow::Continue(())
    }

    /// How far the driver may consume once the scan suspends at `pos`.
    #[inline]
    fn consumable(&self, pos: usize) -> usize {
        if self.pending_spans.is_empty() && self.field_start == self.write_pos {
            pos
        } else {
            self.record_buf_start
        }
    }
}

impl<'a, O: Output + ?Sized> ChunkStepper<O> for DfaStepper<'a> {
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
        debug_assert!(scan_end <= buf.len());
        let base = buf.as_mut_ptr();
        let dfa = self.dfa;
        let mut pos = scan_start;

        self.resume(base);

        while pos < scan_end {
            match self.dfa_state {
                // SAFETY: `pos < scan_end <= buf.len()` from the loop condition
                State::InField => {
                    pos = unsafe { self.scan_run(base, pos, scan_end, dfa.in_field_needles()) }
                }
                State::InQuotedField => {
                    pos = unsafe { self.scan_run(base, pos, scan_end, dfa.in_quoted_needles()) }
                }

                // Bulk-skip empty-line terminators without going through the table
                State::RecordStart => {
                    let mask = dfa.record_start_structural();
                    let run_start = pos;
                    // SAFETY: `pos < scan_end` on every read.
                    while pos < scan_end && !mask[unsafe { *base.add(pos) } as usize] {
                        pos += 1;
                    }

                    let anchor = remaining_in_chunk.max(run_start);
                    if anchor < pos {
                        debug_assert!(self.refs.is_empty(), "record in flight at RecordStart");
                        let end = total_consumed + anchor;
                        self.last_record_end = Some(end);
                        return StepResult::Done(end);
                    }

                    self.record_in_start = total_consumed + pos;
                    self.field_in_start = total_consumed + pos;
                }
                _ => {}
            }
            // Only the arms above can land on `scan_end`; for the rest the
            // loop condition already rules it out.
            if pos >= scan_end {
                break;
            }

            // ── Main DFA step ────────────────────────────────────
            // SAFETY: `pos < scan_end` from the loop condition
            let byte = unsafe { *base.add(pos) };
            let trans = dfa.transition(self.dfa_state, byte);
            if trans.has_output() {
                if self.write_pos != pos {
                    // SAFETY: `write_pos <= pos < scan_end`.
                    unsafe { *base.add(self.write_pos) = byte };
                }
                self.write_pos += 1;
            }
            pos += 1;

            // Action states are never stored: `apply_action` installs the
            // interior state that follows, and the error path returns.
            let s = trans.next_raw();
            if s < FIRST_ACTION {
                self.dfa_state = trans.next_state();
                continue;
            }

            // SAFETY: `field_start <= write_pos <= pos <= scan_end`.
            if let ControlFlow::Break(result) = unsafe {
                self.apply_action::<O, E>(s, base, pos, remaining_in_chunk, total_consumed, emit)
            } {
                return result;
            }
        }

        // demote
        self.suspend(base);

        StepResult::Suspended {
            resume_at: pos,
            consumable: self.consumable(pos),
            last_record_end: self.last_record_end,
        }
    }

    fn consume(&mut self, n: usize) {
        self.field_start = self.field_start.saturating_sub(n);
        self.write_pos = self.write_pos.saturating_sub(n);
        self.record_buf_start = self.record_buf_start.saturating_sub(n);
        for span in self.pending_spans.iter_mut() {
            debug_assert!(span.0 >= n, "consumed past a pending field span");
            span.0 = span.0.saturating_sub(n);
        }
    }

    fn finalize<E>(&mut self, ctx: StepCtx<'_>, emit: &mut E) -> Result<Option<usize>, crate::Error>
    where
        E: FnMut(&mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        let StepCtx {
            buf,
            scan_end,
            total_consumed,
            ..
        } = ctx;

        match self.dfa_state {
            State::RecordStart | State::InComment => Ok(None),

            // Not `field_start`: a write cursor, so it lags a skipped run
            State::InQuotedField | State::InEscapedQuote => Err(crate::Error::UnclosedQuote {
                position: crate::error::Position {
                    byte_offset: self.field_in_start,
                },
            }),

            State::FieldStart | State::InField | State::AfterClosingQuote => {
                let base = buf.as_mut_ptr();
                self.resume(base);
                // SAFETY: `field_start <= write_pos <= scan_end`
                unsafe { self.push_field(base) };
                self.emit_record(emit)?;
                Ok(Some(total_consumed + scan_end))
            }

            // Action states never leak out of the hot loop — they're
            // handled inline and converted back to interior states.
            State::FieldEnd | State::RecordEnd | State::Error => {
                debug_assert!(
                    false,
                    "action state leaked past hot loop: {:?}",
                    self.dfa_state
                );
                Ok(None)
            }
        }
    }
}
