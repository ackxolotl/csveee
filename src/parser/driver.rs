//! Buffer-management driver shared by all chunk parsers.
//!
//! The driver owns everything independent of how a parser steps over
//! bytes: the fill/consume loop against a [`ChunkReader`], incremental
//! UTF-8 validation via [`Output::validate_chunk`], the three-way
//! consume decision at the tail of each outer iteration, and the EOF
//! UTF-8 tail check.

use super::output::Output;
use crate::io::ChunkReader;

/// Per-iteration view handed to the stepper.
pub(crate) struct StepCtx<'b> {
    /// The reader's mutable buffer.
    pub buf: &'b mut [u8],
    /// Where to resume scanning.
    pub scan_start: usize,
    /// Where to end scanning.
    pub scan_end: usize,
    /// Bytes remaining before the chunk boundary.
    pub remaining_in_chunk: usize,
    /// Sum of bytes consumed from the reader so far.
    pub total_consumed: usize,
}

/// Outcome of one stepper invocation.
pub(crate) enum StepResult {
    /// Reached `scan_end` without crossing the chunk boundary.
    Suspended {
        /// Where the stepper resumes.
        resume_at: usize,
        /// How many buffer bytes the driver may consume.
        consumable: usize,
        /// Last record-end as an absolute file offset.
        last_record_end: Option<usize>,
    },
    /// This chunk is done. Carries the final record's absolute end offset.
    Done(usize),
    /// Parser-level error.
    Error(crate::Error),
}

/// Pluggable per-byte parser. The driver calls `step` once per refill
/// and `consume` after each consume to keep the stepper's
/// internal state aligned with buffer movements.
pub(crate) trait ChunkStepper<O: Output + ?Sized> {
    /// Step the parser over `ctx.buf[ctx.scan_start..ctx.scan_end]`, emitting records.
    fn step<E>(&mut self, ctx: StepCtx<'_>, emit: &mut E) -> StepResult
    where
        E: FnMut(&mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Account for `n` consumed buffer bytes, e.g., by shifting any buffer-relative offsets.
    fn consume(&mut self, n: usize);

    /// Flush pending records, validate the terminal parser state.
    /// Returns the final record-end (if any) as an absolute file offset.
    fn finalize<E>(
        &mut self,
        ctx: StepCtx<'_>,
        emit: &mut E,
    ) -> Result<Option<usize>, crate::Error>
    where
        E: FnMut(&mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// Upper bound on how many fresh bytes one driver iteration UTF-8
/// validates. Readers whose `buffer()` exposes everything up to EOF
/// would otherwise make every chunk validate to the end of the file.
const VALIDATE_WINDOW: usize = 128 * 1024;

/// Drives a [`ChunkStepper`] over a [`ChunkReader`].
///
/// One driver per `parse_from` invocation. Holds buffer-level state
/// and contains the fill/consume calls in the parsing pipeline.
pub(crate) struct ChunkDriver<'r, R: ChunkReader> {
    /// Source of buffered bytes that gets filled and consumed.
    reader: &'r mut R,
    /// End of the UTF-8 validated prefix. Buffer-relative, `>= resume_at`.
    validated: usize,
    /// Where the next stepper call resumes. Buffer-relative.
    resume_at: usize,
    /// Absolute file offset of the bytes in the reader's buffer.
    total_consumed: usize,
}

impl<'r, R: ChunkReader> ChunkDriver<'r, R> {
    pub fn new(reader: &'r mut R, start_offset: usize) -> Self {
        Self {
            reader,
            validated: 0,
            resume_at: 0,
            total_consumed: start_offset,
        }
    }

    /// UTF-8 validate fresh buffer bytes, at most `VALIDATE_WINDOW` per call.
    fn validate_fresh<O: Output + ?Sized>(&mut self) -> crate::Result<()> {
        let buf = self.reader.buffer();
        let window_end = buf.len().min(self.validated + 4 + VALIDATE_WINDOW);
        if window_end <= self.validated {
            return Ok(());
        }
        match O::validate_chunk(&buf[self.validated..window_end]) {
            Ok(n) => {
                self.validated += n;
                Ok(())
            }
            Err(rel) => Err(crate::Error::Utf8 {
                position: crate::error::Position {
                    byte_offset: self.total_consumed + self.validated + rel,
                },
            }),
        }
    }

    /// Drive `stepper` until it reports `Done`, errors out, or EOF is
    /// reached. Returns the absolute byte offset just past the last
    /// completed record, or `None` if no record was completed.
    pub fn run<O, S, A, ST>(
        mut self,
        stepper: &mut ST,
        state: &mut S,
        acc: &A,
    ) -> crate::Result<Option<usize>>
    where
        O: Output + ?Sized,
        A: Fn(&mut S, &mut [&mut O]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
        ST: ChunkStepper<O>,
    {
        let mut last_record_end: Option<usize> = None;

        // the emit function
        let mut emit = |fields: &mut [&mut O]| acc(state, fields);

        loop {
            // refill the buffer (at least 4-bytes for multibyte UTF-8 sequences)
            self.reader.fill(self.validated + 4)?;

            // UTF-8 validate fresh buffer bytes. No progress means either no
            // new data at all, or a tail `fill` already had its chance to
            // complete – the check after the loop turns that into a UTF-8 error.
            let validated_before = self.validated;
            self.validate_fresh::<O>()?;
            if self.validated <= validated_before {
                break;
            }

            // set up the stepper context
            let ctx = StepCtx {
                scan_start: self.resume_at,
                scan_end: self.validated,
                remaining_in_chunk: self.reader.remaining_in_chunk(),
                total_consumed: self.total_consumed,
                buf: self.reader.buffer_mut(),
            };

            match stepper.step(ctx, &mut emit) {
                StepResult::Done(lre) => return Ok(Some(lre)),
                StepResult::Error(e) => return Err(e),
                StepResult::Suspended {
                    resume_at,
                    consumable,
                    last_record_end: lre,
                } => {
                    last_record_end = lre;

                    debug_assert!(consumable <= resume_at && resume_at <= self.validated);
                    // at `consume_limit == 0`, nothing moves and `fill` grows the ring buffer
                    if consumable > 0 {
                        self.reader.consume(consumable);
                        stepper.consume(consumable);
                        self.total_consumed += consumable;
                        self.validated -= consumable;
                    }
                    self.resume_at = resume_at - consumable;
                }
            }
        }

        // loop exited with unvalidated tail bytes, so they are genuinely invalid
        if self.reader.buffer().len() > self.validated {
            return Err(crate::Error::Utf8 {
                position: crate::error::Position {
                    byte_offset: self.total_consumed + self.validated,
                },
            });
        }

        // EOF – flush any partial record
        let ctx = StepCtx {
            scan_start: self.resume_at,
            scan_end: self.validated,
            remaining_in_chunk: self.reader.remaining_in_chunk(),
            total_consumed: self.total_consumed,
            buf: self.reader.buffer_mut(),
        };

        // finalize
        let final_lre = stepper.finalize(ctx, &mut emit)?;

        Ok(final_lre.or(last_record_end))
    }
}
