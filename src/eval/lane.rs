//! The per-lane executor: one sample, scalar registers.
//!
//! This is what `eval_row` runs, for the one consumer that cannot batch — the
//! walker judges one candidate before it can propose the next. It is not a
//! second implementation of the language: the operators are the AST's own
//! `apply`, and the instruction semantics are shared with the batched
//! executor, which runs the same tape a tile at a time.

use crate::ast;
use crate::diagnostics::Fault;

use super::EPSILON;
use super::tape::{AllocatedTape, FaultKind, Instruction, LaneFault};

/// The single place a `var[i]` subscript becomes a row position.
///
/// One-based, so `var[0]` lands on `-1` and the one range check covers zero
/// and negatives as well as overrun.
pub(crate) fn resolve_index(value: f64, available: usize) -> Result<usize, FaultKind> {
    let requested_1index = ast::to_index(value).ok_or(FaultKind::NotAnInteger(value))?;
    usize::try_from(requested_1index - 1)
        .ok()
        .filter(|position| *position < available)
        .ok_or(FaultKind::OutOfBounds {
            requested_1index,
            available,
        })
}

/// Evaluates `tape` for one row. `frame` must be [`AllocatedTape::prime`]d.
pub(crate) fn run_lane(tape: &AllocatedTape, row: &[f64], frame: &mut [f64]) -> Result<f64, Fault> {
    let fault = |pc: usize, kind: FaultKind| {
        tape.fault(LaneFault {
            insn: u32::try_from(pc).expect("fewer than 2^32 instructions"),
            kind,
        })
    };
    let checked = |pc: usize, value: f64| {
        if value.is_finite() {
            Ok(value)
        } else {
            Err(fault(pc, FaultKind::NonFinite(value)))
        }
    };

    for (pc, insn) in tape.insns.iter().enumerate() {
        match *insn {
            Instruction::Load { dst, input } => {
                frame[dst.index()] = checked(pc, row[input as usize])?;
            }
            Instruction::Copy { dst, src } => frame[dst.index()] = frame[src.index()],
            Instruction::Unary { dst, op, a } => {
                frame[dst.index()] = checked(pc, op.apply(frame[a.index()]))?;
            }
            Instruction::Binary { dst, op, a, b } => {
                frame[dst.index()] = checked(pc, op.apply(frame[a.index()], frame[b.index()]))?;
            }
            Instruction::Compare { dst, op, a, b } => {
                frame[dst.index()] = checked(pc, compare(op, frame[a.index()], frame[b.index()]))?;
            }
            Instruction::Combine {
                dst,
                how,
                a,
                b,
                last,
            } => {
                let value = how.apply(frame[a.index()], frame[b.index()]);
                frame[dst.index()] = if last { checked(pc, value)? } else { value };
            }
            Instruction::Check { reg } => {
                checked(pc, frame[reg.index()])?;
            }
            Instruction::Gather { dst, index, .. } => {
                let position =
                    resolve_index(frame[index.index()], row.len()).map_err(|k| fault(pc, k))?;
                frame[dst.index()] = checked(pc, row[position])?;
            }
        }
    }
    Ok(frame[tape.result.index()])
}

/// The residual of `a op b` under the `<= 0` convention — four expressions,
/// and the tile kernel computes the same four in the same order so the nudge
/// lands in the same place.
pub(crate) fn compare(op: crate::ast::CompareOp, left: f64, right: f64) -> f64 {
    use crate::ast::CompareOp;
    match op {
        CompareOp::Lte => left - right,
        CompareOp::Gte => right - left,
        CompareOp::Lt => (left - right) + EPSILON,
        CompareOp::Gt => (right - left) + EPSILON,
    }
}
