//! `ast::Program` → [`VirtualTape`]: one post-order walk emitting instructions in
//! exactly the order the tree-walker evaluated nodes, so that a fault lands
//! on the same span and a fold accumulates in the same order.
//!
//! Register classes are decided here and packed by [`super::regalloc`]:
//! literals become constant registers, `LocalSlot`s become local registers
//! one-to-one, and everything else is a temporary.

use crate::ast;
use crate::ast::{BinaryOp, Block, CompareOp, Expr, Kind, Program};
use crate::diagnostics::Span;

use super::tape::{Accumulate, Constants, Instruction, VirtualRegister, VirtualTape};

/// Lowers `program` against a schema of `row_len` variables, with
/// `global_positions[GlobalId]` giving each symbol's row — to the virtual
/// tape, which [`VirtualTape::allocate`] turns into the one the executors run
/// and `differentiate` turns into its gradient's.
pub(crate) fn lower(program: &Program, global_positions: &[u32], row_len: usize) -> VirtualTape {
    let mut lowerer = Lowerer {
        global_positions,
        row_len,
        consts: Constants::default(),
        next_temp: 0,
        insns: Vec::new(),
        spans: Vec::new(),
        loads: Vec::new(),
        assigned: vec![false; program.frame_size as usize],
    };
    let result = lowerer.block(&program.body, None);
    lowerer.finish(result, program.frame_size)
}

struct Lowerer<'a> {
    global_positions: &'a [u32],
    row_len: usize,
    consts: Constants,
    next_temp: u32,
    insns: Vec<Instruction<VirtualRegister>>,
    spans: Vec<Span>,
    /// Row position → register already holding it. A non-finite input faults
    /// at its *first* read, which is the cached `Load`'s span; later reads of
    /// the same variable need no instruction.
    loads: Vec<(u32, VirtualRegister)>,
    /// Which local slots are definitely written by the time they are read.
    /// A read the lowerer cannot prove gets a `Check`, preserving the walker's
    /// NaN-sentinel semantics for a front-end slot bug.
    assigned: Vec<bool>,
}

impl Lowerer<'_> {
    fn emit(&mut self, insn: Instruction<VirtualRegister>, span: Span) {
        self.insns.push(insn);
        self.spans.push(span);
    }

    fn temp(&mut self) -> VirtualRegister {
        let t = VirtualRegister::Temp(self.next_temp);
        self.next_temp += 1;
        t
    }

    fn constant(&mut self, value: f64) -> VirtualRegister {
        self.consts.intern(value)
    }

    /// Delivers a value that already lives in `value` to `dst` if one was
    /// asked for. Only a leaf ever needs this: an instruction-producing node
    /// writes `dst` directly.
    fn place(
        &mut self,
        value: VirtualRegister,
        dst: Option<VirtualRegister>,
        span: Span,
    ) -> VirtualRegister {
        match dst {
            Some(dst) if dst != value => {
                self.emit(Instruction::Copy { dst, src: value }, span);
                dst
            }
            _ => value,
        }
    }

    fn load(&mut self, input: u32, span: Span) -> VirtualRegister {
        if let Some(&(_, reg)) = self.loads.iter().find(|(i, _)| *i == input) {
            return reg;
        }
        let dst = self.temp();
        self.emit(Instruction::Load { dst, input }, span);
        self.loads.push((input, dst));
        dst
    }

    fn local(slot: crate::ast::LocalSlot) -> VirtualRegister {
        VirtualRegister::Local(u16::try_from(slot.index()).expect("fewer than 65536 locals"))
    }

    fn block(&mut self, block: &Block, dst: Option<VirtualRegister>) -> VirtualRegister {
        for assignment in &block.assignments {
            let slot = Self::local(assignment.slot);
            self.expr(&assignment.value, Some(slot));
            self.assigned[assignment.slot.index()] = true;
        }
        self.expr(&block.result, dst)
    }

    /// Lowers `expr`, into `dst` if given, and returns the register holding
    /// the value.
    fn expr(&mut self, expr: &Expr, dst: Option<VirtualRegister>) -> VirtualRegister {
        let span = expr.span;
        match &expr.kind {
            Kind::Literal(value) => {
                let c = self.constant(*value);
                self.place(c, dst, span)
            }
            Kind::Global(id) => {
                let reg = self.load(self.global_positions[id.index()], span);
                self.place(reg, dst, span)
            }
            Kind::Local(slot) => {
                let reg = Self::local(*slot);
                if !self.assigned[slot.index()] {
                    self.emit(Instruction::Check { reg }, span);
                }
                self.place(reg, dst, span)
            }
            Kind::DynamicIndex(subscript) => {
                // A literal subscript that names a real row is a plain load,
                // so `sum(1, 200, i -> var[i]…)` does not spend two hundred
                // constant registers on its indices. An invalid literal keeps
                // the gather, so it still faults at run time as the walker did.
                if let Kind::Literal(value) = subscript.kind
                    && let Some(index) = ast::to_index(value)
                    && index >= 1
                    && usize::try_from(index - 1).is_ok_and(|p| p < self.row_len)
                {
                    let position = u32::try_from(index - 1).expect("checked against row_len");
                    let reg = self.load(position, span);
                    return self.place(reg, dst, span);
                }
                let index = self.expr(subscript, None);
                let dst = dst.unwrap_or_else(|| self.temp());
                self.emit(
                    Instruction::Gather {
                        dst,
                        index,
                        subscript: subscript.span,
                    },
                    span,
                );
                dst
            }
            Kind::Unary { op, arg } => {
                let a = self.expr(arg, None);
                let dst = dst.unwrap_or_else(|| self.temp());
                self.emit(Instruction::Unary { dst, op: *op, a }, span);
                dst
            }
            Kind::Binary { op, lhs, rhs } => {
                // A whole power is the multiplication chain a fold of `n`
                // copies of the base would be, with the base lowered once.
                // `powf` is a per-lane libm call with no vector form and the
                // shader's `pow` is NaN for a negative base; a multiply is one
                // instruction on both and sign-safe. A real or variable
                // exponent stays `Binary { Pow }` and goes through `powf`.
                if *op == BinaryOp::Pow
                    && let Some(n) = rhs.whole_exponent()
                {
                    // `x^0` is one for every base, including one that would
                    // fault, which is what `powf` answers: the base is not
                    // lowered at all.
                    if n == 0 {
                        let one = self.constant(1.0);
                        return self.place(one, dst, span);
                    }
                    let count = usize::try_from(n.unsigned_abs()).expect("bounded by POWER_LIMIT");
                    let base = self.expr(lhs, None);
                    if n > 0 {
                        return self.chain(Accumulate::Prod, count, |_, _| base, dst, span);
                    }
                    // A negative exponent is the reciprocal of the positive
                    // power: a division, with a division's fault at zero.
                    let product = self.chain(Accumulate::Prod, count, |_, _| base, None, span);
                    let one = self.constant(1.0);
                    let dst = dst.unwrap_or_else(|| self.temp());
                    self.emit(
                        Instruction::Binary {
                            dst,
                            op: BinaryOp::Div,
                            a: one,
                            b: product,
                        },
                        span,
                    );
                    return dst;
                }
                let a = self.expr(lhs, None);
                let b = self.expr(rhs, None);
                let dst = dst.unwrap_or_else(|| self.temp());
                self.emit(Instruction::Binary { dst, op: *op, a, b }, span);
                dst
            }
            Kind::Compare { op, lhs, rhs } => {
                let a = self.expr(lhs, None);
                let b = self.expr(rhs, None);
                let dst = dst.unwrap_or_else(|| self.temp());
                self.emit(Instruction::Compare { dst, op: *op, a, b }, span);
                dst
            }
            // `a == b +/- t` is the conjunction it has always meant:
            // `a >= b - t` and `a <= b + t`, worst residual winning. It used to
            // be one fused instruction, which cost every executor an arm — the
            // tree-walker, the SIMD kernel, the tiled path and WGSL — to say
            // what four instructions the backends already had say between them.
            //
            // The arithmetic is unchanged, node for node and in the same order:
            // `Compare::Gte` is `right - left` and `Compare::Lte` is
            // `left - right`, so this computes `max((b - t) - a, a - (b + t))`
            // exactly as the fused version did. `corpus.rs` pins that by value.
            Kind::NearEq {
                lhs,
                rhs,
                tolerance,
            } => {
                let a = self.expr(lhs, None);
                let b = self.expr(rhs, None);
                let tolerance = self.constant(*tolerance);

                let floor = self.temp();
                self.emit(
                    Instruction::Binary {
                        dst: floor,
                        op: BinaryOp::Sub,
                        a: b,
                        b: tolerance,
                    },
                    span,
                );
                let ceiling = self.temp();
                self.emit(
                    Instruction::Binary {
                        dst: ceiling,
                        op: BinaryOp::Add,
                        a: b,
                        b: tolerance,
                    },
                    span,
                );

                let at_least = self.temp();
                self.emit(
                    Instruction::Compare {
                        dst: at_least,
                        op: CompareOp::Gte,
                        a,
                        b: floor,
                    },
                    span,
                );
                let at_most = self.temp();
                self.emit(
                    Instruction::Compare {
                        dst: at_most,
                        op: CompareOp::Lte,
                        a,
                        b: ceiling,
                    },
                    span,
                );

                let dst = dst.unwrap_or_else(|| self.temp());
                self.emit(
                    Instruction::Combine {
                        dst,
                        how: Accumulate::Worst,
                        a: at_least,
                        b: at_most,
                        last: true,
                    },
                    span,
                );
                dst
            }
            Kind::Fold { kind, terms } => {
                self.fold(Accumulate::from_aggregate(*kind), terms, dst, span)
            }
            Kind::And { terms } => self.fold(Accumulate::Worst, terms, dst, span),
            Kind::Block(block) => self.block(block, dst),
            Kind::Aggregate { .. } => {
                unreachable!("`unroll_aggregates` turns every aggregate into a `Fold` or fails")
            }
        }
    }

    fn fold(
        &mut self,
        how: Accumulate,
        terms: &[Expr],
        dst: Option<VirtualRegister>,
        span: Span,
    ) -> VirtualRegister {
        self.chain(
            how,
            terms.len(),
            |this, k| this.expr(&terms[k], None),
            dst,
            span,
        )
    }

    /// Left to right from the identity, one `Combine` per term, checked only
    /// on the last — the walker checked the fold's value, not each step.
    ///
    /// `term` lowers the `k`th operand when the chain reaches it rather than
    /// beforehand: lowering them all first would keep every operand's register
    /// live across the whole chain, and a two-hundred-term sum would want two
    /// hundred registers.
    fn chain(
        &mut self,
        how: Accumulate,
        count: usize,
        mut term: impl FnMut(&mut Self, usize) -> VirtualRegister,
        dst: Option<VirtualRegister>,
        span: Span,
    ) -> VirtualRegister {
        let identity = self.constant(how.identity());
        if count == 0 {
            return self.place(identity, dst, span);
        }
        let acc = dst.unwrap_or_else(|| self.temp());
        let last = count - 1;
        for k in 0..count {
            let b = term(self, k);
            let a = if k == 0 { identity } else { acc };
            self.emit(
                Instruction::Combine {
                    dst: acc,
                    how,
                    a,
                    b,
                    last: k == last,
                },
                span,
            );
        }
        acc
    }

    fn finish(self, result: VirtualRegister, frame_size: u32) -> VirtualTape {
        VirtualTape::new(
            self.consts,
            u16::try_from(frame_size).expect("fewer than 65536 locals"),
            self.next_temp,
            self.insns,
            self.spans,
            result,
        )
    }
}

#[cfg(test)]
mod tests {
    //! Lowering: the instruction sequence a source produces.

    use crate::ast::{BinaryOp, Block, CompareOp, Expr, Kind, LocalSlot, Program, UnaryOp};
    use crate::diagnostics::Span;

    use super::super::tape::{Accumulate, AllocatedTape, Instruction, Register};
    use super::super::tape_for;

    const R: fn(u16) -> Register = Register;

    fn loads(tape: &AllocatedTape, input: u32) -> usize {
        tape.insns
            .iter()
            .filter(|i| matches!(i, Instruction::Load { input: x, .. } if *x == input))
            .count()
    }

    #[test]
    fn a_sum_of_two_globals_lowers_to_two_loads_and_an_add() {
        let tape = tape_for("x1 + x2", &["x1", "x2"]);
        assert_eq!(
            tape.insns,
            vec![
                Instruction::Load {
                    dst: R(0),
                    input: 0
                },
                Instruction::Load {
                    dst: R(1),
                    input: 1
                },
                Instruction::Binary {
                    dst: R(2),
                    op: BinaryOp::Add,
                    a: R(0),
                    b: R(1)
                },
            ]
        );
        assert_eq!(tape.result, R(2));
    }

    #[test]
    fn a_repeated_global_is_loaded_once() {
        let tape = tape_for("x1 * x1", &["x1"]);
        assert_eq!(loads(&tape, 0), 1);
        assert_eq!(
            tape.insns[1],
            Instruction::Binary {
                dst: R(1),
                op: BinaryOp::Mul,
                a: R(0),
                b: R(0)
            }
        );
    }

    #[test]
    fn a_literal_is_a_constant_register_and_no_instruction() {
        let tape = tape_for("3 + 4", &[]);
        assert!(tape.insns.is_empty(), "{:?}", tape.insns);
        assert_eq!(tape.consts, vec![7.0]);
        assert_eq!(tape.result, R(0));

        let tape = tape_for("x1 + 2", &["x1"]);
        assert_eq!(tape.consts, vec![2.0]);
        // Constants come first, so the load lands in register 1.
        assert_eq!(
            tape.insns[0],
            Instruction::Load {
                dst: R(1),
                input: 0
            }
        );
    }

    #[test]
    fn equal_constants_share_a_register_but_signed_zeros_do_not() {
        let tape = tape_for("x1 + 2 + 2", &["x1"]);
        assert_eq!(tape.consts, vec![2.0]);

        let tape = tape_for("x1 + 0.0 - -0.0", &["x1"]);
        assert_eq!(tape.consts.len(), 2, "{:?}", tape.consts);
        assert_ne!(tape.consts[0].to_bits(), tape.consts[1].to_bits());
    }

    #[test]
    fn a_valid_literal_subscript_is_a_plain_load() {
        let tape = tape_for("var[2] + x2", &["x1", "x2", "x3"]);
        assert!(
            !tape
                .insns
                .iter()
                .any(|i| matches!(i, Instruction::Gather { .. }))
        );
        // `var[2]` and `x2` are the same row, so one load serves both.
        assert_eq!(loads(&tape, 1), 1);
        assert_eq!(tape.consts, Vec::<f64>::new());
    }

    #[test]
    fn an_invalid_literal_subscript_keeps_a_gather() {
        let tape = tape_for("var[0]", &["x1"]);
        assert_eq!(tape.consts, vec![0.0]);
        assert_eq!(
            tape.insns,
            vec![Instruction::Gather {
                dst: R(1),
                index: R(0),
                subscript: Span::new(4, 5)
            }]
        );
    }

    #[test]
    fn a_strict_comparison_is_one_compare_instruction() {
        let tape = tape_for("x1 < x2", &["x1", "x2"]);
        assert_eq!(
            tape.insns[2],
            Instruction::Compare {
                dst: R(2),
                op: CompareOp::Lt,
                a: R(0),
                b: R(1)
            }
        );
        // The nudge is inside the instruction, not a constant.
        assert!(tape.consts.is_empty());
    }

    /// `a == b +/- t` lowers to the conjunction it means, not to an
    /// instruction of its own: bracket `b` by the tolerance, compare `a`
    /// against each end, and take the worse residual.
    ///
    /// Five instructions where there was one. That is the trade — every
    /// executor loses an arm, and `corpus.rs` pins that the arithmetic is
    /// unchanged to the bit.
    #[test]
    fn a_near_equality_lowers_to_two_comparisons_and_a_conjunction() {
        let tape = tape_for("x1 == x2 +/- 0.5", &["x1", "x2"]);
        assert_eq!(tape.consts, vec![0.5]);

        let ops: Vec<_> = tape.insns.iter().skip(2).collect();
        assert!(
            matches!(
                ops.as_slice(),
                [
                    Instruction::Binary {
                        op: crate::ast::BinaryOp::Sub,
                        ..
                    },
                    Instruction::Binary {
                        op: crate::ast::BinaryOp::Add,
                        ..
                    },
                    Instruction::Compare {
                        op: crate::ast::CompareOp::Gte,
                        ..
                    },
                    Instruction::Compare {
                        op: crate::ast::CompareOp::Lte,
                        ..
                    },
                    Instruction::Combine {
                        how: Accumulate::Worst,
                        last: true,
                        ..
                    },
                ]
            ),
            "{ops:?}"
        );
    }

    #[test]
    fn a_fold_seeds_from_the_identity_and_checks_only_its_last_step() {
        let tape = tape_for("sum(1, 3, i -> x1 * i)", &["x1"]);
        let identity = tape
            .consts
            .iter()
            .position(|c| c.to_bits() == 0.0f64.to_bits())
            .expect("the identity is a constant");
        let identity = R(u16::try_from(identity).unwrap());
        let combines: Vec<(Register, Register, bool)> = tape
            .insns
            .iter()
            .filter_map(|i| match *i {
                Instruction::Combine {
                    dst,
                    how: Accumulate::Sum,
                    a,
                    last,
                    ..
                } => Some((dst, a, last)),
                _ => None,
            })
            .collect();
        assert_eq!(combines.len(), 3);
        let acc = combines[0].0;
        assert_eq!(combines[0].1, identity, "the first step reads the identity");
        assert!(combines[1..].iter().all(|(d, a, _)| *d == acc && *a == acc));
        assert_eq!(
            combines.iter().map(|c| c.2).collect::<Vec<_>>(),
            [false, false, true]
        );
    }

    /// A whole power is the fold's multiplication chain with the base lowered
    /// once: `(x1 + 1)^3` adds once and multiplies three times.
    #[test]
    fn a_whole_power_lowers_its_base_once() {
        let tape = tape_for("(x1 + 1)^3", &["x1"]);
        let adds = tape
            .insns
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    Instruction::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                )
            })
            .count();
        let multiplies = tape
            .insns
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    Instruction::Combine {
                        how: Accumulate::Prod,
                        ..
                    }
                )
            })
            .count();
        assert_eq!((adds, multiplies), (1, 3), "{:?}", tape.insns);
        assert!(
            !tape.insns.iter().any(|i| matches!(
                i,
                Instruction::Binary {
                    op: BinaryOp::Pow,
                    ..
                }
            )),
            "a whole power reached the tape as `powf`: {:?}",
            tape.insns
        );
    }

    #[test]
    fn a_zeroth_power_is_one_and_never_reads_its_base() {
        let tape = tape_for("x1^0", &["x1"]);
        assert!(tape.insns.is_empty(), "{:?}", tape.insns);
        assert_eq!(tape.consts, vec![1.0]);
    }

    #[test]
    fn a_negative_power_is_one_over_the_positive_one() {
        let tape = tape_for("x1^-2", &["x1"]);
        assert!(
            matches!(
                tape.insns.last(),
                Some(Instruction::Binary {
                    op: BinaryOp::Div,
                    ..
                })
            ),
            "{:?}",
            tape.insns
        );
    }

    #[test]
    fn a_real_or_oversized_exponent_stays_a_power() {
        for source in ["x1^2.5", "x1^65", "x1^x2"] {
            let tape = tape_for(source, &["x1", "x2"]);
            assert!(
                tape.insns.iter().any(|i| matches!(
                    i,
                    Instruction::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                )),
                "{source}: {:?}",
                tape.insns
            );
        }
    }

    /// Values, the half that could go wrong quietly. `powf` and repeated
    /// multiplication differ in the last place for about one case in five at
    /// `n >= 3`, so the assertion is against the multiplication, bit for bit.
    #[test]
    fn a_whole_power_multiplies_rather_than_calling_powf() {
        let x = 2.3_f64;
        let at =
            |source: &str| super::super::eval_one(source, &[("x1", x)]).expect("should evaluate");
        assert_eq!(at("x1^3"), x * x * x);
        // The case that motivates saying so out loud.
        assert_ne!(x * x * x, x.powf(3.0));
        assert_eq!(at("x1^-2"), 1.0 / (x * x));
        // Squaring is where libm and multiplication agree, so this one can be
        // asserted both ways.
        assert_eq!(at("x1^2"), x.powf(2.0));
        assert_eq!(at("x1^0"), 1.0);
    }

    #[test]
    fn an_and_seeds_from_negative_infinity() {
        // `invert_monotone` turns this into `x1 < e^2 and x1 > 0`.
        let tape = tape_for("ln(x1) < 2", &["x1"]);
        assert!(tape.consts.contains(&f64::NEG_INFINITY));
        assert!(tape.insns.iter().any(|i| matches!(
            i,
            Instruction::Combine {
                how: Accumulate::Worst,
                ..
            }
        )));
    }

    #[test]
    fn a_local_is_written_in_place_and_read_without_an_instruction() {
        let tape = tape_for("var x = x1 * 2;\nx + x", &["x1"]);
        let local = R(tape.first_local());
        assert_eq!(tape.locals, 1);
        assert!(tape.insns.iter().any(|i| matches!(
            i,
            Instruction::Binary {
                dst,
                op: BinaryOp::Mul,
                ..
            } if *dst == local
        )));
        assert!(
            !tape
                .insns
                .iter()
                .any(|i| matches!(i, Instruction::Copy { .. }))
        );
        assert!(
            !tape
                .insns
                .iter()
                .any(|i| matches!(i, Instruction::Check { .. }))
        );
    }

    #[test]
    fn a_leaf_assignment_becomes_a_copy() {
        let tape = tape_for("var x = x1;\nx + 1", &["x1"]);
        let local = R(tape.first_local());
        assert!(
            tape.insns
                .iter()
                .any(|i| matches!(i, Instruction::Copy { dst, .. } if *dst == local))
        );
    }

    /// An aggregate reaches the tape as a flat fold: three terms, three combines,
    /// and nothing that jumps.
    #[test]
    fn an_aggregate_is_a_flat_fold() {
        let tape = tape_for("sum(1, 3, i -> i * x1)", &["x1"]);
        let combines = tape
            .insns
            .iter()
            .filter(|i| matches!(i, Instruction::Combine { .. }))
            .count();
        assert_eq!(combines, 3);
    }

    #[test]
    fn every_checked_instruction_carries_the_span_of_its_node() {
        let tape = tape_for("sin(x1) + 2", &["x1"]);
        assert_eq!(
            tape.spans,
            vec![Span::new(4, 6), Span::new(0, 7), Span::new(0, 11)]
        );
        assert!(matches!(
            tape.insns[1],
            Instruction::Unary {
                op: UnaryOp::Sin,
                ..
            }
        ));
    }

    /// A front-end slot bug is the only way to reach this; the tape still reports
    /// it as the walker did rather than reading a stale register.
    #[test]
    fn an_unassigned_local_read_gets_a_check_at_the_local() {
        let program = Program {
            body: Block {
                assignments: Vec::new(),
                result: Expr::new(Kind::Local(LocalSlot::from_index(0)), Span::new(0, 1)),
            },
            frame_size: 1,
        };
        let tape = super::lower(&program, &[], 0).allocate();
        assert_eq!(tape.insns, vec![Instruction::Check { reg: R(0) }]);
        assert_eq!(tape.spans, vec![Span::new(0, 1)]);
        assert_eq!(tape.result, R(0));
    }
}
