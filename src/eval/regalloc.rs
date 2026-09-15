//! Packs temporaries into physical registers by linear scan.
//!
//! The code is straight-line — the front end unrolls every aggregate — so a
//! temporary's live interval is simply its first definition to its last use,
//! and a free-list scan is optimal enough: the register file is
//! `registers × tile` doubles, and keeping it inside L1 is what matters.
//!
//! Constants and locals are pinned and never reused — constants because the
//! file is primed with them once, locals because an unwritten local must read
//! the NaN it was primed with, and a reused temporary could hold anything.

use super::tape::{Instruction, Register, VirtualRegister};

/// Assigns physical registers. Returns the allocated instructions, the
/// physical registers of `outputs` in order, and the total register count.
///
/// `outputs` are the registers the caller reads after the run — the result,
/// and a differentiated tape's partials — so they live to the end.
pub(crate) fn allocate(
    insns: Vec<Instruction<VirtualRegister>>,
    outputs: &[VirtualRegister],
    consts: u16,
    locals: u16,
) -> (Vec<Instruction<Register>>, Vec<Register>, u16) {
    let temps = insns
        .iter()
        .flat_map(|insn| insn.dst().into_iter().chain(insn.sources()))
        .chain(outputs.iter().copied())
        .filter_map(|reg| match reg {
            VirtualRegister::Temp(t) => Some(t as usize + 1),
            _ => None,
        })
        .max()
        .unwrap_or(0);

    // First definition and last mention of every temporary. An output is
    // mentioned by nobody and read by the caller, so it lives to the end.
    let mut def = vec![usize::MAX; temps];
    let mut last = vec![0usize; temps];
    for (i, insn) in insns.iter().enumerate() {
        if let Some(VirtualRegister::Temp(t)) = insn.dst() {
            let t = t as usize;
            def[t] = def[t].min(i);
            last[t] = i;
        }
        for source in insn.sources() {
            if let VirtualRegister::Temp(t) = source {
                last[t as usize] = i;
            }
        }
    }
    for output in outputs {
        if let VirtualRegister::Temp(t) = output {
            last[*t as usize] = insns.len();
        }
    }

    // "Linear-scan register allocation" (Poletto & Sarkar 1999): one pass in
    // instruction order, a free list of physical slots, each temporary taking
    // one at its first write and returning it after its last read. The
    // destination is allocated *before* this instruction's
    // dying operands are released, so it can never alias one of them — which
    // the batched executor's three-way slice split relies on. `Combine` writes
    // its own accumulator, which is the same virtual register on both sides
    // and therefore the same physical one; that is the one aliasing wanted.
    let mut slot = vec![u16::MAX; temps];
    let mut free: Vec<u16> = Vec::new();
    let mut peak: u16 = 0;
    let mut allocated: Vec<Instruction<Register>> = Vec::with_capacity(insns.len());

    let base = consts + locals;
    let physical = |reg: VirtualRegister, slot: &[u16]| -> Register {
        match reg {
            VirtualRegister::Const(c) => Register(c),
            VirtualRegister::Local(l) => Register(consts + l),
            VirtualRegister::Temp(t) => {
                let s = slot[t as usize];
                debug_assert_ne!(s, u16::MAX, "temporary {t} used before definition");
                Register(base + s)
            }
        }
    };

    for (i, insn) in insns.into_iter().enumerate() {
        if let Some(VirtualRegister::Temp(t)) = insn.dst()
            && slot[t as usize] == u16::MAX
        {
            let s = free.pop().unwrap_or_else(|| {
                let s = peak;
                peak += 1;
                s
            });
            slot[t as usize] = s;
        }

        allocated.push(insn.map(|reg| physical(reg, &slot)));

        for reg in insn.dst().into_iter().chain(insn.sources()) {
            if let VirtualRegister::Temp(t) = reg
                && last[t as usize] == i
                && slot[t as usize] != u16::MAX
            {
                free.push(slot[t as usize]);
                // Released once even if the register appears twice here.
                last[t as usize] = usize::MAX;
            }
        }
    }

    let outputs = outputs
        .iter()
        .map(|output| physical(*output, &slot))
        .collect();
    (allocated, outputs, base + peak)
}

#[cfg(test)]
mod tests {
    //! Allocation: how many registers a tape needs, and which it must not share.

    use super::super::tape::Instruction;
    use super::super::tape_for;

    #[test]
    fn a_dead_temporary_frees_its_register() {
        // Two loads, two intermediates, one result: the loads die at the second
        // intermediate and the result takes a freed register.
        let tape = tape_for("(x1 + x2) + (x1 - x2)", &["x1", "x2"]);
        assert_eq!(tape.registers, 4, "{:?}", tape.insns);
    }

    #[test]
    fn an_eight_term_addition_chain_needs_at_most_three_temporaries() {
        let names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"];
        let tape = tape_for("x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8", &names);
        assert!(
            tape.registers <= 3,
            "{} registers: {:?}",
            tape.registers,
            tape.insns
        );
    }

    #[test]
    fn the_two_hundred_term_fold_needs_fewer_than_eight_registers() {
        let names: Vec<String> = (1..=200).map(|i| format!("x{i}")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let tape = tape_for("sum(1, 200, i -> var[i]^2 - 3.0)", &names);
        let temps = tape.registers - tape.first_local() - tape.locals;
        assert!(temps < 8, "{temps} temporaries");
        assert!(tape.consts.len() <= 4, "{:?}", tape.consts);
        // Two hundred loads, and none of the subscripts became a constant.
        assert_eq!(
            tape.insns
                .iter()
                .filter(|i| matches!(i, Instruction::Load { .. }))
                .count(),
            200
        );
    }

    #[test]
    fn a_binary_destination_never_aliases_an_operand() {
        for (source, names) in [
            ("x1 * x1 + x1", vec!["x1"]),
            ("(x1 + x2) * (x1 + x2)", vec!["x1", "x2"]),
            ("max(x1, x1 - x2) < x2 * x2", vec!["x1", "x2"]),
            ("x1 == x2 * x2 +/- 0.1", vec!["x1", "x2"]),
        ] {
            let tape = tape_for(source, &names);
            for insn in &tape.insns {
                if let Some(dst) = insn.dst()
                    && !matches!(insn, Instruction::Combine { .. })
                {
                    assert!(
                        !insn.sources().contains(&dst),
                        "{source:?}: {insn:?} writes an operand"
                    );
                }
            }
        }
    }

    #[test]
    fn constants_and_locals_are_never_reused() {
        let tape = tape_for("var x = x1 * 2;\nvar y = x + 3;\ny * y - x", &["x1"]);
        assert_eq!(tape.consts, vec![2.0, 3.0]);
        assert_eq!(tape.locals, 2);
        for insn in &tape.insns {
            if let Some(dst) = insn.dst() {
                assert!(
                    dst.0 >= tape.first_local(),
                    "{insn:?} writes a constant register"
                );
            }
        }
        // The locals are written exactly by their assignments, never by a
        // temporary's allocation: only two instructions target local registers.
        let first_temp = tape.first_local() + tape.locals;
        let writes_to_locals = tape
            .insns
            .iter()
            .filter_map(Instruction::dst)
            .filter(|dst| dst.0 >= tape.first_local() && dst.0 < first_temp)
            .count();
        assert_eq!(writes_to_locals, 2);
    }
}
