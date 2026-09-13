//! What an equality lets us conclude about the variables it mentions.
//!
//! Babel admits one equality form, `a == b +/- t`, and it covers at least six
//! structurally different problems — the taxonomy is in `docs/todo.md`.
//! Downstream, all six look identical: one residual, satisfied when `<= 0`. This
//! module is what tells them apart.
//!
//! # What it is for
//!
//! Not labelling constraints for their own sake. `tests/cvg_equalities.rs`
//! measured the actual defect and it is singular: on a tight equality the pool
//! delivers two hundred *distinct* feasible points spanning `0.0000%` of the
//! free coordinates' range. It finds the feasible set and cannot move along it.
//!
//! So the useful output is not a label, it is a **split**: which coordinates the
//! walker moves, and which it computes from them. Move `x`, evaluate
//! `y = sin(x)`, and a point that was on the curve stays on it — where a chord
//! drawn through both coordinates leaves it almost immediately.
//!
//! # Why this is `cvg`'s and not the front end's
//!
//! `eval` wants a number and has no use for any of this. The front end must not
//! know it either: a rewrite that dropped `y` from the problem would be lying to
//! the evaluator, which still has to compute the residual of the whole
//! constraint. This is a *reading* of the AST, and it changes nothing.
//!
//! # What is deliberately not here
//!
//! Rows C (`x2 == x1 + x2/2 - x3/x4`, driven after gathering linear terms), D
//! (`abs(x1) == 1`, two branches) and E (an under-determined system, where which
//! variables get driven is a choice) classify as [`Shape::Opaque`]. They are
//! named in the taxonomy and not yet analysed, and saying so is better than
//! guessing at them.
//!
//! Row F — `x == sin(x)`, the variable inside and outside a function no solver
//! will take — is [`Shape::Implicit`], and is refused by
//! [`ConstraintSystem::new`](crate::ConstraintSystem::new) rather than
//! searched for. No use case has turned up for `x == f(x)` and Newton is a
//! great deal of machinery to carry for a shape nobody writes. The line is
//! narrower than "self-referential", which would also reject row C and
//! `x^2 == x + 2`; see [`Shape::Implicit`].

use std::collections::{BTreeMap, BTreeSet};

use rand::RngExt;
use rand::rngs::Xoshiro256PlusPlus;

use crate::ast::{Expr, GlobalId, Kind, Program};
use crate::cvg::interval;
use crate::{Ast, ConstraintSystem, Point, Schema};

/// What one equality lets us conclude.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Shape {
    /// `x1 == pi +/- t`. One variable equals an expression naming no variable at
    /// all, so the dimension is gone: there is nothing to search. Row A.
    ///
    /// Reported separately from [`Driven`](Shape::Driven), which it is a special
    /// case of, because "constant" is the stronger statement and a caller may
    /// want it — a pinned coordinate need not even be re-evaluated per move.
    Pinned { variable: GlobalId },
    /// `y == sin(x) +/- t`. `variable` stands alone on one side and appears
    /// nowhere on the other, so it is *computed* rather than searched. Row B.
    ///
    /// This needs no inverse. That is the whole reason it is the row worth doing
    /// first: nothing has to solve `sin` for anything, because the variable is
    /// already isolated and the other side is a formula for it.
    Driven { variable: GlobalId },
    /// `x == sin(x)`, `x2 == x1 + x2/2 - x3/x4`, `x == x*x + 2`. A variable
    /// defined in terms of itself, so the equality is *implicit* in it: there is
    /// no rearrangement-free way to write `v = ...`. Refused by
    /// [`SystemError::Implicit`](crate::SystemError::Implicit).
    ///
    /// "Implicit" rather than "cyclic": a cycle is a mutual dependency *between*
    /// equations, which [`plan`] handles by driving neither. One equation that
    /// cannot be solved for the variable it names is the textbook implicit
    /// form.
    ///
    /// **What is refused is a phrasing, not a problem.** `x2 == x1 + x2/2` and
    /// `x2/2 - x1 == 0` describe the same set, and only the first asks an
    /// evaluation order to resolve `x2` from `x2`. Every constraint this rejects
    /// can be written with the variable on one side, which is what the
    /// diagnostic says.
    ///
    /// Drawing the line here rather than at "and beyond every solver" is
    /// deliberate. The narrower rule let `x2 == x1 + x2/2 - x3/x4` through
    /// because Z3 can answer it, which is true and beside the point: nothing
    /// downstream can *drive* it, so it falls to whatever the sampler manages
    /// and reads as a capability we do not have. One rule stated once beats a
    /// rule that depends on what the emitter happens to support this month.
    ///
    Implicit { variable: GlobalId },
    /// Nothing structural to say: rows C, D and E, plus anything that is not an
    /// equality at all.
    Opaque,
}

/// Which coordinates the walker moves, and which it computes from them.
///
/// Positions are into the bound [`Schema`], not into any one constraint's
/// symbol list, because the walker works in whole points.
#[derive(Debug, Clone)]
pub(crate) struct Plan {
    /// Driven coordinates, **in evaluation order**. A driven variable may be
    /// defined in terms of another driven variable, and computing them out of
    /// order reads a stale value.
    ///
    /// Positions, and nothing else. This used to carry each coordinate's
    /// defining expression and tolerance so that `retract` could evaluate
    /// `f(free) ± t`; [`interval::slice`] derives that band from the constraint
    /// itself, along with every other constraint naming the coordinate, so the
    /// only thing the walker still needs from a reading of the equalities is
    /// **which coordinates are computed and in what order**.
    driven: Vec<usize>,
    /// Positions the walker is free to move. Everything not driven.
    free: Vec<usize>,
}

impl Plan {
    pub(crate) fn free(&self) -> &[usize] {
        &self.free
    }

    pub(crate) fn driven(&self) -> &[usize] {
        &self.driven
    }
}

/// Recomputes every driven coordinate of `point` from the free ones, in place.
///
/// A point on an equality surface leaves it under almost any move, because
/// the surface has no volume — which is why a walker that moves every
/// coordinate independently is reduced to jitter around wherever it started.
/// Driving is the answer: move what is free, and compute the rest.
///
/// # Why this samples rather than evaluates
///
/// The obvious version assigns `y = f(free)` and is **wrong**, because
/// babel has no bare equality: `y == f(x) +/- t` admits the whole band, and
/// collapsing it to its centre line throws away a dimension of the feasible
/// region. On `xi == 10.75 +/- 0.2` over two hundred variables that is not
/// subtle — every coordinate pins to `10.75` and the pool returns the same
/// point two hundred times, which is exactly what it did before this
/// sampled.
///
/// So it draws uniformly from `f(free) ± t`. That is not a fudge, it is a
/// **Gibbs step**: with the other coordinates held, the feasible slice for a
/// driven variable *is* that interval, and drawing uniformly from a
/// conditional slice is the move that leaves the uniform distribution
/// invariant. Evaluating to the centre would not.
///
/// The slice comes from [`interval::slice`] rather than from the one
/// equality that defined the coordinate, so **every** constraint mentioning
/// it has a say. The band `f(free) ± t` is what that equality contributes
/// and the intersection can only be tighter, which turns draws that used to
/// land outside the other constraints and be rejected into draws that
/// cannot.
///
/// Silent about failure by design. A coordinate whose slice comes back
/// empty is left alone, and the candidate is judged exactly as an
/// unretracted one would be. **Feasibility is never assumed from a
/// successful retraction** — a wrong drive costs rejected moves, not wrong
/// points, and that is what makes this safe to apply without proving it.
///
/// Order still matters, and for the same reason: a driven coordinate may be
/// defined in terms of another, and `slice` reads the point as it stands,
/// so computing them out of order reads a stale value.
pub(crate) fn retract(system: &ConstraintSystem, point: &mut Point, rng: &mut Xoshiro256PlusPlus) {
    let Some(plan) = &system.plan else {
        return;
    };
    // A driven coordinate holds a value that is about to be replaced, so
    // narrowing against a constraint that mentions one still waiting its
    // turn conditions on a stale number. That is not merely wasteful, it
    // **destroys the freedom driving exists to exploit**: on
    // `y == sin(x) +/- t` with `z == y + 1 +/- t`, conditioning `y` on the
    // current `z` pins it within `t` of `z - 1`, and then `z` is pinned
    // within `t` of the new `y`. The pair shuffles by `t` a sweep instead
    // of travelling, and `two_coupled_equalities_are_traversed` measured it
    // as three occupied cells of eighty where twenty-four are wanted.
    //
    // So a coordinate becomes conditionable only once it has been drawn.
    // Everything free is conditionable from the start.
    let mut settled = vec![true; point.len()];
    for driven in plan.driven() {
        settled[*driven] = false;
    }

    for driven in plan.driven().iter().copied() {
        let slice = interval::slice_conditioned(system, point, driven, Some(&settled));
        if !slice.is_empty() {
            point[driven] = if slice.width() > 0.0 {
                rng.random_range(slice.lo()..=slice.hi())
            } else {
                slice.lo()
            };
        }
        settled[driven] = true;
    }
}

/// [`retract`] without the draw: every driven coordinate is clamped into its
/// slice rather than drawn from it.
///
/// The walker must draw, because it is producing a *sample* and the draw is
/// what keeps the uniform distribution invariant. A repair is producing one
/// point, near a given one, and must produce the same point every time it
/// is asked — so it moves each driven coordinate the least distance that
/// puts it inside its band, and no further. Same order, same settled mask,
/// same silence on an empty slice, for the same reasons.
pub(crate) fn settle(system: &ConstraintSystem, point: &mut Point) {
    let Some(plan) = &system.plan else {
        return;
    };
    let mut settled = vec![true; point.len()];
    for driven in plan.driven() {
        settled[*driven] = false;
    }

    for driven in plan.driven().iter().copied() {
        let slice = interval::slice_conditioned(system, point, driven, Some(&settled));
        if !slice.is_empty() {
            point[driven] = point[driven].clamp(slice.lo(), slice.hi());
        }
        settled[driven] = true;
    }
}

/// The shape of one constraint.
///
/// Infallible and total: anything it cannot read is [`Shape::Opaque`], which is
/// the same answer as "not analysed yet" and is always safe — a caller that
/// learns nothing does what it did before.
pub(crate) fn shape(constraint: &Ast) -> Shape {
    // A `var[i]` subscript reads a variable the expression never names, so the
    // static symbol list is not the whole story and a "does it appear on the
    // other side" test cannot be answered. `Ast` documents this and it is
    // exactly the trap it warns about.
    if constraint.contains_dynamic_lookup {
        return Shape::Opaque;
    }
    if !constraint.is_constraint {
        return Shape::Opaque;
    }

    // A block with local bindings could still be an equality, but its sides are
    // not reachable without substituting the assignments through, which is a
    // rewrite and not a reading.
    let Program { body, .. } = &constraint.program;
    if !body.assignments.is_empty() {
        return Shape::Opaque;
    }
    let Kind::NearEq { lhs, rhs, .. } = &body.result.kind else {
        return Shape::Opaque;
    };

    // Any variable named on both sides makes the equality implicit in it, and
    // this runs first because it does not care how either side is *shaped*.
    // Asking only about a side that is a bare variable would make the rule
    // depend on spelling: `x == x*x + 2` refused and `x*x == x + 2` allowed,
    // which are the same set.
    if let Some(&variable) = globals(lhs).intersection(&globals(rhs)).next() {
        return Shape::Implicit { variable };
    }

    // Both orders, since `y == sin(x)` and `sin(x) == y` say the same thing.
    for (candidate, other) in [(lhs, rhs), (rhs, lhs)] {
        let Kind::Global(variable) = candidate.kind else {
            continue;
        };
        debug_assert!(
            !mentions(other, variable),
            "a shared variable is settled above"
        );

        return if globals(other).is_empty() {
            Shape::Pinned { variable }
        } else {
            Shape::Driven { variable }
        };
    }

    // No side is a bare variable, so try to make one: peel the operators around
    // a variable that occurs exactly once and move them to the other side.
    // Schema order rather than discovery order, so the choice does not wander.
    let mut candidates: Vec<GlobalId> = globals(&body.result).into_iter().collect();
    candidates.sort_unstable();
    for variable in candidates {
        if occurrences(&body.result, variable) != 1 {
            continue;
        }
        let side = if mentions(lhs, variable) { lhs } else { rhs };
        if reaches(side, variable) {
            return Shape::Driven { variable };
        }
    }

    Shape::Opaque
}

/// Whether the operators between `side`'s root and `variable` can all be
/// undone, so that the equality determines the variable.
///
/// `x1 + x2 == 3` reaches `x1`: descend into the branch holding it, and every
/// node on the way must be one an inverse exists for.
///
/// # It answers whether, not what
///
/// This used to *build* the rearrangement — `3 - x2` — for `retract` to
/// evaluate. [`interval::slice`] derives the same band by narrowing the constraint
/// itself, so the expression had no consumer left and the walk down the path is
/// all that survives. The arithmetic those rules encoded now lives in
/// `interval::invert_binary`, tested there against the same cases.
///
/// # Only where the variable occurs once
///
/// The caller checks that. In term rewriting a term where a variable appears at
/// most once is *linear* in it, and that is exactly the class where isolating is
/// a walk down a path rather than an algebra problem. Two occurrences would need
/// like terms gathered — normalisation, and the start of a computer algebra
/// system — which is what [`Shape::Implicit`] refuses instead.
///
/// # Only arithmetic
///
/// `Pow`, `Rem`, `Max`, `Min`, `LogB` and every unary function but negation
/// answer `false`, matching `interval::invert_binary` and the arithmetic rows of
/// `invert_unary`. Widening this without widening those would claim a coordinate
/// is driven and then hand the walker the whole box for it, which is the one
/// combination that stalls rather than merely wastes a proposal.
fn reaches(side: &Expr, variable: GlobalId) -> bool {
    match &side.kind {
        Kind::Global(found) => *found == variable,

        Kind::Unary { op, arg } if interval::invertible_unary(*op) => reaches(arg, variable),

        Kind::Binary { op, lhs, rhs } if interval::invertible_binary(*op) => {
            let inner = if mentions(lhs, variable) { lhs } else { rhs };
            reaches(inner, variable)
        }

        _ => false,
    }
}

/// How many times `variable` is read in `expr`.
///
/// [`globals`] answers *whether*, which is not enough: peeling needs the
/// variable to occur exactly once, or the path walk leaves it on both sides.
fn occurrences(expr: &Expr, variable: GlobalId) -> usize {
    match &expr.kind {
        Kind::Global(id) => usize::from(*id == variable),
        Kind::Literal(_) | Kind::Local(_) => 0,
        Kind::Unary { arg, .. } => occurrences(arg, variable),
        Kind::Binary { lhs, rhs, .. }
        | Kind::Compare { lhs, rhs, .. }
        | Kind::NearEq { lhs, rhs, .. } => occurrences(lhs, variable) + occurrences(rhs, variable),
        Kind::And { terms } | Kind::Fold { terms, .. } => {
            terms.iter().map(|term| occurrences(term, variable)).sum()
        }
        Kind::DynamicIndex(index) => occurrences(index, variable),
        Kind::Block(block) => {
            block
                .assignments
                .iter()
                .map(|assignment| occurrences(&assignment.value, variable))
                .sum::<usize>()
                + occurrences(&block.result, variable)
        }
        Kind::Aggregate {
            lower, upper, body, ..
        } => {
            occurrences(lower, variable)
                + occurrences(upper, variable)
                + body
                    .assignments
                    .iter()
                    .map(|assignment| occurrences(&assignment.value, variable))
                    .sum::<usize>()
                + occurrences(&body.result, variable)
        }
    }
}

/// The split the walker consumes, over a whole system.
///
/// Returns `None` when nothing is driven, so a caller can keep its existing path
/// rather than carry an empty plan through it.
///
/// # Which drives are taken
///
/// Not all of them. A variable can only be defined once, a definition must not
/// depend on itself however indirectly, and a variable the schema does not name
/// cannot be a coordinate. Every drive refused this way leaves its constraint
/// exactly as it was — **no constraint is ever dropped**, because the walker
/// still checks feasibility against all of them. A refused drive costs
/// efficiency and can never cost correctness.
pub(crate) fn plan(constraints: &[Ast], schema: &Schema) -> Option<Plan> {
    // Position in the schema, and the positions its value is computed from.
    let mut definitions: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();

    for constraint in constraints {
        // Pinned and driven are handled identically here: a pinned variable is
        // a driven one that happens to depend on nothing.
        let (Shape::Driven { variable } | Shape::Pinned { variable }) = shape(constraint) else {
            continue;
        };

        let Some(position) = position_in(schema, constraint, variable) else {
            continue;
        };
        // Defined twice. Only one definition can hold, and choosing between them
        // is row E's matching problem rather than something to decide by which
        // constraint was written first — so take neither, and let both stay
        // ordinary constraints.
        if definitions.contains_key(&position) {
            definitions.remove(&position);
            continue;
        }

        // Everything the constraint names except the driven variable itself.
        // The definition is the rest of the equality rearranged, so it reads
        // exactly those — and knowing *which* is all the ordering below needs,
        // which is why nothing has to build the rearrangement to find out.
        let dependencies: BTreeSet<usize> = globals(&constraint.program.body.result)
            .into_iter()
            .filter(|global| *global != variable)
            .filter_map(|global| position_in(schema, constraint, global))
            .collect();
        definitions.insert(position, dependencies);
    }

    // Evaluation order, by repeatedly taking a definition whose dependencies are
    // all either free or already ordered. What is left over when nothing more
    // can be taken is a cycle, and every member of it stays undriven.
    let mut ordered: Vec<usize> = Vec::new();
    let mut settled: BTreeSet<usize> = BTreeSet::new();
    loop {
        let ready: Vec<usize> = definitions
            .iter()
            .filter(|(position, _)| !settled.contains(position))
            .filter(|(_, dependencies)| {
                dependencies
                    .iter()
                    .all(|d| settled.contains(d) || !definitions.contains_key(d))
            })
            .map(|(position, _)| *position)
            .collect();
        if ready.is_empty() {
            break;
        }
        for position in ready {
            settled.insert(position);
            ordered.push(position);
        }
    }

    let driven = ordered;
    if driven.is_empty() {
        return None;
    }

    let taken: BTreeSet<usize> = driven.iter().copied().collect();
    let free = (0..schema.len()).filter(|p| !taken.contains(p)).collect();

    Some(Plan { driven, free })
}

/// Where `variable` sits in the schema, given the constraint that named it.
///
/// [`GlobalId`] indexes a constraint's own symbol list, not the schema, so this
/// is a hop through the name. `None` means the schema does not carry it, which
/// binding would already have rejected — but this module must not panic on a
/// caller's behalf.
fn position_in(schema: &Schema, constraint: &Ast, variable: GlobalId) -> Option<usize> {
    let name = constraint.symbols.get(variable.index())?;
    schema
        .names()
        .iter()
        .position(|candidate| candidate == name)
}

/// Whether `variable` is read anywhere in `expr`.
fn mentions(expr: &Expr, variable: GlobalId) -> bool {
    globals(expr).contains(&variable)
}

/// Every global read anywhere in `expr`.
fn globals(expr: &Expr) -> BTreeSet<GlobalId> {
    let mut found = BTreeSet::new();
    collect(expr, &mut found);
    found
}

fn collect(expr: &Expr, found: &mut BTreeSet<GlobalId>) {
    match &expr.kind {
        Kind::Global(id) => {
            found.insert(*id);
        }
        Kind::Literal(_) | Kind::Local(_) => {}
        Kind::Unary { arg, .. } => collect(arg, found),
        Kind::Binary { lhs, rhs, .. } | Kind::Compare { lhs, rhs, .. } => {
            collect(lhs, found);
            collect(rhs, found);
        }
        Kind::NearEq { lhs, rhs, .. } => {
            collect(lhs, found);
            collect(rhs, found);
        }
        Kind::And { terms } | Kind::Fold { terms, .. } => {
            for term in terms {
                collect(term, found);
            }
        }
        Kind::DynamicIndex(index) => collect(index, found),
        Kind::Block(block) => {
            for assignment in &block.assignments {
                collect(&assignment.value, found);
            }
            collect(&block.result, found);
        }
        Kind::Aggregate {
            lower, upper, body, ..
        } => {
            collect(lower, found);
            collect(upper, found);
            for assignment in &body.assignments {
                collect(&assignment.value, found);
            }
            collect(&body.result, found);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> Ast {
        crate::parse(source).expect("test constraint should compile")
    }

    /// The variable a shape names, by name rather than by `GlobalId`, since an
    /// id is only meaningful against the constraint that produced it.
    fn driven_name(constraint: &Ast) -> Option<&str> {
        match shape(constraint) {
            Shape::Driven { variable, .. } | Shape::Pinned { variable, .. } => {
                Some(&constraint.symbols[variable.index()])
            }
            _ => None,
        }
    }

    #[test]
    fn a_bare_variable_on_one_side_is_driven() {
        let constraint = parse("y == sin(x) +/- 0.001");
        assert_eq!(driven_name(&constraint), Some("y"));
        assert!(matches!(shape(&constraint), Shape::Driven { .. }));
    }

    /// `sin(x) == y` says exactly what `y == sin(x)` says. Reading only the left
    /// would classify half of a symmetric relation.
    #[test]
    fn the_driven_side_may_be_either() {
        let constraint = parse("sin(x) == y +/- 0.001");
        assert_eq!(driven_name(&constraint), Some("y"));
    }

    /// A variable on both sides is implicit in it, whatever it is wrapped in.
    ///
    /// The rule was once narrower — on both sides *and* inside something the
    /// emitter refuses — which let `x2 == x1 + x2/2 - x3/x4` through on the
    /// grounds that Z3 can answer it. True, and beside the point: nothing
    /// downstream can drive such a variable, so it fell to whatever the sampler
    /// managed and read as a capability we did not have.
    #[test]
    fn a_variable_on_both_sides_is_implicit() {
        for source in [
            "x == sin(x) +/- 0.001",
            "x == ln(x) +/- 0.001",
            "x2 == x1 + 1/2*x2 - x3 / x4 +/- 0.001",
            "y == y/2 + x1 +/- 0.001",
            // Neither side is a bare variable, and it makes no difference: the
            // rule is about the variable being on both sides, not about how
            // either side is written. `x == x*x + 2` and `x*x == x + 2` are the
            // same equation and must get the same answer.
            "sin(x) == x/2 +/- 0.001",
            "x*x == x + 2 +/- 0.001",
            "x == x*x + 2 +/- 0.001",
        ] {
            assert!(
                matches!(shape(&parse(source)), Shape::Implicit { .. }),
                "{source} defines a variable in terms of itself"
            );
        }
    }

    /// The rearrangement every implicit form has, and which the diagnostic
    /// names. Refusing these too would be refusing the *problem* rather than a
    /// phrasing.
    ///
    /// The linear one now does better than merely being accepted: `x1` occurs
    /// once, so it is isolated and driven. That is the rearranged
    /// `cvg_pools::simple_arithmetic` fixture, and it means asking a user to
    /// write the non-circular form buys them a driven variable rather than
    /// only avoiding a refusal.
    #[test]
    fn the_rearranged_form_is_explicit() {
        assert!(matches!(
            shape(&parse("x2/2 - x1 + x3 / x4 == 0 +/- 0.001")),
            Shape::Driven { .. }
        ));
        // Three occurrences of `x`, so nothing to peel — accepted, not driven.
        assert_eq!(shape(&parse("x*x - x + 2 == 0 +/- 0.001")), Shape::Opaque);
    }

    /// Appearing repeatedly on the *far* side is not a self-reference. A count of
    /// occurrences would get this wrong; membership is the right question.
    #[test]
    fn a_variable_appearing_twice_on_the_far_side_is_still_driven() {
        let constraint = parse("y == x*x + x +/- 0.001");
        assert_eq!(driven_name(&constraint), Some("y"));
    }

    /// Row A. Pinned is a special case of driven, and reporting the weaker of the
    /// two would lose the fact that the coordinate never has to be recomputed.
    #[test]
    fn a_constant_side_is_pinned_not_driven() {
        let constraint = parse("x1 == pi +/- 0.001");
        assert!(matches!(shape(&constraint), Shape::Pinned { .. }));
    }

    /// A compound side still drives, as long as one variable can be peeled out
    /// of it.
    ///
    /// `x1 + x2 == 3` is *easier* than `y == sin(x)`, which drives today: one
    /// subtraction isolates it. It failed only because the test was for a bare
    /// `Kind::Global` on one side, which is a statement about spelling.
    #[test]
    fn a_compound_side_isolates_its_lone_variable() {
        let constraint = parse("x1 + x2 == 3 +/- 0.001");
        let Shape::Driven { variable, .. } = shape(&constraint) else {
            panic!("x1 + x2 == 3 should drive one of its variables");
        };
        // Deterministic, and `addition_undoes_to_subtraction` is what pins the
        // definition it produces.
        assert_eq!(constraint.symbols[variable.index()], "x1");
    }

    /// Peeling needs the variable to occur *once* — "linear in it", in the term
    /// rewriting sense. Twice on one side and the path walk would leave it on
    /// both, which is a wrong answer rather than a missing one.
    #[test]
    fn a_variable_occurring_twice_is_not_isolated() {
        assert_eq!(shape(&parse("x1 + x1 == 3 +/- 0.001")), Shape::Opaque);
        assert_eq!(shape(&parse("x1 * x1 == 3 +/- 0.001")), Shape::Opaque);
    }

    /// One variable being stuck does not stop another from being peeled out.
    ///
    /// `x1 + x2 + x1 == 3` cannot be solved for `x1` without gathering, and
    /// needs nothing at all to be solved for `x2` — `3 - x1 - x1`. The rule is
    /// per variable, not per constraint, which this pins because the first
    /// version of the test assumed otherwise and was wrong.
    #[test]
    fn a_stuck_variable_does_not_block_a_free_one() {
        let constraint = parse("x1 + x2 + x1 == 3 +/- 0.001");
        let Shape::Driven { variable } = shape(&constraint) else {
            panic!("x2 occurs once and should drive");
        };
        assert_eq!(constraint.symbols[variable.index()], "x2");
    }

    /// Only the arithmetic operators have an inverse here. Everything else
    /// declines rather than guesses — `^` and the unary functions are a second
    /// step with a branch problem of their own.
    #[test]
    fn an_operator_without_an_inverse_declines() {
        for source in [
            "x1 ^ 2 == 3 +/- 0.001",
            "max(x1, x2) == 3 +/- 0.001",
            "x1 % 3 == 1 +/- 0.001",
            // Periodic: `interval::invert_unary` answers `ENTIRE` for `sin`
            // rather than picking one solution of infinitely many, so nothing
            // could narrow this even if it were claimed as driven. The variable
            // has to be the only candidate — `sin(x1) + x2` drives *x2*, which
            // sits outside the `sin` and needs no inverse for it.
            "sin(x1) == 1 +/- 0.001",
            "sin(x1) + cos(x1) == 1 +/- 0.001",
        ] {
            assert_eq!(
                shape(&parse(source)),
                Shape::Opaque,
                "{source} has no inverse and must decline"
            );
        }
    }

    /// A function with an inverse is reached through, which the old `isolate`
    /// refused.
    ///
    /// It refused because it *built* the rearrangement, and a symbolic inverse
    /// of a non-injective function has to choose a branch — `sqr` gives two
    /// answers and `asin` infinitely many. `reaches` chooses nothing: it says
    /// the coordinate is determined, and `interval::slice` narrows it by
    /// intersecting the branches with what the argument can already be.
    #[test]
    fn a_function_with_an_inverse_is_reached_through() {
        for source in [
            "sqrt(x1) == 3 +/- 0.001",
            "ln(x1) + x2 == 3 +/- 0.001",
            "sqr(x1) + x2 == 9 +/- 0.001",
            "-cbrt(x1) == 2 +/- 0.001",
        ] {
            assert!(
                matches!(shape(&parse(source)), Shape::Driven { .. }),
                "{source}: `interval` can invert every operator on the path"
            );
        }
    }

    /// Both variables of `x1 + x2 == 3` can be isolated and either is correct,
    /// so the choice must at least not wander between runs. Choosing *well* is
    /// row E's matching problem and is not this.
    #[test]
    fn isolation_is_deterministic() {
        let first = shape(&parse("x1 + x2 == 3 +/- 0.001"));
        for _ in 0..8 {
            assert_eq!(shape(&parse("x1 + x2 == 3 +/- 0.001")), first);
        }
    }

    /// A bare side is still read by the existing path, not peeled to.
    /// `y == x1 + x2` drives `y` — the whole right side is its definition —
    /// where peeling would have isolated `x1` and left `y` to be searched.
    #[test]
    fn a_bare_side_still_wins() {
        let constraint = parse("y == x1 + x2 +/- 0.001");
        let Shape::Driven { variable, .. } = shape(&constraint) else {
            panic!("y == x1 + x2 should drive y");
        };
        assert_eq!(constraint.symbols[variable.index()], "y");
    }

    /// The taxonomy is about equalities. A comparison has no side that defines
    /// the other, and reading `x > 4` as pinning `x` would be badly wrong.
    #[test]
    fn an_inequality_is_not_classified() {
        assert_eq!(shape(&parse("x > 4")), Shape::Opaque);
        assert_eq!(shape(&parse("x1 + x2 > 20 - x3^2")), Shape::Opaque);
    }

    /// A scalar expression is not a constraint and has no sides to read.
    #[test]
    fn a_scalar_expression_is_not_classified() {
        assert_eq!(shape(&parse("x + 1")), Shape::Opaque);
    }

    /// `var[i]` reads a variable the expression never names, so "does it appear
    /// on the other side" has no answer. `Ast::contains_dynamic_lookup` exists to
    /// warn about exactly this and it would be careless to ignore it here.
    #[test]
    fn a_dynamic_lookup_defeats_classification() {
        assert_eq!(
            shape(&parse("1.5 == var[1] + var[2] +/- 0.001")),
            Shape::Opaque
        );
    }

    // -----------------------------------------------------------------------
    // The system level
    // -----------------------------------------------------------------------

    fn plan_over(names: &[&str], sources: &[&str]) -> Option<Plan> {
        let schema = Schema::for_names(names);
        let constraints: Vec<Ast> = sources.iter().map(|s| parse(s)).collect();
        plan(&constraints, &schema)
    }

    #[test]
    fn two_independent_drives_are_both_taken() {
        let plan = plan_over(
            &["x1", "x2", "x3", "x4"],
            &["x1 == sqrt(x2) +/- 0.001", "x3 == cbrt(x4) +/- 0.001"],
        )
        .expect("both should drive");

        let driven = plan.driven();
        assert_eq!(driven, [0, 2]);
        assert_eq!(plan.free(), &[1, 3]);
    }

    /// `z` is defined from `y`, which is itself defined. Computing `z` first
    /// reads whatever `y` happened to hold, so the order is not cosmetic.
    #[test]
    fn a_chain_of_drives_is_ordered() {
        let plan = plan_over(
            &["x", "y", "z"],
            &["y == sin(x) +/- 0.001", "z == y + 1 +/- 0.001"],
        )
        .expect("both should drive");

        let driven = plan.driven();
        assert_eq!(driven, [1, 2], "y must be computed before z");
        assert_eq!(plan.free(), &[0]);
    }

    /// The same chain with the schema declared backwards, so the correct order
    /// is the *reverse* of the numeric one.
    ///
    /// Without this, `a_chain_of_drives_is_ordered` proves nothing: definitions
    /// are held in a `BTreeMap` and iterate by position, so `[1, 2]` comes out
    /// right whether or not any topological sort happened. Here `z` sits at 0 and
    /// `y` at 1, and taking them in position order would compute `z` from a stale
    /// `y`.
    #[test]
    fn evaluation_order_beats_declaration_order() {
        let plan = plan_over(
            &["z", "y", "x"],
            &["y == sin(x) +/- 0.001", "z == y + 1 +/- 0.001"],
        )
        .expect("both should drive");

        let driven = plan.driven();
        assert_eq!(
            driven,
            vec![1, 0],
            "y (at 1) must be computed before z (at 0)"
        );
    }

    /// A cycle of length two, caught by the same graph that catches row F at
    /// length one. Neither can be computed first, so neither is driven — and
    /// both constraints stay in force.
    #[test]
    fn a_cycle_between_drives_is_refused() {
        assert!(
            plan_over(
                &["x", "y"],
                &["y == x + 1 +/- 0.001", "x == y + 1 +/- 0.001"]
            )
            .is_none(),
            "a cycle should drive nothing"
        );
    }

    /// Two definitions of one variable. Choosing between them is row E's matching
    /// problem; picking the first written would be arbitrary, so neither is
    /// taken. What must *not* happen is one constraint quietly ceasing to apply.
    #[test]
    fn a_variable_driven_twice_is_driven_once_or_not_at_all() {
        let plan = plan_over(
            &["x", "y", "z"],
            &["y == x + 1 +/- 0.001", "y == z * 2 +/- 0.001"],
        );
        assert!(
            plan.is_none(),
            "an ambiguous definition should not be resolved by source order"
        );
    }

    /// Nothing to drive is the common case and must not cost the caller a plan
    /// it then has to check.
    #[test]
    fn a_system_with_nothing_driven_has_no_plan() {
        assert!(plan_over(&["x1", "x2", "x3"], &["x1 + x2 > 20 - x3^2"]).is_none());
    }

    /// Row A across every dimension: both pinned, nothing left free. Legal, and
    /// the walker has to cope with an empty free list rather than divide by its
    /// length.
    #[test]
    fn a_fully_pinned_system_leaves_nothing_free() {
        let plan = plan_over(&["x1", "x2"], &["x1 == pi +/- 0.001", "x2 == e +/- 0.001"])
            .expect("both should pin");
        assert_eq!(plan.free(), &[] as &[usize]);
        assert_eq!(plan.driven().len(), 2);
    }
}
