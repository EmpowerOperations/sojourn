//! A feasible point near a given one, deterministically.
//!
//! Reached as [`FeasibleRegion::repair`](crate::FeasibleRegion::repair), which
//! carries the contract; this module is the algorithm, over a
//! `&ConstraintSystem`. It lives on the solved region rather than on the
//! system because a region that could not be solved has nothing to repair
//! toward, and because a caller with a region in hand is the only one who
//! ever asks.
//!
//! The consumer is an optimizer that cannot evaluate its objective at an
//! infeasible point — the constraints encode things like mesh validity, and a
//! bad point is a crashed solver rather than a bad number. Every point it
//! proposes that fails the constraints comes here, thousands of times a run,
//! and must come back feasible by the same oracle the search uses, near where
//! it wanted to be, and the same every time it is asked. The contract and the
//! alternatives it displaced are written up in `docs/todo.md` under *Repair
//! for Artemis*; this is the shape that survived.
//!
//! # Two stages, in series
//!
//! **Clamp.** Every coordinate has a conditional slice — the interval it may
//! occupy with the others held where they are, from [`interval::slice`] — and
//! clamping into it is the exact axis projection onto that coordinate's
//! constraints. Clamps are applied cheapest first, cumulatively, and the point
//! is judged after each. Cheapest first matters: the L1 projection onto a
//! half-space moves the single coordinate with the steepest normal component,
//! and greedy-by-cost reproduces it where a sweep in schema order can answer
//! several times farther. A driven coordinate gets no special treatment here —
//! it is one more coordinate with a slice, and if satisfying its equality by
//! moving *it* costs more than moving something it depends on, the cheaper
//! move wins. The settled-mask ordering in `retract` is for travelling along a
//! surface, not for landing on it once.
//!
//! A clamp lands at the caller's clearance inside the bound, and *on* the
//! bound when the clearance is zero, because a bound is the answer: Artemis
//! measured coordinates sitting exactly at a bound coming out several times
//! more accurate than ones a tolerance away. The slice is a superset padded
//! outward by a few ulps, so a landing on the bound itself nudges inward by a
//! doubling ladder of ulps until the coordinate's own constraints pass.
//!
//! **Project.** Where clamping cannot land — two bands with a gap between, a
//! disc approached from a corner — the point is projected: the feasible point
//! nearest it, sought by a local solve from where clamping left it
//! ([`local::nearest`]), `min ‖u − p‖²` over the unit cube subject to every
//! constraint with the clearance built into its rows. The landing depends on
//! the constraints and the point and on nothing else. That is the property
//! the consumer needs, and the one the previous design lacked: it bisected a
//! chord from the nearest of a set of *anchors* the caller supplied, so every
//! landing was a convex combination of the proposal and a census point — every
//! coordinate dragged toward wherever the census happened to be — and over
//! thousands of repairs the optimizer was herded toward the census rather
//! than along the boundary its objective preferred. Measured on a disc with
//! anchors clustered at angle zero: the corner at 45° landed at 17°. A
//! projection lands it at 45°.
//!
//! A landing with the clearance gets the clamp stage again, which from a
//! feasible point is precisely "step each coordinate its clearance off the
//! nearest wall" and a no-op where it already has it. A landing that is
//! feasible but without the clearance — where no slice can read a wall, as
//! narrowing declines a real exponent — is backed off along the projection's
//! own direction, `p → x*` continued inward, in doublings of the clearance
//! until the oracle passes: that direction is the boundary's normal at the
//! landing, which is the inward direction the chord used to stand in for.
//!
//! **Release.** Both stages can move a coordinate that, in hindsight, did not
//! need to move — a clamp computed against a neighbour that then moved too, a
//! projection that carried every coordinate when one constraint was active. So
//! the last thing done to any answer is to put each moved coordinate back
//! where it was, one at a time, keeping every reversion that keeps the
//! clearance.
//!
//! # Clearance: a deliberate step inside, not an ulp
//!
//! The caller stores points normalised to a unit cube and denormalises them
//! before evaluating. That round trip is one ulp off for some values, and one
//! ulp on a coordinate can move a residual by `1e-8` when the coordinate enters
//! a term at the fourth power. A landing a few ulps inside a bound — which is
//! what the ladder alone produces — is feasible here and infeasible by the
//! time the caller looks at it. Found at a vertex of the tension spring, where two constraints
//! are active at once; `tests/regression_fixture.rs` keeps the case.
//!
//! So `repair` takes a **clearance**: a fraction of each variable's box width
//! that the answer keeps between itself and every wall, in every coordinate
//! direction. One scalar, but `clearance * width` on each coordinate, which is
//! the per-axis tolerance the box's own scale implies, in the unit cube the
//! caller normalises into and the metric this module measures distance in. A
//! point *has* the clearance when it is feasible and so are its `2d` axis
//! neighbours at that distance — [`ConstraintSystem::is_feasible`] with the
//! clearance, which judges each constraint at the point and along each
//! variable it names — which is exactly "survives any per-coordinate
//! perturbation smaller than the step", and on a convex region is the whole
//! L1 ball of that radius.
//!
//! The oracle judges; the slices construct. A clamp lands inside the slice
//! shrunk by the step at each end, and a slice is an outer enclosure, so
//! "inside the shrunk slice" is where to aim and never a proof. A point that
//! truly has the clearance sits inside every shrunk slice, though, so the clamp
//! never moves a point that already qualifies. Where a slice is narrower than
//! twice the step the clamp aims for its middle, and if nothing reached has the
//! clearance the answer is [`RepairError::Cramped`] with the nearest feasible
//! point in it: the region is thinner than the caller's own noise floor, and
//! saying so beats handing back a point that will fail on the next look.
//!
//! # The metric is over box-normalised coordinates
//!
//! Normalised, because the caller thinks in a unit cube and "near" in metres
//! and pascals at once means nothing. What this module measures and reports
//! in — the consolation in [`RepairError::Cramped`], the back-off's unit — is
//! L1, which along a segment is linear in the parameter. The projection
//! itself minimises L2², because COBYLA fits linear models and a kink at every
//! axis fights them; the two agree on what is near enough for the cases here,
//! and the optimizer being repaired measures in its own metric anyway.
//!
//! # What is deliberately not here
//!
//! No randomness — not a draw, not a seed. No gradient: babel has no
//! derivatives, and a finite-difference normal is ill-defined at exactly the
//! vertices the clearance exists for, where the axis is an inward direction
//! already in hand. No anchors: nothing the caller has seen elsewhere may
//! influence where a point lands, for the reason above. No brute force: it
//! cannot localise in high dimension, and it would re-solve the
//! find-a-first-point problem the whole module is built around on every call.

use crate::cvg::incidence::Row;
use crate::cvg::{classify, interval, local};
use crate::{ConstraintSystem, Point};

/// How many rounds of clamping a point gets before the projection takes over.
///
/// A clamp is computed with the other coordinates held, so a coordinate that
/// moves can open room for another; a few rounds catch that. Most points land
/// in one, and a point that has not landed after this many is one the slices
/// do not describe — a gap or a corner the projection has to find its way
/// round.
const CLAMP_SWEEPS: usize = 8;

/// How far inward a landing may nudge, as a power of two in ulps.
///
/// A slice is padded outward by a few ulps per operator, so a clamp onto its
/// edge lands a hair outside. The nudge doubles from one ulp; twenty doublings
/// is a relative `1e-10`, which is well past any padding a real expression
/// accumulates and still nothing next to the tolerances constraints carry.
const LANDING_LADDER: u32 = 20;

/// Why [`FeasibleRegion::repair`](crate::FeasibleRegion::repair) could not answer.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RepairError {
    /// Nothing feasible was reached: clamping could not land, and the
    /// projection found no feasible evaluation within its budget — a region
    /// too thin for a local solve to fall into, or none at all.
    #[error(
        "no feasible point was reached: clamping could not land and the projection found nothing"
    )]
    Stranded,
    /// A feasible point was reached, but nowhere with the clearance asked for:
    /// the feasible room there is narrower than twice the clearance. `nearest`
    /// is that point, feasible by the plain oracle, in case it is better than
    /// nothing; a smaller clearance is the usual answer.
    #[error(
        "a feasible point was reached but none with clearance {clearance}; the feasible room is          narrower than the clearance asked for, so try a smaller one"
    )]
    Cramped { nearest: Point, clearance: f64 },
}

/// A point that satisfies `system` with `clearance` to spare, near `point`,
/// the same every time. The contract — clearance, the guarantees, the errors
/// and the panics — is documented on
/// [`FeasibleRegion::repair`](crate::FeasibleRegion::repair), the only caller.
pub(crate) fn repair(
    system: &ConstraintSystem,
    point: &[f64],
    clearance: f64,
) -> Result<Point, RepairError> {
    let dimensions = system.variables().len();
    assert_eq!(
        point.len(),
        dimensions,
        "a point has one coordinate per variable of the system it is repaired against"
    );
    assert!(
        clearance.is_finite() && clearance >= 0.0,
        "a clearance is a non-negative fraction of the box width, not {clearance}"
    );

    let current: Point = point.to_vec();
    if system.is_feasible(&current, clearance) {
        return Ok(current);
    }

    let widths: Vec<f64> = system
        .variables()
        .iter()
        .map(|variable| variable.upper_bound - variable.lower_bound)
        .collect();
    let distance = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .zip(&widths)
            .map(|((x, y), width)| {
                if *width > 0.0 {
                    (x - y).abs() / width
                } else {
                    0.0
                }
            })
            .sum()
    };

    // The nearest feasible point seen that lacks the clearance: the answer's
    // consolation if nothing with the clearance is found.
    let mut cramped: Option<(f64, Point)> = None;
    let current = match clamped(system, &widths, current, clearance) {
        Ok(landed) => return Ok(released(system, landed, point, clearance)),
        Err(reached) => {
            if system.is_feasible(&reached, 0.0) {
                cramped = Some((distance(&reached, point), reached.clone()));
            }
            reached
        }
    };

    // The projection, from where clamping left the point: a clamp that did not
    // land still moved it onto the right side of the constraints it could
    // read, and the solve starts nearer for it. "Nearest" is measured to the
    // caller's point throughout.
    let projected = local::nearest(system, &current, point, clearance);

    let mut best: Option<(f64, Point)> = None;
    if let Some(landed) = projected.clear {
        // The landing has the clearance; the clamp from a feasible point only
        // steps a coordinate off a wall it is against, and leaves this one be.
        let clear = match clamped(system, &widths, landed.clone(), clearance) {
            Ok(clear) => clear,
            Err(_) => landed,
        };
        best = Some((distance(&clear, point), clear));
    }

    if best.is_none()
        && let Some(landed) = projected.feasible
    {
        // Feasible, without the clearance, and the rows had the margin built
        // in: the constraints here are ones the margin could not read either
        // — a real exponent that faults a step away. The clamp gets its turn,
        // and failing that the only inward direction in hand is the
        // projection's own: back off along `point -> landed`, continued past
        // the landing, in doublings of the clearance until the oracle is
        // satisfied. The first rung that passes is within a factor of two of
        // the least back-off, which at this scale is all the precision the
        // answer can use.
        match clamped(system, &widths, landed.clone(), clearance) {
            Ok(clear) => best = Some((distance(&clear, point), clear)),
            Err(reached) => {
                let length = distance(&landed, point);
                let unit = if clearance > 0.0 && length > 0.0 {
                    clearance / length
                } else {
                    0.0
                };
                let mut rung = unit;
                while unit > 0.0 && rung < 1.0 {
                    let mut probe: Point = point
                        .iter()
                        .zip(&landed)
                        .map(|(from, to)| from + (1.0 + rung) * (to - from))
                        .collect();
                    classify::settle(system, &mut probe);
                    if system.is_feasible(&probe, clearance) {
                        best = Some((distance(&probe, point), probe));
                        break;
                    }
                    rung *= 2.0;
                }
                // The clamp's own result is the consolation where it stayed
                // feasible — the middle of a slab beats its edge — and the
                // landing otherwise.
                let consolation = if system.is_feasible(&reached, 0.0) {
                    reached
                } else {
                    landed
                };
                let at = distance(&consolation, point);
                if cramped.as_ref().is_none_or(|(nearest, _)| at < *nearest) {
                    cramped = Some((at, consolation));
                }
            }
        }
    }

    match (best, cramped) {
        (Some((_, landed)), _) => Ok(released(system, landed, point, clearance)),
        (None, Some((_, nearest))) => Err(RepairError::Cramped { nearest, clearance }),
        (None, None) => Err(RepairError::Stranded),
    }
}

/// `from` clamped into its slices, sweep by sweep, until it has the clearance:
/// `Ok` with that point, `Err` with the point as far as clamping got.
///
/// Called on the caller's point and on the projection's landing. From an
/// infeasible point it is the axis projection described in the module doc; from a
/// feasible one it is the step off the walls, since a coordinate already
/// inside its shrunk slice is not moved.
fn clamped(
    system: &ConstraintSystem,
    widths: &[f64],
    from: Point,
    clearance: f64,
) -> Result<Point, Point> {
    let mut current = from;
    for _ in 0..CLAMP_SWEEPS {
        // Every coordinate's clamp, with what it costs, from the point as it
        // stands at the start of the sweep. Cost is measured before any are
        // applied, which is what makes "cheapest first" a statement about the
        // point rather than about the order the coordinates happen to be in.
        let mut clamps: Vec<(f64, usize, f64)> = (0..current.len())
            .filter_map(|coordinate| {
                let slice = interval::slice(system, &current, coordinate);
                if slice.is_empty() {
                    return None;
                }
                let value = current[coordinate];
                // Aim the clearance inside each edge of the slice; where the
                // slice has no room for that, aim for its middle and let the
                // oracle say whether that was enough.
                let step = clearance * widths[coordinate];
                let (lo, hi) = (slice.lo() + step, slice.hi() - step);
                let target = if lo <= hi {
                    value.clamp(lo, hi)
                } else {
                    0.5 * (slice.lo() + slice.hi())
                };
                if target == value {
                    return None;
                }
                let cost = if widths[coordinate] > 0.0 {
                    (target - value).abs() / widths[coordinate]
                } else {
                    0.0
                };
                Some((cost, coordinate, target))
            })
            .collect();
        if clamps.is_empty() {
            break;
        }
        clamps.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

        for (_, coordinate, target) in clamps {
            let value = current[coordinate];
            current[coordinate] = target;

            // The slice's edge is a hair outside the true bound, so step inward
            // — toward where the value came from is *outward* — until the
            // constraints naming this coordinate pass, with the clearance:
            // a landing `step` inside the padded edge is a hair short of
            // `step` inside the wall, and its neighbours tie with the wall.
            // `landed` is exactly that check; it is a filter over one
            // coordinate's constraints and never the judge, which the full
            // check below remains.
            //
            // The ulp is the box width's, not the value's. A bound at zero has
            // ulps of `5e-324`, and a strict comparison is satisfied only past
            // an absolute `f64::MIN_POSITIVE`, which no ladder of those reaches;
            // the width is the scale the residual's own rounding lives at.
            if !landed(system, &current, coordinate, clearance) {
                let inward = if target > value { 1.0 } else { -1.0 };
                let magnitude = target.abs().max(widths[coordinate]).max(f64::MIN_POSITIVE);
                let ulp = magnitude.next_up() - magnitude;
                for rung in 0..LANDING_LADDER {
                    let nudged = target + inward * ulp * f64::from(1u32 << rung);
                    current[coordinate] = nudged;
                    if landed(system, &current, coordinate, clearance) {
                        break;
                    }
                    current[coordinate] = target;
                }
            }

            if system.is_feasible(&current, clearance) {
                return Ok(current);
            }
        }
    }
    // Nothing left to clamp, or the sweeps ran out. A point with nothing to
    // clamp may already have the clearance — a projection's landing that met
    // no wall — and is judged rather than presumed.
    if system.is_feasible(&current, clearance) {
        Ok(current)
    } else {
        Err(current)
    }
}

/// Whether the constraints naming `coordinate` hold at `point` with the
/// clearance, and the box does on that coordinate.
///
/// The clamp's filter, never its judge: mid-clamp the rest of the system may
/// still be broken, and the question is only whether *this* landing is on the
/// right side of the walls it can see. A constraint naming none of the
/// coordinates that moved is not consulted; a computed subscript counts as
/// naming every coordinate, which the incidence graph carries.
fn landed(system: &ConstraintSystem, point: &Point, coordinate: usize, clearance: f64) -> bool {
    if !system.within(point, coordinate, clearance) {
        return false;
    }
    let mut neighbour = point.clone();
    system.incidence.affected(Row(coordinate)).iter().all(|id| {
        let rows = system.incidence.rows_of(*id);
        system.constraints[id.index()].holds(
            point,
            rows,
            &system.variables,
            clearance,
            &mut neighbour,
        )
    })
}

/// `candidate` with every coordinate put back to its value in `original`
/// wherever that keeps the clearance, one coordinate at a time in schema
/// order.
///
/// Clear in, clear out: a reversion is kept only if the whole point still
/// passes with its clearance, so this can only shorten the distance to
/// `original`, never break the answer. It exists because both stages over-move
/// — a clamp is computed against neighbours that then move too, and a
/// projection carries every coordinate when one constraint was active — and hindsight is
/// one check per coordinate. The clearance is part of the check because a
/// coordinate the caller left an ulp from a wall is exactly what must not be
/// handed back.
fn released(
    system: &ConstraintSystem,
    mut candidate: Point,
    original: &[f64],
    clearance: f64,
) -> Point {
    for coordinate in 0..candidate.len() {
        let moved = candidate[coordinate];
        if moved == original[coordinate] {
            continue;
        }
        candidate[coordinate] = original[coordinate];
        if !system.is_feasible(&candidate, clearance) {
            candidate[coordinate] = moved;
        }
    }
    candidate
}
