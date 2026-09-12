//! A feasible point near a given one, deterministically.
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
//! occupy with the others held where they are, from [`ConstraintSystem::slice`]
//! — and clamping into it is the exact axis projection onto that coordinate's
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
//! **Shotgun.** Where clamping cannot land — two bands with a gap between, a
//! disc approached from a corner — the anchors decide. From each of the nearest
//! few anchors that are themselves feasible, bisect along the segment toward
//! the point: the anchor end is feasible, the far end is not, and the feasible
//! end of the final bracket is a point on the boundary between them. That
//! landing then gets the clamp stage again, which from a feasible point is
//! precisely "step each coordinate its clearance off the nearest wall". The
//! nearest such point is the answer, and an anchor that has the clearance is a
//! candidate in its own right, which is what makes "never farther than the
//! nearest anchor" a guarantee rather than a hope. One directed chord per
//! anchor is the whole of what a chain biased toward the point would find, at
//! a thousandth of the cost.
//!
//! **Release.** Both stages can move a coordinate that, in hindsight, did not
//! need to move — a clamp computed against a neighbour that then moved too, a
//! chord that carried every coordinate when one constraint was active. So the
//! last thing done to any answer is to put each moved coordinate back where it
//! was, one at a time, keeping every reversion that keeps the clearance.
//!
//! # Clearance: a deliberate step inside, not an ulp
//!
//! The caller stores points normalised to a unit cube and denormalises them
//! before evaluating. That round trip is one ulp off for some values, and one
//! ulp on a coordinate can move a residual by `1e-8` when the coordinate enters
//! a term at the fourth power. A landing a few ulps inside a bound — which is
//! what the ladder alone produces, and what a sixty-bit bisection produces
//! along a chord — is feasible here and infeasible by the time the caller
//! looks at it. Found at a vertex of the tension spring, where two constraints
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
//! # The metric is L1 over box-normalised coordinates
//!
//! Normalised, because the caller thinks in a unit cube and "near" in metres
//! and pascals at once means nothing. L1 rather than L2, because the contrast
//! between nearest and farthest neighbour collapses in high dimension and
//! collapses fastest for the higher norms (Aggarwal, Hinneburg & Keim, 2001);
//! at fifty to two hundred dimensions over a census of thousands, the nearest
//! anchor under L2 is barely nearer than the farthest. And along a segment L1
//! is linear in the parameter, which is what makes the bisection's endpoint
//! provably no farther than its anchor.
//!
//! # What is deliberately not here
//!
//! No randomness — not a draw, not a seed. No gradient: babel has no
//! derivatives, and a finite-difference normal is ill-defined at exactly the
//! vertices the clearance exists for, where the axis and the chord are inward
//! directions already in hand. No solver: an SMT call is milliseconds to
//! seconds where this budget is microseconds, and transcendentals are outside
//! its theories anyway. No brute force: it cannot localise in high dimension,
//! and it would re-solve the find-a-first-point problem the whole module is
//! built around on every call.

use faer::MatRef;

use crate::{ConstraintSystem, Point};

/// How many rounds of clamping a point gets before the anchors take over.
///
/// A clamp is computed with the other coordinates held, so a coordinate that
/// moves can open room for another; a few rounds catch that. Most points land
/// in one, and a point that has not landed after this many is one the slices
/// do not describe — a gap the chord has to cross.
const CLAMP_SWEEPS: usize = 8;

/// How many of the nearest anchors get a chord.
///
/// More than one because the nearest anchor by L1 is not always the one whose
/// chord ends nearest — a chord stops at the first boundary it meets, and the
/// second-nearest anchor may see the point across a shorter stretch of
/// infeasible space. Eight is enough that the census's nearest cluster is
/// covered and few enough that a repair stays microseconds.
const ANCHOR_SHOTS: usize = 8;

/// Bisection steps along a chord. Each halves the bracket, so this is a budget
/// in bits and sixty is past where an `f64` parameter in `[0, 1]` can still be
/// halved.
const CHORD_BITS: usize = 60;

/// How far inward a landing may nudge, as a power of two in ulps.
///
/// A slice is padded outward by a few ulps per operator, so a clamp onto its
/// edge lands a hair outside. The nudge doubles from one ulp; twenty doublings
/// is a relative `1e-10`, which is well past any padding a real expression
/// accumulates and still nothing next to the tolerances constraints carry.
const LANDING_LADDER: u32 = 20;

/// Why [`repair`] could not answer.
#[derive(Debug, Clone, PartialEq)]
pub enum RepairError {
    /// Nothing feasible was reached: clamping could not land, and no anchor
    /// among the nearest few was feasible. Between two bands with no anchor to
    /// bisect toward, no interval says which way to go.
    Stranded,
    /// A feasible point was reached, but nowhere with the clearance asked for:
    /// the feasible room there is narrower than twice the clearance. `nearest`
    /// is that point, feasible by the plain oracle, in case it is better than
    /// nothing; a smaller clearance is the usual answer.
    Cramped { nearest: Point, clearance: f64 },
}

impl std::fmt::Display for RepairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stranded => write!(
                f,
                "no feasible point was reached: clamping could not land and no anchor was feasible"
            ),
            Self::Cramped { clearance, .. } => write!(
                f,
                "a feasible point was reached but none with clearance {clearance}; the feasible \
                 room is narrower than the clearance asked for, so try a smaller one"
            ),
        }
    }
}

impl std::error::Error for RepairError {}

/// A point that satisfies `system` with room to spare, near `point`, the same
/// every time.
///
/// `anchors` are feasible points the caller believes in, one per column in
/// schema order — the matrix [`FeasibleSamples::take`](crate::FeasibleSamples::take)
/// hands out is the intended source. They are judged rather than trusted:
/// an infeasible column is skipped. With no anchors at all the answer is
/// whatever clamping alone reaches, which is enough wherever the constraints
/// name where their feasible side is, and [`RepairError::Stranded`] where they
/// do not.
///
/// `clearance` is the room kept from every wall, as a fraction of each
/// variable's box width: the result and each of its `2d` axis neighbours
/// `clearance * width` away pass the same feasibility check the search uses —
/// inside the box, every constraint `<= 0`, nothing non-finite. `0.0` asks for
/// feasibility alone and lands on the bounds. A caller that normalises points
/// and back wants a few thousand ulps of the unit cube, `1e-12`: far above
/// what any per-coordinate round trip loses and invisible to an optimiser.
/// There is no default because the right value is the caller's own noise
/// floor.
///
/// A point that already has the clearance comes back unchanged, so
/// `repair(repair(x)) == repair(x)`; a feasible point without it is moved
/// inward. Given the same system, anchors, point and clearance, the same
/// output, bit for bit. And never farther from `point`, in L1 over
/// box-normalised coordinates, than the nearest anchor with the clearance
/// among the few considered.
///
/// # Errors
/// [`RepairError::Stranded`] when nothing feasible was reached at all, and
/// [`RepairError::Cramped`] when something feasible was but the clearance
/// could not be had there.
///
/// # Panics
/// If `point` or `anchors` do not have one entry per variable, or `clearance`
/// is negative or not finite. That is a caller mixing up systems, not a
/// verdict about the point.
pub fn repair(
    system: &ConstraintSystem,
    anchors: MatRef<'_, f64>,
    point: &[f64],
    clearance: f64,
) -> Result<Point, RepairError> {
    let dimensions = system.variables().len();
    assert_eq!(
        point.len(),
        dimensions,
        "a point has one coordinate per variable of the system it is repaired against"
    );
    assert_eq!(
        anchors.nrows(),
        dimensions,
        "anchors have one row per variable of the system they anchor"
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

    // Anchors nearest to where the point now stands, after whatever clamping
    // achieved; a clamp that did not land still moved the point onto the right
    // side of the constraints it could read, and the chord is shorter for it.
    let mut ranked: Vec<(f64, usize)> = (0..anchors.ncols())
        .map(|column| {
            let anchor: Point = (0..dimensions).map(|row| anchors[(row, column)]).collect();
            (distance(&anchor, &current), column)
        })
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut best: Option<(f64, Point)> = None;
    for (_, column) in ranked.into_iter().take(ANCHOR_SHOTS) {
        let anchor: Point = (0..dimensions).map(|row| anchors[(row, column)]).collect();
        if !system.is_feasible(&anchor, 0.0) {
            continue;
        }

        // The anchor is a candidate at `t = 0` when it has the clearance
        // itself; a chord can only end nearer than that or lose to it.
        if system.is_feasible(&anchor, clearance) {
            let reached = distance(&anchor, point);
            if best.as_ref().is_none_or(|(nearest, _)| reached < *nearest) {
                best = Some((reached, anchor.clone()));
            }
        }

        // `t = 0` is the anchor and feasible; `t = 1` is `current` and lacks
        // the clearance, or the clamp stage would have returned. Halve the
        // bracket toward wherever feasibility ends. A segment through a gap is
        // bracketed just the same — the feasible end is always a probe that
        // passed, or the anchor itself. When `current` is feasible but cramped
        // every probe passes and the landing is `current`; the back-off below
        // then walks it toward the anchor.
        let mut landed = anchor.clone();
        let (mut lower, mut upper) = (0.0_f64, 1.0_f64);
        for _ in 0..CHORD_BITS {
            let middle = 0.5 * (lower + upper);
            if middle <= lower || middle >= upper {
                break;
            }
            let mut probe: Point = anchor
                .iter()
                .zip(&current)
                .map(|(from, to)| from + middle * (to - from))
                .collect();
            system.settle(&mut probe);
            if system.is_feasible(&probe, 0.0) {
                lower = middle;
                landed = probe;
            } else {
                upper = middle;
            }
        }

        // The landing is feasible and within a bit of the boundary along the
        // chord; the clamp steps it off the walls it is against.
        let (stepped, cramped_here) = match clamped(system, &widths, landed.clone(), clearance) {
            Ok(clear) => (Some(clear), landed),
            Err(reached) => {
                // No wall the slices could read — a constraint narrowing
                // declines, as it does a real exponent — so the only inward
                // direction in hand is the chord itself. Back off along it
                // toward the anchor in doublings of the clearance until the
                // oracle is satisfied; the first rung that passes is within a
                // factor of two of the least back-off, which at this scale is
                // all the precision the answer can use.
                let length = distance(&anchor, &current);
                let unit = if clearance > 0.0 && length > 0.0 {
                    clearance / length
                } else {
                    0.0
                };
                let mut backed = None;
                let mut rung = unit;
                while unit > 0.0 && rung < lower {
                    let mut probe: Point = anchor
                        .iter()
                        .zip(&current)
                        .map(|(from, to)| from + (lower - rung) * (to - from))
                        .collect();
                    system.settle(&mut probe);
                    if system.is_feasible(&probe, clearance) {
                        backed = Some(probe);
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
                (backed, consolation)
            }
        };
        match stepped {
            Some(clear) => {
                let reached = distance(&clear, point);
                if best.as_ref().is_none_or(|(nearest, _)| reached < *nearest) {
                    best = Some((reached, clear));
                }
            }
            None => {
                let at = distance(&cramped_here, point);
                if cramped.as_ref().is_none_or(|(nearest, _)| at < *nearest) {
                    cramped = Some((at, cramped_here));
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
/// Called on the caller's point and on every chord landing. From an infeasible
/// point it is the axis projection described in the module doc; from a
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
                let slice = system.slice(&current, coordinate);
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
            // `is_feasible_after` with the clearance is exactly that check; it
            // is used here as a filter over one coordinate's constraints and
            // never as the judge, which the full check below remains.
            //
            // The ulp is the box width's, not the value's. A bound at zero has
            // ulps of `5e-324`, and a strict comparison is satisfied only past
            // an absolute `f64::MIN_POSITIVE`, which no ladder of those reaches;
            // the width is the scale the residual's own rounding lives at.
            if !system.is_feasible_after(&current, coordinate, clearance) {
                let inward = if target > value { 1.0 } else { -1.0 };
                let magnitude = target.abs().max(widths[coordinate]).max(f64::MIN_POSITIVE);
                let ulp = magnitude.next_up() - magnitude;
                for rung in 0..LANDING_LADDER {
                    let nudged = target + inward * ulp * f64::from(1u32 << rung);
                    current[coordinate] = nudged;
                    if system.is_feasible_after(&current, coordinate, clearance) {
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
    // clamp may already have the clearance — a chord landing that met no
    // wall — and is judged rather than presumed.
    if system.is_feasible(&current, clearance) {
        Ok(current)
    } else {
        Err(current)
    }
}

/// `candidate` with every coordinate put back to its value in `original`
/// wherever that keeps the clearance, one coordinate at a time in schema
/// order.
///
/// Clear in, clear out: a reversion is kept only if the whole point still
/// passes with its clearance, so this can only shorten the distance to
/// `original`, never break the answer. It exists because both stages over-move
/// — a clamp is computed against neighbours that then move too, and a chord
/// carries every coordinate when one constraint was active — and hindsight is
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
