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
//! # Clamp, project, and — on a flat constraint — walk in from a reference
//!
//! **Clamp.** Every coordinate has a conditional slice — the interval it may
//! occupy with the others held where they are, from [`interval::slice`] — and
//! clamping into it is the exact axis projection onto that coordinate's
//! constraints. Clamps are applied cheapest first, cumulatively, and the point
//! is judged after each. A driven coordinate gets no special treatment here —
//! it is one more coordinate with a slice. The settled-mask ordering in
//! `retract` is for travelling along a surface, not for landing on it once.
//!
//! A clamp lands at the caller's clearance inside the bound, and *on* the
//! bound when the clearance is zero, because a bound is the answer: Artemis
//! measured coordinates sitting exactly at a bound coming out several times
//! more accurate than ones a tolerance away. The slice is a superset padded
//! outward by a few ulps, so a landing on the bound itself nudges inward by a
//! doubling ladder of ulps until the coordinate's own constraints pass.
//!
//! The clamp is the whole answer where the feasible set is *separable* on the
//! coordinates it moved — every constraint naming one of them names it alone,
//! as bounds do — because there the axis projection is the Euclidean one. It
//! used to be the whole answer wherever it landed at all, which made the
//! metric taxicab: on a slab the point moved one coordinate by the whole gap
//! where two should each have moved half, `√2` farther, every time; a step a
//! hundredth over a curved wall was slid along the wall to wherever one
//! coordinate could reach, up to half again as far. An optimizer stepping over
//! a wall wants to be put back where it stepped from (`tests/regression_fixture.rs`,
//! `repair_lands_axis_aligned_not_nearest`). Everywhere else the clamp's
//! landing is a candidate and the warm start.
//!
//! **Project.** The feasible point nearest the caller's, `min ‖u − p‖²` over
//! the unit cube subject to every constraint, from the clamp's landing when
//! it landed and from wherever the clamp got to when it did not. Two solvers,
//! in order. [`newton::nearest`] first: Newton on the KKT system over the
//! active set, with the gradients the tape's reverse sweep provides — exact
//! in one step on a linear constraint, a handful on a curved one, microseconds
//! at fifty variables — and it declines wherever a constraint that bites has
//! no derivative or the iteration does not converge. Then [`local::nearest`],
//! COBYLA, derivative-free, only where Newton declined or its landing could
//! not be stepped off the walls: it needs no gradient and pays `O(d²m)` per
//! iteration for that, and it is not run beside a successful Newton because
//! on a smooth piece Newton is exact and COBYLA cannot beat it, on a
//! nonconvex one both are local, and the difference is a thousandfold in
//! cost. From a feasible warm start either solve slides along the boundary to
//! the Euclidean foot. The landing depends on the constraints and the point
//! and on nothing else. That is the property the consumer needs, and the one
//! an earlier design lacked: it bisected a chord from the nearest of a set of
//! *anchors* the caller supplied, so every landing was a convex combination of
//! the proposal and a census point — every coordinate dragged toward wherever
//! the census happened to be — and over thousands of repairs the optimizer was
//! herded toward the census rather than along the boundary its objective
//! preferred. Measured on a disc with anchors clustered at angle zero: the
//! corner at 45° landed at 17°. A projection lands it at 45°.
//!
//! A landing with the clearance gets the clamp stage again, which from a
//! feasible point is precisely "step each coordinate its clearance off the
//! nearest wall" and a no-op where it already has it. A landing that is
//! feasible but without the clearance — where no slice can read a wall, as
//! narrowing declines a real exponent — is backed off along the projection's
//! own direction, `p → x*` continued inward, in doublings of the clearance
//! until the oracle passes: that direction is the boundary's normal at the
//! landing, which is the inward direction a chord used to stand in for.
//!
//! **Reference.** A constraint can be flat where the point stands — Keane's
//! `0.75 − ∏xᵢ` with seven coordinates at `1e-11` is `0.75` to fifty digits
//! in every direction — and then no slice reads a wall and no linear model
//! moves: nothing feasible is seen at all. No local method sees a flat
//! constraint; what is needed is a feasible point to walk *in* from. It comes
//! from [`local::find_initial`] over the declared box under a fixed seed, so
//! that it is a function of the system alone — the same reference on every
//! call, on every machine — and not of anything the region sampled, which is
//! how this differs from the anchors: they were the caller's census, arbitrary
//! and different every run, and they were used *first*. The chord from the
//! reference to the point is bisected to the last feasible point along it,
//! and the projection then runs from that landing, where the constraint is
//! well-scaled again, so the reference decides only which basin the answer is
//! in and never where in it. Reached only when the projection from the point
//! itself saw nothing feasible.
//!
//! **Sample.** Whenever Newton declined — off the smooth path — beside the
//! derivative-free solve: a box around the point, uniform draws in it judged
//! one by one with their driven coordinates put on their surfaces, the box
//! doubled while it holds nothing and shrunk onto the nearest hit once it
//! does, for a fixed number of rounds; then the chord from the nearest hit
//! toward the point. The backstop that needs no slice, no gradient and no
//! continuity — only a region fat enough to sample over its free
//! coordinates — which is exactly what a constraint with a *jump* in it
//! leaves: `floor`, `%`, `sgn` defeat the clamp and Newton, COBYLA's landing
//! is then wherever its evaluations happened to fall feasible, and a chord
//! from the reference lands wherever it crossed in — 0.40 of the box away on
//! a comb whose nearest cell sat at 0.20, 0.235 on a checkerboard whose
//! nearest sat at 0.007, ten units away on a checkerboard beside a driven
//! product (`tests/cvg_repair.rs`, the ball oracle across a jump). Sampling
//! around the point finds the nearest cell, and the aggregator takes the
//! nearer of it and the solver's. It cannot localise a thin region, and it
//! cannot localise anything past a handful of dimensions; both of those are
//! the projection's, which is why it waits for Newton to decline. Its draws
//! come from a fixed seed, for the reason the reference's do. `Stranded`
//! means the box and the seed search both found nothing.
//!
//! **Release.** Every stage can move a coordinate that, in hindsight, did not
//! need to move — a clamp computed against a neighbour that then moved too, a
//! projection that carried every coordinate when one constraint was active. So
//! the last thing done to any candidate is to put each moved coordinate back
//! where it was, one at a time, keeping every reversion that keeps the
//! clearance; the nearest candidate after that is the answer.
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
//! # The metric is Euclidean over box-normalised coordinates
//!
//! Normalised, because the caller thinks in a unit cube and "near" in metres
//! and pascals at once means nothing. Euclidean, because that is the
//! consumer's question and human intuition's: the taxicab metric this used to
//! measure in prefers any landing that touches one coordinate over a nearer
//! one that touches two, which is a bias toward the axes an optimizer should
//! not be fed. Taxicab was chosen for ranking anchors, where the contrast
//! between nearest and farthest neighbour collapses in high dimension and
//! collapses fastest for the higher norms (Aggarwal, Hinneburg & Keim, 2001);
//! the anchors are gone and the argument went with them. Candidates are
//! ranked, the consolation chosen and the back-off's unit measured in the
//! same metric the projection minimises.
//!
//! # What is deliberately not here
//!
//! No randomness the caller can see: the draws the reference and the
//! sampling box make come from constant seeds, so the answer is a function
//! of the system, the point and the clearance. No finite differences: the
//! gradients Newton reads are the tape's own reverse sweep, exact, and a
//! constraint without one declines rather than being approximated at a
//! vertex where an approximation is ill-defined. No anchors: nothing the
//! caller has seen elsewhere may influence where a point lands, for the
//! reason above. No brute force *first*: it cannot localise a thin region
//! or a high-dimensional one, and on the cases the projection handles it
//! would re-solve the find-a-first-point problem on every call; it is the
//! last stage, for the cases nothing else can read.

use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};

use crate::cvg::incidence::Row;
use crate::cvg::{Cancellation, classify, interval, local, newton};
use crate::{ConstraintSystem, Point};

/// How many rounds of clamping a point gets before the projection starts
/// from wherever it got to.
///
/// A clamp is computed with the other coordinates held, so a coordinate that
/// moves can open room for another; a few rounds catch that. Most points land
/// in one, and a point that has not landed after this many is one the slices
/// do not describe — a gap or a corner the projection has to find its way
/// round.
const CLAMP_SWEEPS: usize = 8;

/// Bisection steps along the chord from a reference point. Each halves the
/// bracket, so this is a budget in bits and sixty is past where an `f64`
/// parameter in `[0, 1]` can still be halved.
const CHORD_BITS: usize = 60;

/// The seed the reference point's local solve draws its extra starts from.
///
/// A constant, and deliberately so: `repair` is a function of the system,
/// the point and the clearance, and the reference it walks from on a flat
/// constraint has to be a function of the system alone — the same reference
/// on every call, on every machine — or a landing would depend on what the
/// region happened to have sampled, which is the bias the anchors had and the
/// reason they are gone. The first start is the box centre and needs no
/// draw; this seeds the ones after it.
const REFERENCE_SEED: u64 = 0x5E_ED_0F_1D;

/// The seed the sampling box draws from: a constant, for the same reason as
/// [`REFERENCE_SEED`].
const SAMPLING_SEED: u64 = 0xB0_0B_0F_5A;

/// Rounds the sampling box gets: each doubles the box when it held nothing
/// and shrinks it onto the nearest hit when it did.
///
/// The box starts at a sixty-fourth of every width and doubles to the whole
/// declared box in six rounds; the rest shrink. Twenty is far past where
/// shrinking stops finding nearer points at this many draws.
const SAMPLING_ROUNDS: usize = 20;

/// Draws per round of the sampling box, judged one by one.
const SAMPLING_DRAWS: usize = 256;

/// Where the sampling box starts, as a fraction of every width.
const SAMPLING_RADIUS: f64 = 1.0 / 64.0;

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

    // Every landing any stage produced; the nearest with the clearance is the
    // answer, the nearest feasible one without it the consolation.
    let mut landings: Vec<Landing> = Vec::new();

    // The clamp: a candidate where it lands, and where it does not, still the
    // right side of every constraint a slice could read, which is where the
    // projection starts from either way.
    let start = match clamped(system, &widths, current, clearance) {
        Ok(landed) => {
            // Where every constraint that names a coordinate the clamp moved
            // names that coordinate alone — bounds — the feasible set is a
            // product of one-dimensional sets on those coordinates and the
            // axis projection *is* the Euclidean one: nothing for the
            // projection to improve, and at two hundred bounds a solve it
            // would spend minutes on.
            let separable = landed
                .iter()
                .zip(point)
                .enumerate()
                .filter(|(_, (moved, was))| moved != was)
                .all(|(coordinate, _)| {
                    system
                        .incidence
                        .naming(Row(coordinate))
                        .iter()
                        .all(|id| system.incidence.rows_of(*id).len() == 1)
                });
            if separable {
                return Ok(released(system, landed, point, clearance));
            }
            landings.push(Landing::Clear(landed.clone()));
            landed
        }
        Err(reached) => {
            if system.is_feasible(&reached, 0.0) {
                landings.push(Landing::Feasible(reached.clone()));
            }
            reached
        }
    };

    // The projection, always otherwise: from the clamp's landing it slides
    // along the boundary to the Euclidean foot, which the clamp — an axis
    // projection — reaches only where one coordinate is the whole answer.
    // "Nearest" is measured to the caller's point throughout. Newton on the
    // KKT system first, exact and cheap where every active constraint has a
    // gradient; the derivative-free solve only where it declined or its
    // landing could not be stepped off the walls.
    if let Some(landing) = newton::nearest(system, &start, point)
        && let Some(landing) = off_the_boundary(system, &widths, landing, point, clearance)
    {
        landings.push(landing);
    }
    if !landings.iter().any(Landing::is_clear) {
        landings.extend(stepped_off(
            system,
            &widths,
            local::nearest(system, &start, point, clearance),
            point,
            clearance,
        ));

        // Off the smooth path — a constraint with a jump in it, or one too
        // curved for Newton — a derivative-free landing may be anywhere the
        // solver's evaluations happened to fall feasible; the sampling box
        // around the point is the second opinion, and the aggregator picks.
        // See the module doc.
        if let Some(hit) = sampled(system, &widths, point, clearance) {
            let chord = along_chord(system, &hit, point);
            landings.extend(off_the_boundary(system, &widths, chord, point, clearance));
        }
    }

    // Nothing feasible seen at all: a constraint flat where the point
    // stands, which no slice, no linear model and no box around the point
    // can read. Walk in from a reference point instead — see the module doc.
    if landings.is_empty() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(REFERENCE_SEED);
        let reference = local::find_initial(
            system,
            &system.declared(),
            local::STARTS,
            &mut rng,
            &Cancellation::never(),
        );
        if let Some(reference) = reference {
            let chord = along_chord(system, &reference, point);
            landings.extend(stepped_off(
                system,
                &widths,
                local::nearest(system, &chord, point, clearance),
                point,
                clearance,
            ));
            // The projection could not leave the chord's landing either — flat
            // again — so the landing itself, stepped off its walls, stands.
            if !landings.iter().any(Landing::is_clear) {
                landings.push(match clamped(system, &widths, chord.clone(), clearance) {
                    Ok(clear) => Landing::Clear(clear),
                    Err(_) => Landing::Feasible(chord),
                });
            }
        }
    }

    // Every candidate is released before it is measured: reverting a
    // coordinate to the caller's value can only bring it nearer.
    let mut best: Option<(f64, Point)> = None;
    let mut cramped: Option<(f64, Point)> = None;
    for landing in landings {
        match landing {
            Landing::Clear(candidate) => {
                let candidate = released(system, candidate, point, clearance);
                nearer(&mut best, distance(&widths, &candidate, point), candidate);
            }
            Landing::Feasible(candidate) => {
                nearer(
                    &mut cramped,
                    distance(&widths, &candidate, point),
                    candidate,
                );
            }
        }
    }

    match (best, cramped) {
        (Some((_, landed)), _) => Ok(landed),
        (None, Some((_, nearest))) => Err(RepairError::Cramped { nearest, clearance }),
        (None, None) => Err(RepairError::Stranded),
    }
}

/// What a stage of `repair` hands up: a point with the clearance, or one
/// that is feasible without it.
enum Landing {
    Clear(Point),
    Feasible(Point),
}

impl Landing {
    fn is_clear(&self) -> bool {
        matches!(self, Self::Clear(_))
    }
}

/// Keeps `candidate` in `slot` if it is nearer than what is there.
fn nearer(slot: &mut Option<(f64, Point)>, at: f64, candidate: Point) {
    if slot.as_ref().is_none_or(|(nearest, _)| at < *nearest) {
        *slot = Some((at, candidate));
    }
}

/// Euclidean distance over box-normalised coordinates: the metric the
/// consumer measures in, see the module doc. A zero-width coordinate
/// contributes nothing.
fn distance(widths: &[f64], a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .zip(widths)
        .map(|((x, y), width)| {
            if *width > 0.0 {
                let scaled = (x - y) / width;
                scaled * scaled
            } else {
                0.0
            }
        })
        .sum::<f64>()
        .sqrt()
}

/// The landings a derivative-free projection yields, once its points have
/// been stepped off their walls: at most one with the clearance and one
/// without.
///
/// A `clear` landing gets the clamp from a feasible point, which only steps a
/// coordinate off a wall it is against and leaves one with room alone. A
/// `feasible` landing without the clearance is a landing on the boundary,
/// and goes the way every such landing goes: [`off_the_boundary`].
fn stepped_off(
    system: &ConstraintSystem,
    widths: &[f64],
    projected: local::Projected,
    point: &[f64],
    clearance: f64,
) -> Vec<Landing> {
    let mut landings = Vec::new();
    if let Some(landing) = projected.clear {
        let clear = match clamped(system, widths, landing.clone(), clearance) {
            Ok(clear) => clear,
            Err(_) => landing,
        };
        landings.push(Landing::Clear(clear));
    }
    if let Some(landing) = projected.feasible {
        landings.extend(off_the_boundary(system, widths, landing, point, clearance));
    }
    landings
}

/// A landing on the boundary — Newton's, which is there by construction, or
/// a derivative-free one judged feasible without the clearance — stepped
/// inside.
///
/// The clamp first: from a feasible point it is precisely "step each
/// coordinate its clearance off the nearest wall". Where no slice can read
/// the wall — narrowing declines a real exponent — the only inward direction
/// in hand is the projection's own: back off along `point -> landing`,
/// continued past the landing, in doublings of the clearance until the
/// oracle is satisfied. That direction is `−Σ λᵢ ∇gᵢ`, the boundary's normal
/// at the landing, which is what makes a small step along it the least the
/// clearance can cost. The first rung that passes is within a factor of two
/// of the least back-off, which at this scale is all the precision the
/// answer can use. Where nothing passes, the consolation is the clamp's own
/// result if it stayed feasible — the middle of a slab beats its edge — and
/// the landing itself if that is; a landing a hair outside with nothing
/// inside it is no landing at all.
fn off_the_boundary(
    system: &ConstraintSystem,
    widths: &[f64],
    landing: Point,
    point: &[f64],
    clearance: f64,
) -> Option<Landing> {
    let reached = match clamped(system, widths, landing.clone(), clearance) {
        Ok(clear) => return Some(Landing::Clear(clear)),
        Err(reached) => reached,
    };
    let length = distance(widths, &landing, point);
    let unit = if clearance > 0.0 && length > 0.0 {
        clearance / length
    } else {
        0.0
    };
    let mut rung = unit;
    while unit > 0.0 && rung < 1.0 {
        let mut probe: Point = point
            .iter()
            .zip(&landing)
            .map(|(from, to)| from + (1.0 + rung) * (to - from))
            .collect();
        classify::settle(system, &mut probe);
        if system.is_feasible(&probe, clearance) {
            return Some(Landing::Clear(probe));
        }
        rung *= 2.0;
    }
    if system.is_feasible(&reached, 0.0) {
        Some(Landing::Feasible(reached))
    } else if system.is_feasible(&landing, 0.0) {
        Some(Landing::Feasible(landing))
    } else {
        None
    }
}

/// The nearest feasible point the sampling box finds around `point`: uniform
/// draws in a box centred on it, the box doubled while it holds nothing and
/// shrunk onto the nearest hit once it does, for a fixed number of rounds.
///
/// The backstop that needs no slice, no gradient and no continuity — only
/// that the region be fat enough to sample *over the free coordinates*: a
/// driven equality is not sampled but computed, every draw's driven
/// coordinates put at the middle of their bands, so a band a millionth
/// wide costs nothing. Draws are judged with the clearance. A hit is the box's answer;
/// the chord from it toward the point refines along that one ray. Nothing
/// past a handful of dimensions localises this way, and it is reached only
/// when every stage that could has declined.
fn sampled(
    system: &ConstraintSystem,
    widths: &[f64],
    point: &[f64],
    clearance: f64,
) -> Option<Point> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(SAMPLING_SEED);
    let mut radius = SAMPLING_RADIUS;
    let mut best: Option<(f64, Point)> = None;
    for _ in 0..SAMPLING_ROUNDS {
        let mut found = false;
        for _ in 0..SAMPLING_DRAWS {
            let mut draw: Point = point
                .iter()
                .zip(widths)
                .zip(system.variables())
                .map(|((centre, width), variable)| {
                    let step = radius * width;
                    rng.random_range(centre - step..=centre + step)
                        .clamp(variable.lower_bound, variable.upper_bound)
                })
                .collect();
            // Driven coordinates onto their surfaces, at the middle of the
            // band: a clamp to its edge lands an ulp outside and every draw
            // fails, which is how a jump beside a driven equality stranded.
            classify::centre(system, &mut draw);
            if !system.is_feasible(&draw, clearance) {
                continue;
            }
            found = true;
            let at = distance(widths, &draw, point);
            if best.as_ref().is_none_or(|(nearest, _)| at < *nearest) {
                best = Some((at, draw));
            }
        }
        radius = match &best {
            // Shrink onto the nearest hit: it sits on the box's boundary.
            Some((nearest, _)) if found => *nearest,
            Some((nearest, _)) => nearest.min(radius),
            None => (radius * 2.0).min(1.0),
        };
    }
    best.map(|(_, hit)| hit)
}

/// The last feasible point along the segment from `reference` to `point`,
/// by bisection: `t = 0` is the reference and feasible, `t = 1` is the point
/// and is not, and the bracket halves toward wherever feasibility ends, with
/// driven coordinates settled at every probe. A segment through a gap is
/// bracketed just the same — the feasible end is always a probe that passed,
/// or the reference itself.
fn along_chord(system: &ConstraintSystem, reference: &[f64], point: &[f64]) -> Point {
    let mut landed: Point = reference.to_vec();
    let (mut lower, mut upper) = (0.0_f64, 1.0_f64);
    for _ in 0..CHORD_BITS {
        let middle = 0.5 * (lower + upper);
        if middle <= lower || middle >= upper {
            break;
        }
        let mut probe: Point = reference
            .iter()
            .zip(point)
            .map(|(from, to)| from + middle * (to - from))
            .collect();
        classify::settle(system, &mut probe);
        if system.is_feasible(&probe, 0.0) {
            lower = middle;
            landed = probe;
        } else {
            upper = middle;
        }
    }
    landed
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
