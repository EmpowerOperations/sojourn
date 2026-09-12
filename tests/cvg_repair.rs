//! `cvg::repair`: a feasible point near a given one, deterministically.
//!
//! The contract is Artemis's (the optimizer that consumes this crate): hand
//! over any point in the declared box, get back one the same feasibility
//! oracle passes, near the input in **L1 over box-normalised coordinates**,
//! the same answer every time. The design and the alternatives it rejected are
//! in `docs/todo.md` under *Repair for Artemis*.
//!
//! The geometry cases have closed-form answers under that metric, which is not
//! the Euclidean projection anyone would sketch: the L1-nearest point on a
//! half-space is reached by moving *one* coordinate, the one with the steepest
//! normal component, and the L1-nearest point on a disc from a point whose
//! other coordinate is already in range is straight along one axis. Every
//! expected value below was derived under L1 first and the test written second.
//!
//! Anchors are hand-written wherever the geometry is closed-form, so these
//! cases do not depend on the sampler. Only the property test draws its anchors
//! from a solve.

mod common;

use faer::Mat;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;
use sojourn::{ConstraintSystem, InputVariable, RepairError, Satisfiability};

/// A validated [`ConstraintSystem`], panicking on a fixture that does not bind.
fn system(variables: Vec<InputVariable>, constraints: &[&str]) -> ConstraintSystem {
    ConstraintSystem::new(variables, constraints.iter().copied())
        .expect("a fixture's constraints should bind to its own box")
}

fn variables(specs: &[(&str, f64, f64)]) -> Vec<InputVariable> {
    specs
        .iter()
        .map(|(name, low, high)| InputVariable::new(*name, *low, *high))
        .collect()
}

/// Points as the matrix `repair` takes: one column each, the shape
/// `FeasibleSamples::take` returns.
fn anchors(points: &[&[f64]], rows: usize) -> Mat<f64> {
    Mat::from_fn(rows, points.len(), |row, column| points[column][row])
}

/// Same value the other cvg suites use, so a point seen in one is the point
/// seen in another.
const SEED: u64 = 0x50_50_1E_5E_ED;

/// The clearance the clearance fixtures ask for: a thousandth of each box
/// width, large enough to see in a closed-form answer and small next to every
/// fixture's geometry. The geometry fixtures above it run at `0.0`, which is
/// the landing-on-the-bound contract, pinned as it was.
const CLEARANCE: f64 = 1e-3;

/// Whether every constraint holds at `point`, judged independently of `repair`
/// through the public evaluator. A test that trusts the thing it is testing is
/// not a test. Strict: the residual must be `<= 0`, no tolerance, because that
/// is what the caller's own evaluator will demand.
fn holds(system: &ConstraintSystem, point: &[f64]) -> bool {
    let bindings: Vec<(&str, f64)> = system
        .variables()
        .iter()
        .zip(point)
        .map(|(variable, value)| (variable.name.as_str(), *value))
        .collect();
    system.constraints().all(|constraint| {
        common::eval_one(constraint, &bindings).is_ok_and(|residual| residual <= 0.0)
    })
}

/// Whether `point` and each of its `2d` axis neighbours `clearance` box
/// widths away hold, through the same independent evaluator. This is the
/// clearance contract as `repair` states it, checked without `repair`.
fn has_clearance(system: &ConstraintSystem, point: &[f64], clearance: f64) -> bool {
    if !holds(system, point) || !in_box(system, point) {
        return false;
    }
    let mut neighbour = point.to_vec();
    for (coordinate, variable) in system.variables().iter().enumerate() {
        let step = clearance * (variable.upper_bound - variable.lower_bound);
        for sign in [-1.0, 1.0] {
            neighbour[coordinate] = point[coordinate] + sign * step;
            if !holds(system, &neighbour) || !in_box(system, &neighbour) {
                return false;
            }
        }
        neighbour[coordinate] = point[coordinate];
    }
    true
}

fn in_box(system: &ConstraintSystem, point: &[f64]) -> bool {
    system
        .variables()
        .iter()
        .zip(point)
        .all(|(variable, value)| variable.contains(*value))
}

/// L1 distance with every coordinate scaled by its box width: the metric
/// `repair` claims to be near in.
fn normalised_l1(system: &ConstraintSystem, a: &[f64], b: &[f64]) -> f64 {
    system
        .variables()
        .iter()
        .zip(a.iter().zip(b))
        .map(|(variable, (x, y))| (x - y).abs() / (variable.upper_bound - variable.lower_bound))
        .sum()
}

#[test]
fn a_half_space_is_entered_along_its_steep_coordinate() {
    // `2*x1 + x2 < 1` from (1, 1). Moving `x1` alone reaches the boundary at
    // `x1 = 0`, a cost of 1; moving `x2` alone needs `x2 = -1`, a cost of 2.
    // The L1 projection is the first, and nothing about the second coordinate
    // should change at all.
    let system = system(
        variables(&[("x1", -2.0, 2.0), ("x2", -2.0, 2.0)]),
        &["2*x1 + x2 < 1"],
    );
    let anchors = anchors(&[&[0.0, 0.0]], 2);

    let repaired = sojourn::repair(&system, anchors.as_ref(), &[1.0, 1.0], 0.0)
        .expect("a half-space is reachable");

    assert!(
        holds(&system, &repaired),
        "{repaired:?} violates the half-space"
    );
    assert_eq!(repaired[1], 1.0, "the cheap coordinate was left alone");
    assert!(
        repaired[0] < 0.0 && repaired[0] > -1e-9,
        "x1 should land just inside the boundary at 0, got {}",
        repaired[0]
    );
}

#[test]
fn a_disc_is_entered_where_the_diamond_touches_it() {
    // From (2, 0.5) the L1 ball grows as a diamond, and its vertex reaches the
    // unit disc at (sqrt(0.75), 0.5) before any edge does. Only `x` moves.
    //
    // Written with `sqr`; the next fixture is the same disc spelled `x^2`.
    let system = system(
        variables(&[("x", -2.0, 2.0), ("y", -2.0, 2.0)]),
        &["sqr(x) + sqr(y) < 1"],
    );
    let anchors = anchors(&[&[0.0, 0.0]], 2);

    let repaired =
        sojourn::repair(&system, anchors.as_ref(), &[2.0, 0.5], 0.0).expect("a disc is reachable");

    assert!(
        holds(&system, &repaired),
        "{repaired:?} is outside the disc"
    );
    assert_eq!(
        repaired[1], 0.5,
        "y was already in range and should not move"
    );
    let expected = 0.75_f64.sqrt();
    assert!(
        repaired[0] < expected && expected - repaired[0] < 1e-6,
        "x should land just inside the circle at {expected}, got {}",
        repaired[0]
    );
}

#[test]
fn a_disc_spelled_with_a_power_is_entered_the_same_way() {
    // `x^2` is how every optimizer formulation spells it. Narrowing inverts a
    // whole power through its root, so the clamp finds the landing the `sqr`
    // spelling finds. Before it did, the front end had expanded `x^2` into a
    // product fold nothing could invert, and without an interval for `x` this
    // fell to the chord from the origin and landed on the radial point —
    // feasible, but a tenth farther in L1 than the answer.
    let system = system(
        variables(&[("x", -2.0, 2.0), ("y", -2.0, 2.0)]),
        &["x^2 + y^2 < 1"],
    );
    let anchors = anchors(&[&[0.0, 0.0]], 2);

    let repaired =
        sojourn::repair(&system, anchors.as_ref(), &[2.0, 0.5], 0.0).expect("a disc is reachable");

    assert!(
        holds(&system, &repaired),
        "{repaired:?} is outside the disc"
    );
    assert_eq!(
        repaired[1], 0.5,
        "y was already in range and should not move"
    );
    let expected = 0.75_f64.sqrt();
    assert!(
        repaired[0] < expected && expected - repaired[0] < 1e-6,
        "x should land just inside the circle at {expected}, got {}",
        repaired[0]
    );
}

#[test]
fn a_driven_coordinate_is_not_privileged() {
    // `2*x1 + x2 == 3 +/- 0.001` from (3, 3). Driving `x2` to satisfy the
    // equality moves it to -2.999, a cost of 6; clamping `x1` moves it to
    // 0.0005, a cost of 3. The equality classifies as driven, and repair must
    // still pick the cheaper coordinate rather than the computed one.
    let system = system(
        variables(&[("x1", -5.0, 5.0), ("x2", -5.0, 5.0)]),
        &["2*x1 + x2 == 3 +/- 0.001"],
    );
    let anchors = anchors(&[&[1.0, 1.0]], 2);

    let repaired =
        sojourn::repair(&system, anchors.as_ref(), &[3.0, 3.0], 0.0).expect("a slab is reachable");

    assert!(
        holds(&system, &repaired),
        "{repaired:?} is outside the slab"
    );
    assert_eq!(
        repaired[1], 3.0,
        "x2 is the expensive coordinate and should not move"
    );
    assert!(
        (repaired[0] - 0.0005).abs() < 1e-6,
        "x1 should land at the slab's edge near 0.0005, got {}",
        repaired[0]
    );
}

#[test]
fn the_nearer_band_wins_over_the_anchor_it_started_from() {
    // Two bands, at -2 and 1, each about 0.00033 wide. No interval narrowing
    // separates them, so this is decided by the anchors: from 0.9 the band at 1
    // is a tenth away and the band at -2 is nearly three, and from -1 it is the
    // other way round. Whichever anchor is tried first, the result is the
    // nearer band.
    let system = system(
        variables(&[("x", -5.0, 5.0)]),
        &["(x + 2) * (x - 1) == 0 +/- 0.001"],
    );
    let anchors = anchors(&[&[-2.0], &[1.0]], 1);

    let near_one =
        sojourn::repair(&system, anchors.as_ref(), &[0.9], 0.0).expect("a band is reachable");
    assert!(
        holds(&system, &near_one),
        "{near_one:?} is outside both bands"
    );
    assert!(
        (near_one[0] - 1.0).abs() < 0.001,
        "from 0.9 the band at 1 is nearer, got {}",
        near_one[0]
    );

    let near_minus_two =
        sojourn::repair(&system, anchors.as_ref(), &[-1.0], 0.0).expect("a band is reachable");
    assert!(
        holds(&system, &near_minus_two),
        "{near_minus_two:?} is outside both bands"
    );
    assert!(
        (near_minus_two[0] + 2.0).abs() < 0.001,
        "from -1 the band at -2 is nearer, got {}",
        near_minus_two[0]
    );
}

#[test]
fn a_domain_hole_is_just_infeasible() {
    // `ln(x1) > 0` cannot be evaluated at -0.5: the evaluator faults rather than
    // producing a residual. That is an infeasible point like any other, and the
    // interval `ln` inverts to says where the feasible ones are. Where exactly
    // the boundary sits is the evaluator's call — its own rounding admits
    // `x1 = 1` — so the claim is "at the boundary as the oracle draws it", not
    // "above 1 in the reals".
    let system = system(variables(&[("x1", -1.0, 3.0)]), &["ln(x1) > 0"]);
    let anchors = anchors(&[&[2.0]], 1);

    let repaired = sojourn::repair(&system, anchors.as_ref(), &[-0.5], 0.0)
        .expect("the log's domain is reachable");

    assert!(
        holds(&system, &repaired),
        "{repaired:?} is outside ln's feasible range"
    );
    assert!(
        (repaired[0] - 1.0).abs() < 1e-9,
        "x1 should land on the boundary at 1, got {}",
        repaired[0]
    );
}

#[test]
fn two_hundred_bounds_are_landed_on_exactly() {
    // Every coordinate bounded below by 10.5, every one starting at 10.2. A
    // clamp lands each on its bound in one sweep, and the landing must be
    // exact: Artemis measured coordinates *at* a bound coming out several times
    // more accurate than ones merely near it.
    const DIMENSIONS: usize = 200;
    let names: Vec<String> = (1..=DIMENSIONS).map(|i| format!("x{i}")).collect();
    let specs: Vec<(&str, f64, f64)> = names.iter().map(|n| (n.as_str(), 10.0, 11.0)).collect();
    let sources: Vec<String> = names.iter().map(|n| format!("{n} > 10.5")).collect();
    let sources: Vec<&str> = sources.iter().map(String::as_str).collect();
    let system = system(variables(&specs), &sources);
    let anchor = vec![10.75; DIMENSIONS];
    let anchors = anchors(&[&anchor], DIMENSIONS);

    let repaired = sojourn::repair(&system, anchors.as_ref(), &vec![10.2; DIMENSIONS], 0.0)
        .expect("a corner is reachable");

    assert!(
        holds(&system, &repaired),
        "some coordinate is not above its bound"
    );
    for (index, value) in repaired.iter().enumerate() {
        assert!(
            *value > 10.5 && value - 10.5 < 1e-9,
            "x{} should land just above 10.5, got {value}",
            index + 1
        );
    }
}

#[pollster::test]
async fn repair_holds_its_contract_over_a_polytope() {
    // Five variables under three loose inequalities, anchored on a census from
    // a solve: the shape Artemis actually runs. For points scattered over the
    // whole box: the result is feasible with clearance by an independent
    // evaluation, inside the box, a fixed point of `repair`, the same on a
    // second call, and never farther than the nearest anchor that has the
    // clearance — which is a candidate itself.
    const ANCHORS: usize = 256;
    const TRIALS: usize = 64;
    let inputs = variables(&[
        ("x1", 0.0, 1.0),
        ("x2", 0.0, 1.0),
        ("x3", 0.0, 1.0),
        ("x4", 0.0, 1.0),
        ("x5", 0.0, 1.0),
    ]);
    let sources = ["x1 + x2 > x3", "x2 + x3 > x4", "x3 + x4 > x5"];
    let system = system(inputs.clone(), &sources);

    let solution = common::solver()
        .with_seed(SEED)
        .solve(system.clone())
        .await
        .expect("solving should not fail");
    let mut pool = match solution {
        Satisfiability::Satisfied { samples } => samples,
        Satisfiability::Unsatisfiable { because } => panic!("reported unsatisfiable: {because:?}"),
    };
    let anchors = pool.take(ANCHORS);
    assert_eq!(anchors.ncols(), ANCHORS, "the census should fill");
    // Only an anchor with the clearance is a candidate `repair` may answer
    // with, so only those bound how far it may move. Judged once here: the
    // independent evaluator compiles per call, and this is the hot loop.
    let clear_anchors: Vec<Vec<f64>> = (0..ANCHORS)
        .map(|column| {
            (0..anchors.nrows())
                .map(|row| anchors[(row, column)])
                .collect()
        })
        .filter(|anchor: &Vec<f64>| has_clearance(&system, anchor, CLEARANCE))
        .collect();
    assert!(
        !clear_anchors.is_empty(),
        "the census should have room to spare"
    );

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
    let mut complaints = Vec::new();
    for _ in 0..TRIALS {
        let point: Vec<f64> = (0..inputs.len())
            .map(|_| rng.random_range(0.0..1.0))
            .collect();
        let Ok(repaired) = sojourn::repair(&system, anchors.as_ref(), &point, CLEARANCE) else {
            complaints.push(format!("{point:?}: no repair"));
            continue;
        };
        if !has_clearance(&system, &repaired, CLEARANCE) {
            complaints.push(format!("{point:?} -> {repaired:?}: no clearance"));
        }
        if !holds(&system, &repaired) {
            complaints.push(format!("{point:?} -> {repaired:?}: infeasible"));
        }
        if !in_box(&system, &repaired) {
            complaints.push(format!("{point:?} -> {repaired:?}: outside the box"));
        }
        let again = sojourn::repair(&system, anchors.as_ref(), &repaired, CLEARANCE);
        if again.as_deref() != Ok(repaired.as_slice()) {
            complaints.push(format!(
                "{point:?} -> {repaired:?} -> {again:?}: not a fixed point"
            ));
        }
        let twice = sojourn::repair(&system, anchors.as_ref(), &point, CLEARANCE);
        let same = twice.as_ref().is_ok_and(|twice| {
            twice
                .iter()
                .zip(&repaired)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        });
        if !same {
            complaints.push(format!(
                "{point:?} -> {repaired:?} then {twice:?}: not deterministic"
            ));
        }
        let moved = normalised_l1(&system, &point, &repaired);
        let nearest = clear_anchors
            .iter()
            .map(|anchor| normalised_l1(&system, &point, anchor))
            .fold(f64::INFINITY, f64::min);
        if moved > nearest {
            complaints.push(format!(
                "{point:?} -> {repaired:?}: moved {moved} where an anchor was {nearest} away"
            ));
        }
    }
    assert!(complaints.is_empty(), "{}", complaints.join("\n"));
}

#[test]
fn without_anchors_a_gap_is_not_crossed() {
    // Between the two bands, with nothing to bisect toward: no interval says
    // which way to go, so there is no honest answer, and `None` is the honest
    // answer.
    let system = system(
        variables(&[("x", -5.0, 5.0)]),
        &["(x + 2) * (x - 1) == 0 +/- 0.001"],
    );
    let none = Mat::<f64>::zeros(1, 0);

    assert_eq!(
        sojourn::repair(&system, none.as_ref(), &[0.0], 0.0),
        Err(RepairError::Stranded)
    );
}

#[test]
fn without_anchors_a_bound_is_still_reached() {
    // The clamp needs no anchor: the constraint itself says where the feasible
    // side is.
    let system = system(
        variables(&[("x1", -2.0, 2.0), ("x2", -2.0, 2.0)]),
        &["2*x1 + x2 < 1"],
    );
    let none = Mat::<f64>::zeros(2, 0);

    let repaired = sojourn::repair(&system, none.as_ref(), &[1.0, 1.0], 0.0)
        .expect("a half-space needs no anchor");
    assert!(
        holds(&system, &repaired),
        "{repaired:?} violates the half-space"
    );
}

#[test]
fn a_feasible_point_is_returned_untouched() {
    let system = system(
        variables(&[("x1", -2.0, 2.0), ("x2", -2.0, 2.0)]),
        &["2*x1 + x2 < 1"],
    );
    let none = Mat::<f64>::zeros(2, 0);
    let point = [-0.3, 0.7];

    assert_eq!(
        sojourn::repair(&system, none.as_ref(), &point, 0.0).as_deref(),
        Ok(point.as_slice())
    );
}

// ---- clearance: a deliberate step inside, not an ulp ----------------------

#[test]
fn a_half_space_is_entered_clear_of_its_wall() {
    // The steep-coordinate fixture again, asked for a thousandth of the box.
    // `x1` lands `CLEARANCE * 4` inside the wall at 0 rather than on it, `x2`
    // still does not move, and every axis neighbour at that distance holds.
    let system = system(
        variables(&[("x1", -2.0, 2.0), ("x2", -2.0, 2.0)]),
        &["2*x1 + x2 < 1"],
    );
    let anchors = anchors(&[&[0.0, 0.0]], 2);

    let repaired = sojourn::repair(&system, anchors.as_ref(), &[1.0, 1.0], CLEARANCE)
        .expect("a half-space is reachable");

    assert!(
        has_clearance(&system, &repaired, CLEARANCE),
        "{repaired:?} lacks the clearance"
    );
    assert_eq!(repaired[1], 1.0, "the cheap coordinate was left alone");
    let expected = -CLEARANCE * 4.0;
    assert!(
        (repaired[0] - expected).abs() < 1e-12,
        "x1 should land {expected} inside the wall at 0, got {}",
        repaired[0]
    );
}

#[test]
fn a_vertex_is_landed_clear_of_both_walls() {
    // Two walls meeting at (0.5, 0.5), approached from (1, 1). Each clamp is
    // its own axis projection, so the corner is the closed form: both
    // coordinates land the clearance inside their wall. This is the shape the
    // spring's optimum has, where a landing an ulp inside was found wanting.
    let system = system(
        variables(&[("x1", 0.0, 1.0), ("x2", 0.0, 1.0)]),
        &["x1 < 0.5", "x2 < 0.5"],
    );
    let anchors = anchors(&[&[0.25, 0.25]], 2);

    let repaired = sojourn::repair(&system, anchors.as_ref(), &[1.0, 1.0], CLEARANCE)
        .expect("a corner is reachable");

    assert!(
        has_clearance(&system, &repaired, CLEARANCE),
        "{repaired:?} lacks the clearance"
    );
    for (index, value) in repaired.iter().enumerate() {
        let expected = 0.5 - CLEARANCE;
        assert!(
            (value - expected).abs() < 1e-12,
            "x{} should land at {expected}, got {value}",
            index + 1
        );
    }
}

#[test]
fn a_chord_landing_is_backed_off() {
    // The two-bands fixture, which only the shotgun can answer, asked for a
    // clearance a tenth of a band's half-width: the chord lands on the band's
    // edge and the answer must be stepped inside it.
    let system = system(
        variables(&[("x", -5.0, 5.0)]),
        &["(x + 2) * (x - 1) == 0 +/- 0.001"],
    );
    let anchors = anchors(&[&[-2.0], &[1.0]], 1);
    // The band at 1 is about 0.00033 wide in `x`; a hundredth of that, over
    // the box's width of 10.
    let clearance = 0.000_033 / 10.0;

    let repaired =
        sojourn::repair(&system, anchors.as_ref(), &[0.9], clearance).expect("a band is reachable");

    assert!(
        has_clearance(&system, &repaired, clearance),
        "{repaired:?} lacks the clearance"
    );
    assert!(
        (repaired[0] - 1.0).abs() < 0.001,
        "from 0.9 the band at 1 is nearer, got {}",
        repaired[0]
    );
}

#[test]
fn a_slab_thinner_than_the_clearance_is_cramped() {
    // A slab `0.002` wide in a box `2` wide, asked for a clearance of a
    // hundredth of the box: `0.02` each side, ten times more room than the
    // slab has. Nothing can be handed back with that clearance, and the
    // honest answer names the nearest feasible point and says so.
    let system = system(variables(&[("x", -1.0, 1.0)]), &["x == 0 +/- 0.001"]);
    let anchors = anchors(&[&[0.0]], 1);

    let verdict = sojourn::repair(&system, anchors.as_ref(), &[0.5], 1e-2);

    match verdict {
        Err(RepairError::Cramped { nearest, clearance }) => {
            assert!(holds(&system, &nearest), "{nearest:?} is not even feasible");
            assert_eq!(clearance, 1e-2);
            assert!(
                nearest[0].abs() <= 0.001,
                "the nearest feasible point should be in the slab, got {}",
                nearest[0]
            );
        }
        other => panic!("expected Cramped, got {other:?}"),
    }
}

#[test]
fn a_point_with_clearance_is_returned_untouched() {
    let system = system(
        variables(&[("x1", -2.0, 2.0), ("x2", -2.0, 2.0)]),
        &["2*x1 + x2 < 1"],
    );
    let none = Mat::<f64>::zeros(2, 0);
    let point = [-0.3, 0.7];

    assert_eq!(
        sojourn::repair(&system, none.as_ref(), &point, CLEARANCE).as_deref(),
        Ok(point.as_slice())
    );
}

#[test]
fn a_feasible_point_without_clearance_is_moved_inward() {
    // Feasible by a hair — `2 * x1 + x2` is `1 - 1e-12` — is the fixed point
    // the caller sent back in, and with a clearance it is not returned as it
    // came: it steps inside the wall. Not necessarily along one axis: on a
    // wall every coordinate's clamp costs exactly the clearance, the tie goes
    // to rounding, and a coordinate that moved first without clearing the
    // other's neighbour is not always put back. What is promised is the
    // clearance, and a move of at most the clearance per coordinate.
    let system = system(
        variables(&[("x1", -2.0, 2.0), ("x2", -2.0, 2.0)]),
        &["2*x1 + x2 < 1"],
    );
    let none = Mat::<f64>::zeros(2, 0);
    let point = [0.0, 1.0 - 1e-12];
    assert!(holds(&system, &point), "the fixture should start feasible");
    assert!(!has_clearance(&system, &point, CLEARANCE));

    let repaired = sojourn::repair(&system, none.as_ref(), &point, CLEARANCE)
        .expect("a half-space needs no anchor");

    assert!(
        has_clearance(&system, &repaired, CLEARANCE),
        "{repaired:?} lacks the clearance"
    );
    let moved = normalised_l1(&system, &point, &repaired);
    assert!(
        moved <= 2.0 * CLEARANCE + 1e-12,
        "{point:?} -> {repaired:?} moved {moved}, more than the clearance per coordinate"
    );
    for (index, (before, after)) in point.iter().zip(&repaired).enumerate() {
        assert!(
            after <= before,
            "x{} should only ever step inward, {before} -> {after}",
            index + 1
        );
    }
}

#[test]
fn a_box_bound_is_a_wall_too() {
    // A constraint that never binds; the box's own edge is the only wall. A
    // point on it has no room to move outward, so it is moved the clearance
    // inside — the caller's round trip can miss the edge by an ulp as well.
    let system = system(variables(&[("x", 0.0, 1.0)]), &["x > -1"]);
    let none = Mat::<f64>::zeros(1, 0);

    let repaired = sojourn::repair(&system, none.as_ref(), &[1.0], CLEARANCE)
        .expect("the box's inside is reachable");

    assert!(
        has_clearance(&system, &repaired, CLEARANCE),
        "{repaired:?} lacks the clearance"
    );
    assert!(
        (repaired[0] - (1.0 - CLEARANCE)).abs() < 1e-12,
        "x should step to {}, got {}",
        1.0 - CLEARANCE,
        repaired[0]
    );
}
