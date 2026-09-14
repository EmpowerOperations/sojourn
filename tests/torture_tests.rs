//! Constraints built to defeat the solver, so the pool has to survive without it.
//!
//! A real literal exponent is the one power shape every backend refuses to turn
//! into multiplication and every solver refuses outright: `x^1.234` is
//! `exp(1.234 * ln x)`, Z3 has no `exp`, and handing Z3 its own `^` is worse
//! than a refusal — it runs minutes past its rlimit on exactly this exponent
//! (`docs/todo.md`, "Z3 holes"). So the emitter leaves the constraint out of
//! the document, the relaxed document is satisfiable trivially, its witness
//! fails the real constraint, and the pool retreats to sampling: `powf` on
//! every CPU lane and `pow` on the GPU when the sieve is compiled in, both of
//! them bound by the special-function hardware rather than by arithmetic.
//!
//! The contract these pin: sampling finds what is there to be found, and what
//! it cannot find is reported as *not found* — never as proved empty, since
//! nothing could prove it — with the sentence a caller can act on.

mod common;

use anyhow::Context;
use faer::Mat;
use sojourn::{
    ConstraintSolver, ConstraintSystem, FeasibleRegion, Infeasibility, InputVariable, Strategy,
};

/// Same value the other cvg suites use, so a point seen in one is the point
/// seen in another.
const SEED: u64 = 0x50_50_1E_5E_ED;

/// A fixture's [`ConstraintSystem`]; one that does not bind is the test's error.
fn system(
    variables: &[(&str, f64, f64)],
    constraints: &[&str],
) -> anyhow::Result<ConstraintSystem> {
    let variables = variables
        .iter()
        .map(|(name, low, high)| InputVariable::new(*name, *low, *high))
        .collect();
    Ok(ConstraintSystem::new(
        variables,
        constraints.iter().copied(),
    )?)
}

/// Whether every constraint holds at `point`, judged independently of the
/// pool through the public evaluator, strictly: the residual must be `<= 0`.
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

fn in_box(system: &ConstraintSystem, point: &[f64]) -> bool {
    system
        .variables()
        .iter()
        .zip(point)
        .all(|(variable, value)| (variable.lower_bound..=variable.upper_bound).contains(value))
}

/// The solver every fixture here starts from: the test-sized budget, the CPU
/// alone so the verdict is a function of the seed, and the seed.
fn solver() -> ConstraintSolver {
    ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_seed(SEED)
}

/// The region a solve returns, which is where `repair` lives. The fixture
/// below hands it its own anchor, so the census is not what is being tested.
async fn region(system: &ConstraintSystem) -> anyhow::Result<FeasibleRegion> {
    solver()
        .solve(system)
        .await
        .context("the fixture should be satisfiable")
}

/// A curve `x1^1.234 + x2^1.234 == 5` thickened to a band a hundredth wide in
/// a hundred-unit box: about one proposal in two thousand lands, which is
/// well inside brute force's reach and far outside the probe's luck. The
/// solver can say nothing about it, so nothing here was proved; every point
/// was sampled or walked from a sampled or locally solved seed.
#[pollster::test]
async fn a_thin_curve_is_found_without_the_solver_helping() -> anyhow::Result<()> {
    const WANTED: usize = 10;

    let system = system(
        &[("x1", 0.0, 10.0), ("x2", 0.0, 10.0)],
        &["x1^1.234 + x2^1.234 == 5 +/- 0.01"],
    )?;
    let mut samples = solver()
        .solve(&system)
        .await
        .context("a curve that sampling can reach should be found")?;

    let points = samples.take(WANTED);
    assert_eq!(points.ncols(), WANTED, "the stream ended early");
    for column in 0..points.ncols() {
        let point: Vec<f64> = (0..points.nrows())
            .map(|row| points[(row, column)])
            .collect();
        assert!(holds(&system, &point), "{point:?} is off the curve");
        assert!(in_box(&system, &point), "{point:?} is outside the box");
    }
    Ok(())
}

/// `x^1.234` never reaches a million on `[0, 10]`, but nothing can prove that:
/// the solver was not allowed to see the constraint, and sampling can only
/// report that it found nothing. The verdict has to say exactly that, name
/// the constraint no solver could be asked about, and stop short of calling
/// the region empty.
///
/// This spends brute force's whole budget — a billion `powf` calls across
/// every thread in release, about ten seconds — because giving up early
/// would be the bug.
#[pollster::test]
async fn what_sampling_cannot_find_is_reported_not_proved() -> anyhow::Result<()> {
    let source = "x1^1.234 > 1000000";
    let system = system(&[("x1", 0.0, 10.0)], &[source])?;
    let verdict = solver().solve(&system).await;

    let Err(because) = verdict else {
        panic!("a point beyond the box was reported {verdict:?}");
    };
    let sentence = because.to_string();
    let Infeasibility::NotFound { unexpressed } = because else {
        panic!("nothing could have proved this empty, yet it was reported {because:?}");
    };
    let named: Vec<&str> = unexpressed.iter().map(|c| c.source.as_str()).collect();
    assert_eq!(named, vec![source]);
    assert!(
        sentence.contains("sampling found nothing") && sentence.contains(source),
        "the verdict should read as a sentence a caller can act on, got {sentence:?}"
    );
    Ok(())
}

/// The same band, seeded by the local solve alone: a real exponent is outside
/// every solver's theories and a band this thin is outside the probe's luck,
/// but a point on it is an ordinary constrained optimisation from the box
/// centre, and that is what the local solve is for.
#[pollster::test]
async fn a_thin_curve_is_seeded_by_the_local_solve() -> anyhow::Result<()> {
    let system = system(
        &[("x1", 0.0, 10.0), ("x2", 0.0, 10.0)],
        &["x1^1.234 + x2^1.234 == 5 +/- 0.01"],
    )?;
    let mut region = solver()
        .with_strategies(vec![Strategy::LocalSolve, Strategy::HitAndRun])
        .solve(&system)
        .await
        .context("the band is not empty")?;
    let points = region.take(4);
    assert_eq!(
        points.ncols(),
        4,
        "the walker should carry on from the seed"
    );
    for column in 0..points.ncols() {
        let point: Vec<f64> = (0..points.nrows())
            .map(|row| points[(row, column)])
            .collect();
        assert!(holds(&system, &point), "{point:?} is off the curve");
    }
    Ok(())
}

/// Repair has no interval to clamp to either — narrowing declines a real
/// exponent as every solver does — so the chord from the anchor is all it has,
/// and that is enough: the point lands inside, judged by the evaluator alone.
/// The clearance has to come from the chord too, since no slice can say where
/// the wall is to step off it: the landing is backed off along the chord
/// until the axis neighbours pass.
#[pollster::test]
async fn repair_lands_without_an_interval_to_clamp_to() -> anyhow::Result<()> {
    const CLEARANCE: f64 = 1e-3;
    let system = system(
        &[("x1", 0.0, 10.0), ("x2", 0.0, 10.0)],
        &["x1^1.234 + x2^1.234 < 5"],
    )?;
    let anchors = Mat::from_fn(2, 1, |_, _| 0.0);

    let region = region(&system).await?;
    let repaired = region
        .repair(anchors.as_ref(), &[9.0, 9.0], CLEARANCE)
        .context("the origin is feasible, so something is reachable")?;

    assert!(
        holds(&system, &repaired),
        "{repaired:?} is outside the region"
    );
    assert!(
        in_box(&system, &repaired),
        "{repaired:?} is outside the box"
    );
    for coordinate in 0..2 {
        for sign in [-1.0, 1.0] {
            let mut neighbour = repaired.clone();
            neighbour[coordinate] += sign * CLEARANCE * 10.0;
            assert!(
                holds(&system, &neighbour) && in_box(&system, &neighbour),
                "{repaired:?} lacks the clearance: {neighbour:?} is outside"
            );
        }
    }
    Ok(())
}
