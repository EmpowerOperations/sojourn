//! Constraint-solving pool cases, ported from `Z3SolvingPoolFixture.kt`.
//!
//! The JVM harness asserts a *property*, not a value: ask for ten points, get
//! ten, and every one satisfies every constraint. That is strategy-agnostic, so
//! the same cases run against whatever the pool is currently made of and
//! red-versus-green tracks *capability* rather than feature-completeness.
//!
//! Which is why these split by constraint shape:
//!
//! * **Inequalities are samplable.** `20 > 2^x5` admits about 43% of its range,
//!   so rejection sampling finds points immediately.
//! * **Equality-with-tolerance is not.** `x1 == sqrt(x2) +/- 0.0001` is a
//!   measure-zero ribbon that uniform sampling will essentially never land on.
//!   Those stay red until a solver is wired up — that is the honest picture, not
//!   a gap in the port.
//!
//! All sixteen cases from the fixture are here. Several of them exercise babel
//! features — `ln`, `%`, `sgn`, `var[i]` — that nothing else puts through a
//! pool, which is worth more than the solver coverage they were written for.
//!
//! Deliberately not ported: `Z3Fixture` and `Z3ExtensionsFixture` test the Z3
//! API we are not using; `LanguageFixture` is half JVM sanity checks and half
//! decimal-to-rational conversion that belongs with the SMT emitter;
//! `IntegrationTests` asserts a list equals an integer and calls `.all()`
//! without a terminal assertion, so it either always fails or asserts nothing.

mod common;

use faer::Mat;

use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;
use sojourn::{ConstraintSolver, ConstraintSystem, Infeasibility, InputVariable, Status};

/// A fixture's [`ConstraintSystem`]; one that does not bind is the test's error.
fn system(variables: Vec<InputVariable>, constraints: &[&str]) -> anyhow::Result<ConstraintSystem> {
    Ok(ConstraintSystem::new(
        variables,
        constraints.iter().copied(),
    )?)
}

fn variables(specs: &[(&str, f64, f64)]) -> Vec<InputVariable> {
    specs
        .iter()
        .map(|(name, low, high)| InputVariable::new(*name, *low, *high))
        .collect()
}

/// A sample matrix back as one `Vec<f64>` per point.
///
/// The pool speaks in matrices because that is what the evaluator eats, but
/// almost every assertion here is about *a point* — its coordinates, its
/// residual, its position in a distribution. Converting once at the boundary
/// keeps those assertions saying what they mean instead of indexing `(row,
/// column)` pairs.
fn columns(samples: &Mat<f64>) -> Vec<Vec<f64>> {
    (0..samples.ncols())
        .map(|column| {
            (0..samples.nrows())
                .map(|row| samples[(row, column)])
                .collect()
        })
        .collect()
}

/// Pinned so a failure is reproducible. The JVM version faked this with
/// `OneHundredBraindeadPoints`, a hard-coded array of 100 doubles.
const SEED: u64 = 0x50_50_1E_5E_ED;

const REQUESTED: usize = 10;

/// The oracle the corpus tests share: ten points came back, every one inside
/// the box, every one satisfying every source — re-checked independently of
/// the pool, which filters its own output, and a test that trusts the thing
/// it is testing is not a test.
fn assert_ten_feasible(system: &ConstraintSystem, sources: &[&str], points: &[Vec<f64>]) {
    assert_eq!(
        points.len(),
        REQUESTED,
        "wanted {REQUESTED} points, got {}",
        points.len()
    );
    for point in points {
        for (variable, value) in system.variables().iter().zip(point) {
            assert!(
                variable.contains(*value),
                "{} = {value} is outside {}..={}",
                variable.name,
                variable.lower_bound,
                variable.upper_bound
            );
        }
        let bindings: Vec<(&str, f64)> = system
            .variables()
            .iter()
            .map(|v| v.name.as_str())
            .zip(point.iter().copied())
            .collect();
        for source in sources {
            let residual = common::eval_one(source, &bindings)
                .unwrap_or_else(|e| panic!("evaluating {source:?} at {point:?}: {e}"));
            // Matching the JVM harness's tolerance: a solver-produced point can
            // sit a hair outside, where a sampled one never does.
            assert!(
                residual <= 1e-10,
                "{point:?} fails {source:?} (residual {residual})"
            );
        }
    }
}

/// `Unknown` is a claim about what we *know*, not about what we can deliver.
///
/// This case was written expecting the system to come up empty: the band is one
/// part in a million of the box, far too thin to sample, and `sin` is outside
/// what the emitter will put to a solver. It does not come up empty, and the
/// reason is worth keeping.
///
/// With `sin` refused, the document Z3 receives contains only the bounds — so it
/// returns some arbitrary point satisfying those, near the origin. And
/// `sin(0) = 0`, so that point happens to sit *on the curve*. Hit-and-run seeds
/// from it and walks **along** the curve, since shrinkage converges onto the
/// feasible piece containing the current point however thin that piece is.
///
/// So the pool delivers real points for a constraint nothing in the pipeline can
/// reason about, and `Satisfied` is the honest answer: points exist and are in
/// hand. There used to be a third verdict for this — `Unknown` — reporting the
/// epistemic state, and it is gone because the state it described is not one a
/// caller can act on. What replaces it is the `tracing` line the emitter now
/// writes naming the constraint and why it could not be expressed, so the
/// information is still reported rather than dropped, which is the whole
/// difference from the JVM version.
///
/// The known weakness is *coverage*, not correctness: the points cluster around
/// wherever the solver's arbitrary model landed, and no fairness oracle applies,
/// because the region has no closed form and rejection sampling cannot reach it
/// to serve as a reference.
#[pollster::test]
async fn a_constraint_nothing_can_reason_about_still_yields_points_and_says_so()
-> anyhow::Result<()> {
    let source = "y == sin(x) +/- 0.000001";
    let inputs = vec![
        InputVariable::new("x", -1.0, 1.0),
        InputVariable::new("y", -1.0, 1.0),
    ];

    let mut pool = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system(inputs.clone(), &[source])?)
        .await
        .expect("solving should not fail");

    // The points are real. Checked here rather than trusted, because the pool
    // filtering its own output is the thing under test: the emitter never put
    // this constraint to a solver, so nothing but the filter stands between a
    // proposed point and the caller.
    let points = columns(&pool.take(5));
    assert_eq!(points.len(), 5, "status {:?}", pool.status());
    for point in &points {
        let bindings = [("x", point[0]), ("y", point[1])];
        let residual = common::eval_one(source, &bindings).expect("evaluation should not fail");
        assert!(residual <= 0.0, "{point:?} does not satisfy {source:?}");
    }
    Ok(())
}

// ------------------------------------------------ the background worker

/// A pool that will never produce another point must *say so*, not wait.
///
/// This is the failure mode worth fearing now that filling happens on a worker
/// thread: every other bug here shows up as an assertion, but confusing "nothing
/// yet" with "nothing ever" makes `generate` block for points that are not
/// coming, and the suite stops finishing rather than failing. Hence the
/// `terminate-after` in `.config/nextest.toml`, and hence this test existing
/// before the ones that merely check numbers.
///
/// The constraints contradict each other through `%`, which the emitter cannot
/// express — so the solver cannot rule the region out and the pool is left
/// genuinely searching for something that is not there.
#[pollster::test]
async fn a_pool_that_can_never_deliver_reports_exhausted_rather_than_blocking() -> anyhow::Result<()>
{
    let verdict = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system(
            vec![InputVariable::new("x1", 0.0, 10.0)],
            &["x1 % 3.0 >= 2", "x1 % 3.0 <= 1"],
        )?)
        .await;

    let mut pool = match verdict {
        Ok(region) => region,
        // Also a fine answer, and it would mean the emitter grew `%`.
        Err(because) => {
            assert!(matches!(because, Infeasibility::Proved { ref blamed } if !blamed.is_empty()));
            return Ok(());
        }
    };

    // If exhaustion is broken this call never comes back.
    let points = columns(&pool.take(10));

    assert!(
        points.is_empty(),
        "found points in an empty region: {points:?}"
    );
    assert_eq!(
        pool.status(),
        Status::Exhausted,
        "the pool is still calling itself busy"
    );
    Ok(())
}

/// Dropping a pool has to join its worker, and the join has to not deadlock.
///
/// `Drop` runs before the pool's fields do, so the receiving end of the channel
/// is still alive while we wait — which means a worker parked on a full channel
/// stays parked unless the drop drains it first. Getting that wrong hangs here.
///
/// Asserting termination rather than latency on purpose: a deadline would flake
/// on a loaded CI box, and the timeout in `.config/nextest.toml` already turns a
/// genuine hang into a legible failure.
#[pollster::test]
async fn dropping_a_pool_mid_fill_does_not_deadlock() -> anyhow::Result<()> {
    let mut pool = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system(
            vec![
                InputVariable::new("x1", 0.0, 10.0),
                InputVariable::new("x2", 0.0, 10.0),
            ],
            &["x1 < x2"],
        )?)
        .await
        .expect("solving should not fail");

    // Take one batch's worth and leave the worker mid-stride, most likely parked
    // against a full channel, which is the case that deadlocks if `Drop` waits
    // without draining.
    assert!(!columns(&pool.take(1)).is_empty());
    drop(pool);
    Ok(())
}

/// Points arrive in the same order however the worker happened to be scheduled.
///
/// The whole determinism argument for putting the search on a thread: one
/// worker, one seeded generator, so *timing* varies between runs but the
/// *sequence* does not. If this ever fails, every seeded expectation in the
/// suite is resting on luck.
#[pollster::test]
async fn the_same_seed_delivers_the_same_points() -> anyhow::Result<()> {
    let mut runs = Vec::new();
    for _ in 0..2 {
        let mut pool = ConstraintSolver::new()
            .with_proposal_budget(common::PROPOSAL_BUDGET)
            .with_gpu(false)
            .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
            .solve(&system(
                vec![
                    InputVariable::new("x1", 0.0, 10.0),
                    InputVariable::new("x2", 0.0, 10.0),
                ],
                &["x1 < x2"],
            )?)
            .await
            .expect("solving should not fail");
        runs.push(columns(&pool.take(500)));
    }

    assert_eq!(runs[0].len(), 500);
    assert_eq!(runs[0], runs[1], "the same seed produced different points");
    Ok(())
}

// ------------------------------------------- only a solver can say this

#[pollster::test]
async fn contradictory_constraints_are_reported_as_unsatisfiable() -> anyhow::Result<()> {
    // `x > 8` and `x < 2` cannot both hold. Sampling cannot tell that apart from
    // "I did not find one" — it looks identical from the outside — so this is
    // the one path in the whole crate that can produce `Unsatisfiable`, and it
    // exists only because a solver is wired up.
    let solution = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system(
            vec![InputVariable::new("x", 0.0, 10.0)],
            &["x > 8", "x < 2"],
        )?)
        .await;

    let Err(because) = solution else {
        panic!("expected Unsatisfiable, got {solution:?}");
    };

    let Infeasibility::Proved { blamed } = because else {
        panic!("a plain contradiction should be proved, not merely unfound");
    };

    // Both, because a contradiction is a relationship: either constraint alone
    // is perfectly satisfiable, and naming one would be picking arbitrarily.
    let mut sources: Vec<&str> = blamed.iter().map(|c| c.source.as_str()).collect();
    sources.sort_unstable();
    assert_eq!(sources, vec!["x < 2", "x > 8"]);
    Ok(())
}

/// The mistake a user actually makes: three constraints that describe a
/// region, and a fourth with its comparison backwards. Each holds on its own
/// and the four together hold nothing, which sampling cannot distinguish from
/// bad luck and a local solve cannot distinguish from a bad start. What a
/// user needs back is the *set* that conflicts, so they know which line to
/// look at — which is the unsat core, and the one thing only a solver gives.
/// Measured 2026-09-13: Z3 blames exactly `x1 < 3`, `x2 > 5`, `x1 > x2`, and
/// the message reads "no point satisfies these constraints together:
/// `x1 < 3`, `x2 > 5`, `x1 > x2`". This is the test the question "does Z3
/// earn its build" is answered against; see `docs/todo.md`.
#[pollster::test]
async fn a_backwards_comparison_is_blamed_together_with_what_it_contradicts() -> anyhow::Result<()>
{
    let system = system(
        vec![
            InputVariable::new("x1", 0.0, 10.0),
            InputVariable::new("x2", 0.0, 10.0),
        ],
        &[
            "x1 < 3",
            "x2 > 5",
            "x1 + x2 < 20",
            // Meant `x1 < x2`, and every one of the three above agrees with
            // that; this one contradicts the first two.
            "x1 > x2",
        ],
    )?;
    let verdict = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await;

    let Err(because) = verdict else {
        panic!("expected Unsatisfiable, got {verdict:?}");
    };
    let Infeasibility::Proved { blamed } = &because else {
        panic!("a linear contradiction should be proved, not merely unfound: {because}");
    };

    let sources: Vec<&str> = blamed.iter().map(|c| c.source.as_str()).collect();
    assert!(
        sources.contains(&"x1 > x2"),
        "the backwards comparison should be in the blame: {sources:?}"
    );
    assert!(
        !sources.contains(&"x1 + x2 < 20"),
        "a constraint that takes no part in the contradiction should not be blamed: {sources:?}"
    );
    // What the user reads.
    assert_eq!(
        because.to_string(),
        format!(
            "no point satisfies these constraints together: {}",
            sources
                .iter()
                .map(|s| format!("`{s}`"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    );
    Ok(())
}

#[pollster::test]
async fn a_satisfiable_problem_is_not_blamed_on_anything() -> anyhow::Result<()> {
    // The other half of the above: the machinery has to stay quiet when there is
    // nothing wrong, or an `Unsatisfiable` means nothing.
    let solution = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system(
            vec![InputVariable::new("x", 0.0, 10.0)],
            &["x > 8", "x < 9"],
        )?)
        .await;
    assert!(solution.is_ok(), "got {solution:?}");
    Ok(())
}

// ------------------------------------------------- samplable: inequalities

#[pollster::test]
async fn power_with_variable_as_exponent() -> anyhow::Result<()> {
    // 2^x5 < 20 means x5 < log2(20) ~ 4.32, so ~43% of the range.
    // The JVM comment reads "nope, Z3 wont reason about real-exponents" —
    // rejection sampling has no such trouble.
    let sources = &["20 > 2^x5"];
    let system = system(variables(&[("x5", 0.0, 10.0)]), sources)?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn a_deeply_transcendental_constraint() -> anyhow::Result<()> {
    // `x1 > sin(ln(cos(2.1^x1)))`. Feasible for x1 below about 0.61, where
    // cos(2.1^x1) is still positive. The JVM name for this was "should simply
    // drop provided expression" — it could not transcode it at all.
    let sources = &["x1 > sin(ln(cos(2.1^x1)))"];
    let system = system(variables(&[("x1", 0.0, 1.0), ("x2", 0.0, 1.0)]), sources)?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn sine_over_multiple_periods() -> anyhow::Result<()> {
    // Ported with its assertion *inverted*. The JVM version asserted the
    // infeasible results were `isNotEmpty()`, pinning the fact that its
    // Taylor-series `sin` produced points that did not satisfy the constraint.
    // A correct pool returns feasible points.
    let sources = &["y > sin(theta)"];
    let system = system(
        variables(&[
            ("theta", std::f64::consts::PI, std::f64::consts::PI * 3.0),
            ("y", -1.0, 1.0),
        ]),
        sources,
    )?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn a_simple_inequality() -> anyhow::Result<()> {
    // Ours, not the fixture's: the simplest possible two-variable constraint,
    // here so that a failure everywhere else has something trivial to be
    // contrasted against.
    let sources = &["x1 < x2"];
    let system = system(variables(&[("x1", 0.0, 10.0), ("x2", 0.0, 10.0)]), sources)?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn logarithms() -> anyhow::Result<()> {
    // `2 < ln(x1)` is `x1 > e^2`, about 26% of the range. The JVM case has two
    // further constraints commented out — `x4 == log(4) +/- 0.0001` and
    // `x6 > log(2.0, x5)` — so those are left out here too rather than invented.
    // `x2` is declared and unused, exactly as over there: a schema may be wider
    // than the constraints that reference it.
    let sources = &["2 < ln(x1)"];
    let system = system(variables(&[("x1", 0.0, 10.0), ("x2", 0.0, 10.0)]), sources)?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn modulo_with_a_symbolic_divisor() -> anyhow::Result<()> {
    // `10 % x1` where the divisor is the variable. Note `x1 = 0` gives NaN, and
    // a NaN residual is not a pass — so this also pins that the pool rejects
    // rather than propagates it.
    let sources = &["3 > 10 % x1"];
    let system = system(variables(&[("x1", 0.0, 10.0)]), sources)?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn equality_with_a_loose_tolerance() -> anyhow::Result<()> {
    // The tolerance is what decides whether an equality is samplable. At
    // `+/- 0.1` on a 2x2 square the band is about 9.75% of the area, so this
    // goes green while every other equality case in this file does not — the
    // difference is measure, not kind.
    let sources = &["x1 == x2 +/- 0.1"];
    let system = system(variables(&[("x1", -1.0, 1.0), ("x2", -1.0, 1.0)]), sources)?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[expect(
    clippy::approx_constant,
    reason = "the fixture's bounds are literally -3.14..3.14, a truncation rather \
         than an attempt at pi — and the difference shows at the endpoints, \
         where sin(3.14) is 0.0016 and sin(pi) is zero"
)]
#[pollster::test]
async fn sine_below_zero() -> anyhow::Result<()> {
    // Half the range of x1. `y` is unused by the constraint.
    let sources = &["sin(x1) <= 0"];
    let system = system(variables(&[("x1", -3.14, 3.14), ("y", 0.9, 1.0)]), sources)?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

// ------------------------------ needs a solver: equality with tolerance

#[pollster::test]
async fn simple_arithmetic() -> anyhow::Result<()> {
    // Was `x2 == x1 + 1/2*x2 - x3/x4`, which is the same set written
    // circularly and is now refused at construction — see
    // `SystemError::Cyclic`. Rearranged rather than dropped, so the
    // fixture still exercises what it was ported for.
    let sources = &["1/2*x2 - x1 + x3 / x4 == 0 +/- 0.00001"];
    let system = system(
        variables(&[
            ("x1", 0.0, 1.0),
            ("x2", 0.0, 1.0),
            ("x3", 0.0, 1.0),
            ("x4", 0.0, 1.0),
        ]),
        sources,
    )?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn roots() -> anyhow::Result<()> {
    let sources = &["x1 == sqrt(x2) +/- 0.0001", "x3 == cbrt(x4) +/- 0.0001"];
    let system = system(
        variables(&[
            ("x1", 0.0, 10.0),
            ("x2", 0.0, 10.0),
            ("x3", 0.0, 10.0),
            ("x4", 0.0, 10.0),
        ]),
        sources,
    )?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn power() -> anyhow::Result<()> {
    let sources = &["x1 == x2^3 +/- 0.0001"];
    let system = system(variables(&[("x1", 0.0, 10.0), ("x2", 0.0, 10.0)]), sources)?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn absolute_value() -> anyhow::Result<()> {
    // Three variables, each pinned to a magnitude from a different side of zero:
    // x1 from the positive range, x2 from the negative, x3 from a range that
    // excludes the answer's sign entirely. Bands of a thousandth in ranges of
    // one, so about two parts in a billion once combined.
    let sources = &[
        "abs(x1) == 1 +/- 0.001",
        "abs(x2) == 1 +/- 0.001",
        "abs(x3) == 1.5 +/- 0.001",
    ];
    let system = system(
        variables(&[("x1", 0.0, 1.0), ("x2", -1.0, 0.0), ("x3", -2.0, -1.0)]),
        sources,
    )?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn modulo() -> anyhow::Result<()> {
    // Two constraints of different shapes, as the JVM case had them.
    // `x1 % 3.0 >= 2` alone is samplable — a third of the range — but
    // `x3 == x4 % 4.5 +/- 0.0001` is a curve of width 0.0002, and a test is only
    // as green as its hardest constraint. Kept together rather than split, so
    // that what goes green when the solver lands is the case as written.
    let sources = &["x1 % 3.0 >= 2", "x3 == x4 % 4.5 +/- 0.0001"];
    let system = system(
        variables(&[
            ("x1", 0.0, 10.0),
            ("x2", 0.0, 10.0),
            ("x3", 0.0, 10.0),
            ("x4", 0.0, 10.0),
        ]),
        sources,
    )?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn constants() -> anyhow::Result<()> {
    // Two bands 0.002 wide in a 10x10 box: about four parts in a hundred
    // million. Sampling is not going to stumble onto pi.
    let sources = &["x1 == pi +/- 0.001", "x2 == e +/- 0.001"];
    let system = system(variables(&[("x1", 0.0, 10.0), ("x2", 0.0, 10.0)]), sources)?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn signum() -> anyhow::Result<()> {
    // `sgn` is a step, so x2 has to land within 0.001 of exactly -1 or +1 — two
    // slivers of a range four wide. Also the only place `sgn` meets a pool, and
    // worth having for that alone: Java's `Math.signum` and Rust's `f64::signum`
    // disagree about zero and NaN.
    let sources = &["x2 == sgn(x1) +/- 0.001"];
    let system = system(variables(&[("x1", -1.0, 1.0), ("x2", -2.0, 2.0)]), sources)?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn dynamic_variable_lookup() -> anyhow::Result<()> {
    // The only exercise of `var[i]` under a pool anywhere in the suite. Red for
    // its measure rather than its subject — but it still proves the indexed form
    // compiles, binds against a schema, and evaluates through the pool, which is
    // the part that would otherwise go untested until a solver arrived.
    let sources = &[
        "1.5 == var[1] + var[2] +/- 0.001",
        "1.5 == var[2] - var[3] +/- 0.001",
    ];
    let system = system(
        variables(&[("x1", -1.0, 1.0), ("x2", -2.0, 2.0), ("x3", -2.0, 2.0)]),
        sources,
    )?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}

#[pollster::test]
async fn ceiling_and_floor() -> anyhow::Result<()> {
    let sources = &["x1 > floor(x2)", "x3 > ceil(x4) + floor(x4)"];
    let system = system(
        variables(&[
            ("x1", 0.0, 10.0),
            ("x2", 0.0, 10.0),
            ("x3", 0.0, 10.0),
            ("x4", 0.0, 10.0),
        ]),
        sources,
    )?;
    let mut region = ConstraintSolver::new()
        .with_proposal_budget(common::PROPOSAL_BUDGET)
        .with_gpu(false)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
        .solve(&system)
        .await?;
    let points = columns(&region.take(REQUESTED));

    assert_ten_feasible(&system, sources, &points);
    Ok(())
}
