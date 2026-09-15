//! Regressions reported by callers, kept in the form they arrived in.
//!
//! Each case is a bug report: the system, the seed, and what the caller saw.
//! They are pinned to seeds on purpose — a regression that depends on the
//! probe's luck is only reproducible with the luck held still — and they stay
//! here after the fix so the luck can never turn again.

mod common;

use sojourn::{ConstraintSolver, ConstraintSystem, InputVariable, Status, Strategy};

/// Artemis, 2026-09-11. `x1 == x2 + 1 +/- 0.01` over `[-32.768, 32.768]^20`:
/// a 0.02-wide slab in a 65.5-wide box, one driven variable, the rest free.
/// Seeds 0, 1 and 3..=9 streamed 1500 points. Seed 2 reported `Satisfied`,
/// delivered exactly 25 points, and ended in `Status::Exhausted`.
///
/// The pool used to pick a route on the probe, one batch, and on seed 2 it
/// landed enough hits to choose plain sampling for a region whose true rate
/// is a third of the threshold. The delivery batches then came back empty
/// three times in a row, which the fill loop read as the region running dry.
/// A connected region with 25 known feasible points cannot run dry. There is
/// no route now: every batch is sampled first and walked for the rest.
mod a_lucky_probe_must_not_strand_the_sampling_route {
    use super::*;

    const DIM: usize = 20;
    const HALF_WIDTH: f64 = 32.768;
    const SLAB: &str = "x1 == x2 + 1 +/- 0.01";
    const WANTED: usize = 1500;

    fn system() -> anyhow::Result<ConstraintSystem> {
        let inputs: Vec<InputVariable> = (1..=DIM)
            .map(|i| InputVariable::new(format!("x{i}"), -HALF_WIDTH, HALF_WIDTH))
            .collect();
        Ok(ConstraintSystem::new(inputs, [SLAB])?)
    }

    #[pollster::test]
    async fn seed_2_streams_the_whole_request() -> anyhow::Result<()> {
        let mut region = ConstraintSolver::new()
            .with_seed(2)
            .solve(&system()?)
            .await?;
        let got = region.take(WANTED).ncols();

        assert_eq!(
            got,
            WANTED,
            "seed 2 delivered {got} of {WANTED} and ended in {:?}; a connected slab \
             with points in hand should never exhaust",
            region.status()
        );
        assert_eq!(region.status(), Status::Filling);
        Ok(())
    }

    #[pollster::test]
    async fn every_other_seed_streams_the_same_slab() -> anyhow::Result<()> {
        for seed in (0..10u64).filter(|s| *s != 2) {
            let mut region = ConstraintSolver::new()
                .with_seed(seed)
                .solve(&system()?)
                .await?;
            let got = region.take(WANTED).ncols();
            assert_eq!(
                got,
                WANTED,
                "seed {seed} ended early in {:?}",
                region.status()
            );
        }
        Ok(())
    }
}

/// Artemis, 2026-09-11, `e03-spring-3`. The tension/compression spring (Arora
/// 1989 via Coello 2000): `d` wire diameter, `D` coil diameter, `N` active
/// coils, four constraints. Its optimum is a vertex where the deflection and
/// shear-stress constraints are both active, and `repair` of a point just
/// outside it came back with both residuals at about `-1e-15`: feasible, by
/// less than the rounding of one arithmetic operation on the coordinates.
///
/// Artemis stores points normalised to `[-1, 1]` per coordinate and hands the
/// evaluator the denormalised value, `lo + (((x - lo) / (hi - lo) * 2 - 1) + 1)
/// / 2 * (hi - lo)`. That round trip is exact for most values and one ulp off
/// for some, and one ulp on `d` moves the two residuals by about `1e-8` (`d`
/// enters as `d^4` in a term of order `1e5`) — seven orders of magnitude more
/// than the margin `repair` left. A margin of a few ulps of the coordinate is
/// not the "final deliberate step inward" the contract asks for at a vertex;
/// it has to be large enough that any one-ulp perturbation stays feasible,
/// which is what the clearance argument now is.
mod repair_lands_too_close_at_a_vertex {
    use super::*;
    use anyhow::Context;
    use faer::Mat;
    use sojourn::{FeasibleRegion, compile};

    const NAMES: [&str; 3] = ["d", "D", "N"];
    const LO: [f64; 3] = [0.05, 0.25, 2.0];
    const HI: [f64; 3] = [2.0, 1.3, 15.0];
    const CONSTRAINTS: [&str; 4] = [
        "1 - (D^3) * N / (71785 * (d^4)) < 0",
        "(4 * (D^2) - d * D) / (12566 * (D * (d^3) - (d^4))) + 1 / (5108 * (d^2)) - 1 < 0",
        "1 - 140.45 * d / ((D^2) * N) < 0",
        "(D + d) / 1.5 - 1 < 0",
    ];
    /// A few thousand ulps of the unit cube, which is what `repair`'s doc
    /// recommends for a caller that normalises and back.
    const CLEARANCE: f64 = 1e-12;

    fn system() -> anyhow::Result<ConstraintSystem> {
        let vars = NAMES
            .iter()
            .zip(LO)
            .zip(HI)
            .map(|((n, lo), hi)| InputVariable::new(*n, lo, hi))
            .collect();
        Ok(ConstraintSystem::new(
            vars,
            CONSTRAINTS.iter().map(|s| (*s).to_string()),
        )?)
    }

    /// Every constraint's residual at `x`, through the public evaluator: the
    /// same numbers the user's evaluator would see.
    fn residuals(x: &[f64]) -> anyhow::Result<Vec<f64>> {
        let sample = Mat::from_fn(3, 1, |i, _| x[i]);
        CONSTRAINTS
            .iter()
            .map(|src| Ok(compile(src, &NAMES)?.eval(sample.as_ref())?[0]))
            .collect()
    }

    fn worst(residuals: &[f64]) -> f64 {
        residuals.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Artemis's frame round trip, per coordinate.
    fn round_trip(x: &[f64]) -> Vec<f64> {
        (0..3)
            .map(|i| {
                let n = (x[i] - LO[i]) / (HI[i] - LO[i]) * 2.0 - 1.0;
                LO[i] + (n + 1.0) / 2.0 * (HI[i] - LO[i])
            })
            .collect()
    }

    /// The solved region, which is where `repair` lives.
    async fn region() -> anyhow::Result<FeasibleRegion> {
        ConstraintSolver::new()
            .with_seed(0x50_50_1E_5E_ED)
            .solve(&system()?)
            .await
            .context("the spring should be satisfiable")
    }

    /// The point Artemis's wrapper received from `repair` (its own output,
    /// passed back in) was a fixed point with both binding residuals at
    /// `-1e-15`, and it did not survive the frame round trip. With a clearance
    /// it is not a fixed point any more: it comes back stepped inside, and
    /// the round trip leaves it feasible.
    #[pollster::test]
    async fn the_spring_vertex_is_landed_one_rounding_error_inside() -> anyhow::Result<()> {
        let region = region().await?;
        let landed = [
            0.052_986_411_203_565_92_f64,
            0.388_738_764_466_29,
            9.632_040_910_614,
        ];

        let back = region.repair(&landed, CLEARANCE).context("repairable")?;

        let g = residuals(&back)?;
        assert!(
            g.iter().all(|v| *v <= 0.0),
            "repair's own output is not feasible: {g:?}"
        );
        let g_rt = residuals(&round_trip(&back))?;
        assert!(
            g_rt.iter().all(|v| *v <= 0.0),
            "repair's output is feasible by {:.1e} but infeasible after a one-ulp frame round \
             trip ({:.1e})",
            worst(&g),
            worst(&g_rt)
        );
        Ok(())
    }

    /// The general statement: from a point just outside, `repair` lands
    /// somewhere that any one-ulp move of any coordinate leaves feasible. It
    /// passed before the clearance existed, one step away from the vertex; it
    /// is here so a fix for the vertex does not lose it.
    #[pollster::test]
    async fn a_repaired_point_should_survive_a_one_ulp_perturbation() -> anyhow::Result<()> {
        let region = region().await?;
        // Just outside the vertex: a thicker wire violates the deflection
        // constraint (the first, which grows with `d^4` in the denominator)
        // while the others stay satisfied.
        let outside = [0.0529 * 1.02, 0.3887, 9.632];
        let g0 = residuals(&outside)?;
        assert!(
            g0.iter().any(|v| *v > 0.0),
            "the starting point should be infeasible: {g0:?}"
        );

        let landed = region.repair(&outside, CLEARANCE).context("repairable")?;

        let g = residuals(&landed)?;
        assert!(g.iter().all(|v| *v <= 0.0));
        for i in 0..3 {
            for up in [false, true] {
                let mut p = landed.clone();
                p[i] = if up { p[i].next_up() } else { p[i].next_down() };
                let nudged = worst(&residuals(&p)?);
                assert!(
                    nudged <= 0.0,
                    "one ulp {} on {} makes the landed point infeasible by {nudged:.1e}; repair's \
                     margin is {:.1e}",
                    if up { "up" } else { "down" },
                    NAMES[i],
                    worst(&g)
                );
            }
        }
        Ok(())
    }
}

/// Artemis, 2026-09-11, `e06-stepped-beam-20`. The stepped cantilever
/// (Vanderplaats 1984; OASIS `Samples/PowerShell/Stepped Beam 100` for the
/// constants and box), `n` segments: `b1..bn in [1, 5]`, `h1..hn in [5, 100]`;
/// per segment a bending-stress limit and the aspect ratio `h_i <= 20 b_i`;
/// one tip-deflection limit coupling every segment. Feasible fraction of the
/// box by Monte Carlo: `5e-4` at `n = 5`, below `5e-6` at `n = 20`. The region
/// is easy to construct a point in — the all-max corner is feasible.
///
/// At `n = 5` the census took 264 ms; at `n = 20` it ran 3.7 CPU-hours
/// without returning. Sampling found the region in under a second either
/// way; what did not return was gap coverage, up to sixteen Z3 queries before
/// `solve` returns, each given the whole solver limit and the n = 20
/// document taking the full 3,000,000 units (115 s) to answer `unknown`.
/// Coverage is now a bisection under a budget of contractions
/// (`DEFAULT_PRUNE_BUDGET`), which on a region in one piece is bought for
/// nothing and bounded; see `cover` in `src/cvg/mod.rs`.
mod census_does_not_return_on_the_20_segment_beam {
    use super::*;
    use crate::common::stepped_beam;

    #[pollster::test]
    async fn the_5_segment_beam_census_is_quick() -> anyhow::Result<()> {
        let mut region = ConstraintSolver::new()
            .with_seed(0)
            .solve(&stepped_beam(5)?)
            .await?;
        assert_eq!(region.take(256).ncols(), 256);
        Ok(())
    }

    /// At the default solver limit — the one Artemis ran. No wall clock of
    /// its own: nextest's slow-timeout is the bound that turns a census that
    /// does not return into a failure with a name.
    ///
    /// The regression is pinned at `n = 10`, where it reproduced — the census
    /// did not return within that bound before the coverage budget existed,
    /// and returns in about 35 s with it — rather than at Artemis's `n = 20`,
    /// where the honest answer is slow rather than absent: the one gap query
    /// the budget affords spends its 3,000,000 units in about 110 s and the
    /// walker's 256 points take another 30, for 137 s measured on 2026-09-12.
    /// A test that needs its own timeout is a test at the wrong size.
    #[pollster::test]
    async fn the_10_segment_beam_census_returns() -> anyhow::Result<()> {
        let mut region = ConstraintSolver::new()
            .with_seed(0)
            .solve(&stepped_beam(10)?)
            .await?;
        assert_eq!(region.take(256).ncols(), 256);
        Ok(())
    }

    /// The ladder without the solver: the probe, the local solve, the walker.
    /// Beyond ten segments the probe lands nothing and the seed is the local
    /// solve's; with the solver configured the opening would then spend one
    /// budgeted gap query on the beam's document — minutes at a hundred
    /// segments — looking for components a local seed cannot vouch against.
    /// What that coverage should be after a local seed is the next question
    /// (`docs/todo.md`); these tests pin what the seed itself costs.
    fn without_the_solver() -> ConstraintSolver {
        ConstraintSolver::new().with_seed(0).with_strategies(vec![
            Strategy::BruteSquad,
            Strategy::LocalSolve,
            Strategy::HitAndRun,
        ])
    }

    /// Artemis's target scale, 2026-09-13: the census opens — a feasible point
    /// is in hand — in about 150 ms at 200 variables, by a local solve whose
    /// first feasible point is its 24th evaluation. The points that follow are
    /// the walker's, and its burn-in at 200 dimensions on this system is about
    /// 95 s before the first one, so this test asks for the opening alone.
    #[pollster::test]
    async fn the_100_segment_beam_opens_by_local_solve() -> anyhow::Result<()> {
        let verdict = without_the_solver().solve(&stepped_beam(100)?).await;
        assert!(
            verdict.is_ok(),
            "the beam is not empty (b = 5, h = 100 is feasible), yet: {verdict:?}"
        );
        Ok(())
    }

    /// Past where the probe reaches, the whole census: seed by local solve,
    /// then 256 walked points — about 35 s at thirty segments.
    #[pollster::test]
    async fn the_30_segment_beam_census_returns_from_a_local_seed() -> anyhow::Result<()> {
        let mut region = without_the_solver().solve(&stepped_beam(30)?).await?;
        assert_eq!(region.take(256).ncols(), 256);
        Ok(())
    }
}

/// Artemis 0.13.2, 2026-09-15, against `587199c`: `c06-rosenbrock-50-slab`
/// came out worse at every concurrency level and the ball cases did not
/// move. The report is a table of where `repair` landed against the
/// closed-form Euclidean-nearest feasible point: the shell exact, the far
/// disc exact, the disc's near misses 1.03–1.54× farther, and the slab
/// `√2×` farther every time — `x1` moved the whole gap where `x1` and `x2`
/// should each have moved half.
///
/// The clamp is an axis projection — the L1-nearest point, which moves one
/// coordinate wherever one can reach — and `repair` used to return the
/// moment it landed, so the Euclidean projection never ran on exactly the
/// cases an optimizer produces: a step a hair over a curved wall, a step
/// off a slab. The contract is Euclidean now, in box-normalised
/// coordinates, and the projection runs from the clamp's landing on every
/// repair that is not separable. The three tables are the acceptance, each
/// against its closed form.
mod repair_lands_axis_aligned_not_nearest {
    use super::*;
    use anyhow::Context;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};
    use sojourn::FeasibleRegion;

    const CLEARANCE: f64 = 1e-12;
    /// The projection converges to `FINAL_RADIUS` in the cube and lands
    /// `LEAST_MARGIN` inside; against a proposal a hundredth of a unit off a
    /// wall both are visible, and a tenth of a percent is generous to them
    /// and nowhere near the ratios reported.
    const ALLOWANCE: f64 = 1e-3;

    fn l2(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    fn system(n: usize, lo: f64, hi: f64, sources: &[String]) -> anyhow::Result<ConstraintSystem> {
        let variables = (1..=n)
            .map(|i| InputVariable::new(format!("x{i}"), lo, hi))
            .collect();
        Ok(ConstraintSystem::new(variables, sources.iter().cloned())?)
    }

    async fn region(system: &ConstraintSystem) -> anyhow::Result<FeasibleRegion> {
        ConstraintSolver::new()
            .with_seed(7)
            .solve(system)
            .await
            .context("the fixture should be satisfiable")
    }

    /// Every landing against its closed form: feasible with the clearance,
    /// and no farther from the proposal than the nearest feasible point,
    /// within the allowance. Reported together, as the table it is.
    fn complaints(
        system: &ConstraintSystem,
        proposals: &[Vec<f64>],
        landings: &[Result<Vec<f64>, sojourn::RepairError>],
        nearest: impl Fn(&[f64]) -> Vec<f64>,
    ) -> Vec<String> {
        let mut complaints = Vec::new();
        for (k, (proposal, landing)) in proposals.iter().zip(landings).enumerate() {
            let closed = nearest(proposal);
            assert!(
                system.is_feasible(&closed, 0.0),
                "the closed-form nearest point is not feasible: {closed:?}"
            );
            let reach = l2(proposal, &closed);
            match landing {
                Ok(landed) => {
                    let ratio = l2(proposal, landed) / reach;
                    if !system.is_feasible(landed, CLEARANCE) {
                        complaints.push(format!("#{k}: landed without the clearance"));
                    }
                    if ratio > 1.0 + ALLOWANCE {
                        complaints.push(format!(
                            "#{k}: landed {ratio:.3}x farther than the nearest feasible point"
                        ));
                    }
                }
                Err(error) => complaints.push(format!("#{k}: {error}")),
            }
        }
        complaints
    }

    /// `c05`: `(x1 - 3)^2 + (x2 - 3)^2 < 2.25` over `[-10, 10]^50`. The nearest
    /// point is radial onto the circle of radius 1.5 about `(3, 3)`; the
    /// proposals sit a hundredth, three tenths, three and eight units outside
    /// it at random angles, the other 48 coordinates anywhere.
    #[pollster::test]
    async fn a_ball_near_miss_lands_on_the_radial_projection() -> anyhow::Result<()> {
        let system = system(
            50,
            -10.0,
            10.0,
            &["(x1 - 3)^2 + (x2 - 3)^2 < 2.25".to_owned()],
        )?;
        let region = region(&system).await?;
        let mut rng = SmallRng::seed_from_u64(1);
        let mut proposals = Vec::new();
        for scale in [0.01, 0.3, 3.0, 8.0] {
            for _ in 0..2 {
                let mut p: Vec<f64> = (0..50).map(|_| rng.random_range(-10.0..10.0)).collect();
                let angle: f64 = rng.random_range(0.0..std::f64::consts::TAU);
                p[0] = (3.0 + (1.5 + scale) * angle.cos()).clamp(-10.0, 10.0);
                p[1] = (3.0 + (1.5 + scale) * angle.sin()).clamp(-10.0, 10.0);
                proposals.push(p);
            }
        }

        let landings: Vec<_> = proposals
            .iter()
            .map(|p| region.repair(p, CLEARANCE))
            .collect();

        let complaints = complaints(&system, &proposals, &landings, |p| {
            let (dx, dy) = (p[0] - 3.0, p[1] - 3.0);
            let scale = (1.5 - 1e-9) / dx.hypot(dy);
            let mut q = p.to_vec();
            q[0] = 3.0 + dx * scale;
            q[1] = 3.0 + dy * scale;
            q
        });
        assert!(complaints.is_empty(), "{}", complaints.join("\n"));
        Ok(())
    }

    /// `c06`: `x1 == x2 + 1 +/- 0.01` over `[-10, 10]^50`. The nearest point
    /// shifts `x1` and `x2` symmetrically onto the nearer face; the clamp
    /// alone moved `x1` by the whole gap, `√2×` farther, on every row.
    #[pollster::test]
    async fn a_slab_landing_moves_both_coordinates() -> anyhow::Result<()> {
        let system = system(50, -10.0, 10.0, &["x1 == x2 + 1 +/- 0.01".to_owned()])?;
        let region = region(&system).await?;
        let mut rng = SmallRng::seed_from_u64(2);
        let mut proposals = Vec::new();
        for gap in [0.02, 0.1, 1.0, 5.0, -0.02, -1.0, -5.0] {
            let mut p: Vec<f64> = (0..50).map(|_| rng.random_range(-9.0..9.0)).collect();
            p[0] = p[1] + 1.0 + gap;
            if p[0].abs() > 10.0 {
                p[1] = 0.0;
                p[0] = 1.0 + gap;
            }
            proposals.push(p);
        }

        let landings: Vec<_> = proposals
            .iter()
            .map(|p| region.repair(p, CLEARANCE))
            .collect();

        let complaints = complaints(&system, &proposals, &landings, |p| {
            let gap = p[0] - p[1] - 1.0;
            let excess = gap.abs() - (0.01 - 1e-9);
            let mut q = p.to_vec();
            if excess > 0.0 {
                let shift = excess / 2.0 * gap.signum();
                q[0] -= shift;
                q[1] += shift;
            }
            q
        });
        assert!(complaints.is_empty(), "{}", complaints.join("\n"));
        Ok(())
    }

    /// `c12`: `sum x_i^2 == 1 +/- 1e-4` over `[0, 1]^20`. The nearest point
    /// is the radial scaling onto the nearer face of the shell. Passed on
    /// `587199c` already — clamping cannot land on a twenty-variable
    /// equality, so the projection always ran — and is here so the contract
    /// is pinned on a shape where the clamp never had a say.
    #[pollster::test]
    async fn a_sphere_shell_landing_is_the_radial_projection() -> anyhow::Result<()> {
        let system = system(
            20,
            0.0,
            1.0,
            &["sum(1, 20, i -> (var[i])^2) == 1 +/- 0.0001".to_owned()],
        )?;
        let region = region(&system).await?;
        let mut rng = SmallRng::seed_from_u64(3);
        let mut proposals: Vec<Vec<f64>> = (0..5)
            .map(|_| (0..20).map(|_| rng.random_range(0.0..1.0)).collect())
            .collect();
        proposals.push(vec![1.0; 20]);
        proposals.push(vec![0.5; 20]);
        proposals.push((0..20).map(|i| 0.05 + 0.04 * i as f64).collect());

        let landings: Vec<_> = proposals
            .iter()
            .map(|p| region.repair(p, CLEARANCE))
            .collect();

        let complaints = complaints(&system, &proposals, &landings, |p| {
            let r2: f64 = p.iter().map(|x| x * x).sum();
            let target = if r2 > 1.0 {
                (1.0_f64 + 1e-4 - 1e-10).sqrt()
            } else {
                (1.0_f64 - 1e-4 + 1e-10).sqrt()
            };
            let scale = target / r2.sqrt();
            p.iter().map(|x| x * scale).collect()
        });
        assert!(complaints.is_empty(), "{}", complaints.join("\n"));
        Ok(())
    }
}

/// Artemis 0.13.2, 2026-09-15, `c10-keane-bump-50` and `c11-keane-bump-100`
/// against `587199c`: `repair` answered `Stranded` on Keane's bump —
/// `0.75 - prod x_i < 0`, `sum x_i - 7.5 n < 0` over `[0, 10]^n` — for
/// proposals with several coordinates driven to the lower wall plus the
/// clearance, where the product is `1e-59` against the `0.75` needed and a
/// feasible point is a few units away (every coordinate below 1 lifted to 1).
/// Both runs aborted on it; the census had no trouble at all.
///
/// A constraint that flat is invisible to every local method: every slice is
/// empty, and COBYLA's linear model of a function flat to fifty digits moves
/// nothing. What was needed is a feasible point to walk *in* from, which the
/// chord design had from its anchors and the projection did not. It has one
/// now without anchors: a local solve from the box centre under a fixed seed
/// — a function of the system — a chord bisected from it toward the
/// proposal, and the projection from where the chord lands, where the
/// constraint is well-scaled again. The proposal is Artemis's, verbatim;
/// the hundred-variable one repairs the same way and is not here because
/// the projection from a point that flat spends its whole budget before the
/// reference is tried — 4.7 s in release, minutes unoptimised.
mod repair_strands_on_a_product_constraint {
    use super::*;
    use anyhow::Context;

    const CLEARANCE: f64 = 1e-12;

    const POINT_50: [f64; 50] = [
        0.08928918738349267,
        0.8128352441524971,
        9.423980250530082,
        8.201514214434269e-7,
        0.074313852199003,
        9.787239247939075,
        1.4211922856333103e-6,
        0.1897163472654313,
        9.84180144604797,
        9.365967211219166,
        0.050729876983561795,
        9.964434055086267,
        9.99999999999,
        9.998420069515479,
        0.6367844193595209,
        1.000177718424311e-11,
        0.2817004031539234,
        0.08928918738349179,
        0.04463556293675364,
        9.888842446816353,
        9.755546509597457,
        9.993325488268553,
        0.04070536976125716,
        9.982578066498466,
        6.753114992849506,
        9.916261792957172,
        9.917970006505211,
        9.84180144604797,
        9.965996110813085,
        0.04224343643094741,
        0.08928918738349179,
        0.0892891873834909,
        9.995418246406189,
        1.000177718424311e-11,
        0.45955550570587533,
        0.07810019738970642,
        3.6437449990600612e-6,
        9.84180144604797,
        0.09112711432073706,
        9.992216303186801,
        9.5554706945567,
        1.000177718424311e-11,
        0.021898415600998256,
        0.047301883444006876,
        10.0,
        1.000177718424311e-11,
        0.0575386197917398,
        9.980275812968333,
        9.994656793123873,
        0.21906571038074585,
    ];

    fn keane(n: usize) -> anyhow::Result<ConstraintSystem> {
        let variables = (1..=n)
            .map(|i| InputVariable::new(format!("x{i}"), 0.0, 10.0))
            .collect();
        Ok(ConstraintSystem::new(
            variables,
            [
                format!("0.75 - prod(1, {n}, i -> var[i]) < 0"),
                format!("sum(1, {n}, i -> var[i]) - {} < 0", 7.5 * n as f64),
            ],
        )?)
    }

    fn l2(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    /// A feasible point within a few units of the proposal: every coordinate
    /// below 1 lifted to 1, and the ones on the upper wall pulled in so the
    /// clearance has room there too.
    fn lifted(point: &[f64]) -> Vec<f64> {
        point.iter().map(|&x| x.clamp(1.0, 10.0 - 1e-9)).collect()
    }

    async fn is_repaired(n: usize, point: &[f64]) -> anyhow::Result<()> {
        let system = keane(n)?;
        let region = ConstraintSolver::new()
            .with_seed(0x50_50_1E_5E_ED)
            .solve(&system)
            .await
            .context("Keane's region is nearly the whole box")?;
        let lifted = lifted(point);
        assert!(
            system.is_feasible(&lifted, CLEARANCE),
            "the lifted point should be plainly feasible"
        );

        let repaired = region.repair(point, CLEARANCE).with_context(|| {
            format!(
                "stranded at n = {n} with a feasible point {:.2} away",
                l2(point, &lifted)
            )
        })?;

        assert!(
            system.is_feasible(&repaired, CLEARANCE),
            "landed without the clearance: {repaired:?}"
        );
        assert!(
            l2(point, &repaired) <= l2(point, &lifted) + 1e-9,
            "landed farther than the lifted point ({} > {})",
            l2(point, &repaired),
            l2(point, &lifted)
        );
        Ok(())
    }

    #[pollster::test]
    async fn the_50_variable_proposal_is_repaired() -> anyhow::Result<()> {
        is_repaired(50, &POINT_50).await
    }
}
