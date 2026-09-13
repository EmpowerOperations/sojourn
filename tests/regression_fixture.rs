//! Regressions reported by callers, kept in the form they arrived in.
//!
//! Each case is a bug report: the system, the seed, and what the caller saw.
//! They are pinned to seeds on purpose — a regression that depends on the
//! probe's luck is only reproducible with the luck held still — and they stay
//! here after the fix so the luck can never turn again.

use sojourn::{ConstraintSolver, ConstraintSystem, InputVariable, Satisfiability, Status};

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
    use anyhow::anyhow;

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

    async fn points_from(seed: u64) -> anyhow::Result<(usize, Status)> {
        let verdict = ConstraintSolver::new()
            .with_seed(seed)
            .solve(&system()?)
            .await?;
        let mut region = match verdict {
            Satisfiability::Satisfied { region } => region,
            Satisfiability::Unsatisfiable { because } => {
                return Err(anyhow!("seed {seed}: unsatisfiable: {because}"));
            }
        };
        let got = region.take(WANTED).ncols();
        Ok((got, region.status()))
    }

    #[pollster::test]
    async fn seed_2_streams_the_whole_request() -> anyhow::Result<()> {
        let (got, status) = points_from(2).await?;
        assert_eq!(
            got, WANTED,
            "seed 2 delivered {got} of {WANTED} and ended in {status:?}; a connected slab \
             with points in hand should never exhaust"
        );
        assert_eq!(status, Status::Filling);
        Ok(())
    }

    #[pollster::test]
    async fn every_other_seed_streams_the_same_slab() -> anyhow::Result<()> {
        for seed in (0..10u64).filter(|s| *s != 2) {
            let (got, status) = points_from(seed).await?;
            assert_eq!(got, WANTED, "seed {seed} ended early in {status:?}");
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
    use anyhow::{Context, anyhow};
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

    /// The solved region, with 256 census points taken as anchors.
    async fn region_and_anchors() -> anyhow::Result<(FeasibleRegion, Mat<f64>)> {
        match ConstraintSolver::new()
            .with_seed(0x50_50_1E_5E_ED)
            .solve(&system()?)
            .await?
        {
            Satisfiability::Satisfied { mut region } => {
                let anchors = region.take(256);
                Ok((region, anchors))
            }
            Satisfiability::Unsatisfiable { because } => {
                Err(anyhow!("the spring is unsatisfiable: {because}"))
            }
        }
    }

    /// The point Artemis's wrapper received from `repair` (its own output,
    /// passed back in) was a fixed point with both binding residuals at
    /// `-1e-15`, and it did not survive the frame round trip. With a clearance
    /// it is not a fixed point any more: it comes back stepped inside, and
    /// the round trip leaves it feasible.
    #[pollster::test]
    async fn the_spring_vertex_is_landed_one_rounding_error_inside() -> anyhow::Result<()> {
        let (region, anchors) = region_and_anchors().await?;
        let landed = [
            0.052_986_411_203_565_92_f64,
            0.388_738_764_466_29,
            9.632_040_910_614,
        ];

        let back = region
            .repair(anchors.as_ref(), &landed, CLEARANCE)
            .context("repairable")?;

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
        let (region, anchors) = region_and_anchors().await?;
        // Just outside the vertex: a thicker wire violates the deflection
        // constraint (the first, which grows with `d^4` in the denominator)
        // while the others stay satisfied.
        let outside = [0.0529 * 1.02, 0.3887, 9.632];
        let g0 = residuals(&outside)?;
        assert!(
            g0.iter().any(|v| *v > 0.0),
            "the starting point should be infeasible: {g0:?}"
        );

        let landed = region
            .repair(anchors.as_ref(), &outside, CLEARANCE)
            .context("repairable")?;

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
