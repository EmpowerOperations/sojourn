//! The projection by Newton on the KKT system, over the active set.
//!
//! # The problem, and why it is this easy
//!
//! The nearest feasible point to `p` is `min ½‖u − p‖²` subject to
//! `gᵢ(u) ≤ 0`. At the answer `u*` the KKT conditions hold: `u* − p =
//! −Σ λᵢ ∇gᵢ(u*)` with `λᵢ ≥ 0` for the constraints *active* there and zero
//! for the rest. That is the college question — the shortest distance from a
//! point to a surface is along the surface's normal — in as many dimensions
//! as there are variables. The objective's Hessian is the identity, so the
//! Newton step for the active set `A` is the projection of `p` onto the
//! constraints *linearised* at the current iterate:
//!
//! ```text
//! (J Jᵀ) λ = J (p − uₖ) + g(uₖ)        J = ∇g_A(uₖ), one row per active constraint
//! uₖ₊₁ = p − Jᵀ λ
//! ```
//!
//! — a `|A| × |A|` solve, and `|A|` is a few. On a linear constraint it is
//! exact in one step (the foot of the perpendicular, closed-form); on a
//! curved one it converges linearly, the curvature being the one thing it
//! does not model — see [`NEWTON_ITERATIONS`]. COBYLA, which this replaces
//! where it applies, estimates the same gradients by finite differences
//! dressed as linear models and pays `O(d²m)` per iteration for the
//! privilege.
//!
//! # The active set is a fixed point
//!
//! Which constraints are active at the answer is not known in advance. The
//! guess is the ones violated where the point stands; the solve nominates a
//! point; every constraint is judged there; a newly violated one joins the
//! set and one whose multiplier came out negative — a constraint the
//! projection is pulling *away* from, which at the answer cannot be active —
//! leaves it; and the solve runs again from where it got to. That converges
//! when the nominated point violates nothing off the set and every multiplier
//! is non-negative: the KKT conditions, verified rather than assumed. Bounds
//! are constraints like any other, linear ones with gradient `±eⱼ`, and take
//! part in the set only when violated. So the constraints a gradient is
//! computed for are the few that bite, which is what keeps this cheap on a
//! system of two hundred.
//!
//! # Fallible by design
//!
//! Every way this can fail answers `None` and the caller tries something
//! that needs no gradient: a constraint in the active set whose tape has no
//! derivative ([`ConstraintSystem::gradient`] is `None`), a gradient that is
//! not finite at the iterate, a singular or non-finite solve (dependent
//! constraints), a Newton iteration that has not converged in its budget, an
//! active set that has not settled in its. All counts; nothing waits.
//!
//! The landing is *on* the boundary of the active constraints, which by the
//! strict oracle is a hair outside; stepping it inside by the clearance is
//! `repair`'s business, along the direction `p → landing`, which is exactly
//! `−Σ λᵢ ∇gᵢ`, the KKT normal.

use faer::linalg::solvers::Solve;
use faer::{Col, Mat};

use super::incidence::ConstraintId;
use crate::{ConstraintSystem, Point};

/// Newton iterations one active set gets to converge.
///
/// A linear constraint converges in one. A curved one converges linearly,
/// because the step models the objective's Hessian — the identity — and not
/// the constraint's: the missing term is `Σ λᵢ ∇²gᵢ`, and its size is the
/// contraction factor. On the sine band of the ball oracle
/// (`tests/cvg_repair.rs`) that factor is a half, and a step of `1e-13` is
/// forty-odd halvings away; on a boundary that bends more sharply across
/// the distance it is nearer one and the count grows, and past one the
/// iteration diverges, which the cap catches. A hundred and twenty-eight is
/// a few hundred gradient evaluations at worst — each one tape run — and a
/// set that has not converged by then is declined, not pushed.
const NEWTON_ITERATIONS: usize = 128;

/// The step below which an iteration has converged, in the unit cube.
///
/// A thousandth of the clearance the doc on `repair` recommends, so the
/// landing is exact at every scale the caller can see.
const NEWTON_STEP: f64 = 1e-13;

/// How many times the active set may change before the search is declined.
///
/// Each round either adds a violated row or drops one with a negative
/// multiplier; a set that cycles is one this method cannot settle, and the
/// cap is the number of rows plus a margin for the drops.
fn active_set_rounds(rows: usize) -> usize {
    rows + 8
}

/// A converged projection: the point, on the boundary of the constraints
/// active there, and the direction *into* the region from it.
pub(crate) struct Landing {
    pub(crate) point: Point,
    /// The negated sum of the unit normals of every constraint the landing
    /// stands on — the active set, and any row that is *tight* there without
    /// having been violated, a box wall the proposal already sat on — in the
    /// caller's coordinates: the bisector of the wedge they make, which
    /// enters every one of them. The KKT direction `p → x*` need not, on a
    /// sharp wedge like the spring's vertex, where one wall's normal is far
    /// steeper than the other's and the weighted sum leans out of it; and
    /// the active set alone need not either, since a wall the proposal never
    /// left is not in it, and a step that ignores it leaves the landing on
    /// that wall with no clearance — measured by Artemis on a slab at ten
    /// variables, where the box then did the stepping and moved a coordinate
    /// no constraint names by a tenth of a unit.
    pub(crate) inward: Vec<f64>,
}

/// How far below zero a residual may be at the landing for its row to count
/// as a wall the landing stands on. A bound the proposal sat on is exactly
/// zero; a constraint Newton landed on is within its own rounding. Generous,
/// because a row wrongly counted only adds an inward component, and the
/// ladder that follows judges every rung by the oracle.
const TIGHT: f64 = 1e-9;

/// The nearest point of `system`'s feasible set to `target`, sought from
/// `from`, on the boundary of the constraints active there; `None` where
/// the method does not apply or does not converge.
pub(crate) fn nearest(system: &ConstraintSystem, from: &[f64], target: &[f64]) -> Option<Landing> {
    let _span = tracing::debug_span!("newton").entered();
    let dimensions = system.variables.len();
    let constraints = system.constraints.len();
    let rows = constraints + 2 * dimensions;

    let lower: Vec<f64> = system
        .variables
        .iter()
        .map(|variable| variable.lower_bound)
        .collect();
    let width: Vec<f64> = system
        .variables
        .iter()
        .map(|variable| variable.upper_bound - variable.lower_bound)
        .collect();
    let normalised = |point: &[f64]| -> Vec<f64> {
        point
            .iter()
            .zip(&lower)
            .zip(&width)
            .map(|((x, lo), w)| if *w > 0.0 { (x - lo) / w } else { 0.0 })
            .collect()
    };
    let denormalised = |unit: &[f64]| -> Point {
        unit.iter()
            .zip(&lower)
            .zip(&width)
            .map(|((u, lo), w)| lo + u * w)
            .collect()
    };

    // A row's residual at `x`, and its gradient with respect to the unit
    // coordinates: a constraint's partials scaled by the widths of the
    // coordinates it names, a bound's `∓1` on its own coordinate.
    let residual = |row: usize, x: &[f64], unit: &[f64]| -> Option<f64> {
        if row < constraints {
            let value = system.constraints[row].compiled.eval_row(x).ok()?;
            value.is_finite().then_some(value)
        } else if row < constraints + dimensions {
            Some(-unit[row - constraints])
        } else {
            Some(unit[row - constraints - dimensions] - 1.0)
        }
    };
    let gradient = |row: usize, x: &[f64]| -> Option<Vec<(usize, f64)>> {
        if row < constraints {
            let partials = system.gradient(row, x)?;
            let named = system.incidence.rows_of(ConstraintId(row));
            let scaled: Vec<(usize, f64)> = named
                .iter()
                .zip(partials)
                .map(|(coordinate, partial)| {
                    (coordinate.index(), partial * width[coordinate.index()])
                })
                .collect();
            scaled
                .iter()
                .all(|(_, partial)| partial.is_finite())
                .then_some(scaled)
        } else if row < constraints + dimensions {
            Some(vec![(row - constraints, -1.0)])
        } else {
            Some(vec![(row - constraints - dimensions, 1.0)])
        }
    };

    let p = normalised(target);
    let mut u = normalised(from);
    let mut active: Vec<usize> = (0..rows)
        .filter(|row| residual(*row, from, &u).is_some_and(|value| value > 0.0))
        .collect();
    // A row the residual could not be evaluated on is one the method cannot
    // reason about.
    if (0..rows).any(|row| residual(row, from, &u).is_none()) {
        tracing::debug!(stage = "newton", declined = "a residual faults");
        return None;
    }

    let mut iterations = 0;
    let mut round = 0;
    let declined = |why: &'static str, iterations: usize, round: usize, active: usize| {
        tracing::debug!(
            stage = "newton",
            declined = why,
            iterations,
            rounds = round,
            active
        );
    };
    for _ in 0..active_set_rounds(rows) {
        round += 1;
        // Newton on the active set, from where the point stands.
        let mut multipliers: Vec<f64> = vec![0.0; active.len()];
        let mut converged = false;
        for _ in 0..NEWTON_ITERATIONS {
            iterations += 1;
            let x = denormalised(&u);
            let k = active.len();
            let mut jacobian = Mat::<f64>::zeros(k, dimensions);
            let mut rhs = Col::<f64>::zeros(k);
            for (i, row) in active.iter().enumerate() {
                let Some(g) = residual(*row, &x, &u) else {
                    declined("a residual faults", iterations, round, active.len());
                    return None;
                };
                let Some(partials) = gradient(*row, &x) else {
                    declined("no gradient", iterations, round, active.len());
                    return None;
                };
                let mut j_dot_step = 0.0;
                for (coordinate, partial) in partials {
                    jacobian[(i, coordinate)] += partial;
                }
                for coordinate in 0..dimensions {
                    j_dot_step += jacobian[(i, coordinate)] * (p[coordinate] - u[coordinate]);
                }
                rhs[i] = j_dot_step + g;
            }

            let next: Vec<f64> = if k == 0 {
                p.clone()
            } else {
                let jjt = &jacobian * jacobian.transpose();
                let lambda = jjt.partial_piv_lu().solve(&rhs);
                if lambda.iter().any(|value| !value.is_finite()) {
                    declined("a singular solve", iterations, round, active.len());
                    return None;
                }
                let pull = jacobian.transpose() * &lambda;
                multipliers = lambda.iter().copied().collect();
                (0..dimensions).map(|c| p[c] - pull[c]).collect()
            };

            let step = next
                .iter()
                .zip(&u)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            u = next;
            if step < NEWTON_STEP {
                converged = true;
                break;
            }
        }
        if !converged {
            declined("no convergence", iterations, round, active.len());
            return None;
        }

        // The fixed point: drop what pulls the wrong way, add what is
        // violated, and stop when neither happens.
        let x = denormalised(&u);
        if let Some(position) = multipliers.iter().position(|lambda| *lambda < 0.0) {
            active.remove(position);
            continue;
        }
        let joining: Vec<usize> = (0..rows)
            .filter(|row| !active.contains(row))
            .filter(|row| residual(*row, &x, &u).is_some_and(|value| value > 0.0))
            .collect();
        if (0..rows)
            .filter(|row| !active.contains(row))
            .any(|row| residual(row, &x, &u).is_none())
        {
            declined("a residual faults", iterations, round, active.len());
            return None;
        }
        if joining.is_empty() {
            tracing::debug!(
                stage = "newton",
                iterations,
                rounds = round,
                active = active.len(),
                "converged"
            );
            // The bisector: each row the landing stands on — active, or
            // tight without having been violated — by its unit normal in the
            // cube, summed and negated, then scaled back to the caller's
            // coordinates.
            let standing: Vec<usize> = (0..rows)
                .filter(|row| {
                    active.contains(row)
                        || residual(*row, &x, &u).is_some_and(|value| value >= -TIGHT)
                })
                .collect();
            let mut inward = vec![0.0; dimensions];
            for row in &standing {
                let partials = gradient(*row, &x)?;
                let norm = partials
                    .iter()
                    .map(|(_, partial)| partial * partial)
                    .sum::<f64>()
                    .sqrt();
                if norm > 0.0 {
                    for (coordinate, partial) in partials {
                        inward[coordinate] -= partial / norm;
                    }
                }
            }
            for (direction, w) in inward.iter_mut().zip(&width) {
                *direction *= w;
            }
            return Some(Landing { point: x, inward });
        }
        active.extend(joining);
    }
    declined(
        "the active set did not settle",
        iterations,
        round,
        active.len(),
    );
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InputVariable;
    use crate::system::tests::system;

    fn close(a: &[f64], b: &[f64], within: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= within)
    }

    /// A linear constraint is exact in one round: the foot of the
    /// perpendicular from `(1, 1)` on `2 x1 + x2 = 1` is `(0.2, 0.6)`.
    #[test]
    fn a_half_space_lands_on_the_foot_of_the_normal() {
        let system = system(
            vec![
                InputVariable::new("x1", -2.0, 2.0),
                InputVariable::new("x2", -2.0, 2.0),
            ],
            &["2*x1 + x2 < 1"],
        );
        let landed = nearest(&system, &[1.0, 1.0], &[1.0, 1.0])
            .expect("applies")
            .point;
        assert!(close(&landed, &[0.2, 0.6], 1e-12), "{landed:?}");
    }

    /// A curved constraint converges to the radial point.
    #[test]
    fn a_disc_lands_radially() {
        let system = system(
            vec![
                InputVariable::new("x", -2.0, 2.0),
                InputVariable::new("y", -2.0, 2.0),
            ],
            &["x^2 + y^2 < 1"],
        );
        let landed = nearest(&system, &[2.0, 0.5], &[2.0, 0.5])
            .expect("applies")
            .point;
        let scale = 4.25_f64.sqrt();
        assert!(
            close(&landed, &[2.0 / scale, 0.5 / scale], 1e-12),
            "{landed:?}"
        );
    }

    /// Two walls meeting at a vertex: both active, the landing is the vertex.
    #[test]
    fn a_vertex_takes_both_constraints() {
        let system = system(
            vec![
                InputVariable::new("x1", 0.0, 1.0),
                InputVariable::new("x2", 0.0, 1.0),
            ],
            &["x1 < 0.5", "x2 < 0.5"],
        );
        let landed = nearest(&system, &[1.0, 1.0], &[1.0, 1.0])
            .expect("applies")
            .point;
        assert!(close(&landed, &[0.5, 0.5], 1e-12), "{landed:?}");
    }

    /// The foot of the normal lies outside the box, so a bound joins the
    /// active set on the second round and the landing is on the bound.
    #[test]
    fn a_bound_joins_the_active_set() {
        let system = system(
            vec![
                InputVariable::new("x1", 0.0, 1.0),
                InputVariable::new("x2", 0.0, 1.0),
            ],
            &["x1 + x2 < 0.2"],
        );
        // From (1, 0): the foot on `x1 + x2 = 0.2` is (0.6, -0.4), below the
        // box; with `x2 >= 0` active the answer is (0.2, 0).
        let landed = nearest(&system, &[1.0, 0.0], &[1.0, 0.0])
            .expect("applies")
            .point;
        assert!(close(&landed, &[0.2, 0.0], 1e-12), "{landed:?}");
    }

    /// A constraint violated at the start but inactive at the projection is
    /// dropped by its negative multiplier.
    #[test]
    fn a_negative_multiplier_drops_a_row() {
        let system = system(
            vec![
                InputVariable::new("x1", -2.0, 2.0),
                InputVariable::new("x2", -2.0, 2.0),
            ],
            // From (1.5, 1.5) both are violated; the projection onto the
            // first alone, (0.5, 0.5), satisfies the second (0.5 + 1 = 1.5 < 2)
            // — the second's multiplier comes out negative and it leaves.
            &["x1 + x2 < 1", "x1 + 2*x2 < 2"],
        );
        let landed = nearest(&system, &[1.5, 1.5], &[1.5, 1.5])
            .expect("applies")
            .point;
        assert!(close(&landed, &[0.5, 0.5], 1e-12), "{landed:?}");
    }

    /// A constraint with no derivative in the active set declines the whole
    /// projection; the same constraint off the set is no obstacle.
    #[test]
    fn a_jump_in_the_active_set_declines() {
        let system = system(
            vec![
                InputVariable::new("x1", 0.0, 10.0),
                InputVariable::new("x2", 0.0, 10.0),
            ],
            &["floor(x1) < 3", "x2 < 1"],
        );
        assert!(nearest(&system, &[5.0, 5.0], &[5.0, 5.0]).is_none());
        let landed = nearest(&system, &[1.0, 5.0], &[1.0, 5.0])
            .expect("the floor is not active")
            .point;
        assert!(close(&landed, &[1.0, 1.0], 1e-12), "{landed:?}");
    }

    /// The same input gives the same point, bit for bit.
    #[test]
    fn a_projection_is_deterministic() {
        let system = system(
            vec![
                InputVariable::new("x", -2.0, 2.0),
                InputVariable::new("y", -2.0, 2.0),
            ],
            &["x^2 + y^2 < 1", "x + y > 0.5"],
        );
        let once = nearest(&system, &[1.2, 1.2], &[1.2, 1.2]).map(|landing| landing.point);
        let twice = nearest(&system, &[1.2, 1.2], &[1.2, 1.2]).map(|landing| landing.point);
        assert_eq!(once, twice);
        assert!(once.is_some());
    }
}
