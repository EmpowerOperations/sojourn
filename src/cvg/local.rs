//! A local solve for the search's first point.
//!
//! Finding one feasible point of a nonlinear inequality system is an ordinary
//! constrained optimisation, and a local method eats it: Artemis's SLSQP
//! reaches the stepped beam's optimum from a corner in seconds, where the SMT
//! solver spends a million resource units per query and brute force cannot
//! find a region a millionth of its box at two hundred variables. This is
//! COBYLA (Powell 1994), derivative-free, with the constraints handled natively
//! by linear models inside a trust region — `basin`'s implementation, pure Rust.
//! It runs after the probe finds nothing and before any solver is asked, so
//! that a solver is reached only when there is nothing to find, which is when
//! a proof is what is wanted.
//!
//! # The objective is the sum of the residuals, and the solve stops at the
//! first feasible point
//!
//! The constraints are COBYLA's to satisfy; the objective only has to give
//! its linear models a slope toward satisfying them from wherever the start
//! is. The sum `Σ g_i(x)` does, everywhere: every coordinate that any
//! constraint names moves it. The worst residual `max_i g_i(x)` — the obvious
//! choice, since descending it is descending toward feasibility — does not:
//! on two hundred bounds `x_i > 10.5` started at the box centre every residual
//! is a hair positive, moving one coordinate fixes one constraint and the
//! maximum does not move, and COBYLA converges on that plateau at a point the
//! oracle rejects, three hundred iterations later. Under the sum the same
//! start lands inside the initial simplex.
//!
//! The solve is not run to any optimum. Every evaluation is a callback of
//! ours, and the moment one is judged feasible the run is stopped: on the
//! stepped beam the first feasible point is evaluation 24 at 100 segments,
//! inside the initial simplex, where converging from there would cost hours,
//! since COBYLA's model algebra grows with the square of the dimension times
//! the constraint count. A seed on the boundary is a seed; the walker's
//! burn-in is what moves away from walls, far more cheaply than the solver.
//!
//! # In the unit cube
//!
//! COBYLA's trust region has one radius for every coordinate, and a box is
//! `[1, 5]` on one coordinate and `[5, 100]` on the next. The solve runs over
//! `u ∈ [0, 1]^d` with `x = lo + u·width`, so a radius means the same on every
//! coordinate of every problem, and the box itself is `2d` more constraints in
//! the same `<= 0` form.
//!
//! # Judged, never trusted
//!
//! Every point the solver evaluates is judged as it goes — the box, every
//! residual `<= 0`, nothing non-finite — and the answer is the judged point
//! with the lowest worst residual seen, never the solver's reported optimum.
//! The one returned is put to [`ConstraintSystem::is_feasible`] once more on
//! the way out, so what leaves here passes the same oracle as every other seed.
//!
//! # Counts, not clocks
//!
//! The starts are the box centre and then draws from the search's seeded rng;
//! each runs for a number of cost evaluations fixed by the dimension. COBYLA
//! is deterministic from a start, so the same seed finds the same point on any
//! machine. Cancellation is honoured per evaluation, through the same flag
//! that stops a run on landing.

use std::cell::{Cell, RefCell};
use std::convert::Infallible;
use std::rc::Rc;

use basin::core::termination::TerminationReason;
use basin::{Cobyla, CobylaState, CostFunction, Executor, NonlinearInequalityConstraints};
use rand::RngExt;
use rand::rngs::Xoshiro256PlusPlus;

use super::Cancellation;
use crate::{ConstraintSystem, Point};

/// How many starts a local solve gets before conceding the region to the
/// solver: the box centre, then this many minus one draws from the seeded rng.
///
/// Four, because a local method's failure is a bad basin rather than a bad
/// problem, and a fresh start elsewhere is the cheap remedy; more than a few
/// and the solve is doing the sampler's job worse than the sampler.
const STARTS: usize = 4;

/// Cost evaluations a start may spend, per dimension plus one, before it is
/// given up as a bad basin.
///
/// A count, so the same seed finds the same point on every machine, and
/// scaled by the dimension because COBYLA's initial simplex alone is `d + 1`
/// evaluations. Eight of those: on the stepped beam the first feasible point
/// came inside the first simplex at both 20 and 100 segments, so a start that
/// has not landed after eight is not going to. The cap matters only to a start
/// that fails, and a failing start at two hundred variables is minutes of
/// model algebra per simplex-worth — the price of not conceding to the solver
/// early, paid rarely.
const EVALS_PER_DIMENSION: u64 = 8;

/// The initial trust-region radius, in the unit cube: a quarter of the box on
/// every coordinate, a coarse first step that the models then shrink.
const INITIAL_RADIUS: f64 = 0.25;

/// The final radius: a start that has shrunk its models this fine without a
/// feasible evaluation has converged somewhere infeasible and is given up.
const FINAL_RADIUS: f64 = 1e-6;

/// A feasible point of `problem`, found by local solves from a few starts, or
/// `None` when every start ran out of budget without one.
///
/// `rng` draws the starts after the first; `cancel` is checked between starts.
pub(crate) fn seed(
    problem: &ConstraintSystem,
    rng: &mut Xoshiro256PlusPlus,
    cancel: &Cancellation<'_>,
) -> Option<Point> {
    let dimensions = problem.variables.len();
    if dimensions == 0 {
        return None;
    }

    let budget = EVALS_PER_DIMENSION * (dimensions as u64 + 1);
    for start in 0..STARTS {
        if cancel.is_requested() {
            return None;
        }
        let from: Vec<f64> = if start == 0 {
            vec![0.5; dimensions]
        } else {
            (0..dimensions)
                .map(|_| rng.random_range(0.0..1.0))
                .collect()
        };

        let landing = Landing::over(problem, cancel);
        let solver = Cobyla::new()
            .with_initial_radius(INITIAL_RADIUS)
            .with_final_radius(FINAL_RADIUS);
        // The run stops the moment a point is judged feasible, or the search
        // is cancelled: `stop_when` wants a `'static` closure, so the flag it
        // reads is shared with the landing rather than borrowed from it.
        let stop = Rc::clone(&landing.stop);
        // `Infallible` is the error type, so `run` cannot fail; the `Ok` is a
        // type-level formality.
        let Ok(result) = Executor::new(&landing, solver, CobylaState::new(from))
            .max_cost_evals(budget)
            .stop_when(move |_| stop.get().then_some(TerminationReason::UserRequested))
            .run();

        let best = landing.best.into_inner();
        tracing::debug!(
            start,
            evaluations = result.cost_evals(),
            reason = ?result.reason,
            landed = best.is_some(),
            "local solve"
        );
        if let Some((_, point)) = best
            && problem.is_feasible(&point, 0.0)
        {
            tracing::info!(
                start,
                evaluations = result.cost_evals(),
                "a local solve seeded the search"
            );
            return Some(point);
        }
    }
    None
}

/// The problem as COBYLA sees it: over the unit cube, judged as it goes.
struct Landing<'a> {
    problem: &'a ConstraintSystem,
    cancel: &'a Cancellation<'a>,
    lower: Vec<f64>,
    width: Vec<f64>,
    /// The judged point with the lowest worst residual so far, with that
    /// residual — normally the first, since the run stops on it, but the stop
    /// is read between iterations and an iteration may evaluate a few more.
    /// Interior mutability because the solver holds the problem by shared
    /// reference and the cost function is, to it, pure.
    best: RefCell<Option<(f64, Point)>>,
    /// Raised when a point has been judged feasible or the search cancelled;
    /// the executor's stop check reads it between iterations.
    stop: Rc<Cell<bool>>,
}

impl<'a> Landing<'a> {
    fn over(problem: &'a ConstraintSystem, cancel: &'a Cancellation<'a>) -> Self {
        let lower = problem
            .variables
            .iter()
            .map(|variable| variable.lower_bound)
            .collect();
        let width = problem
            .variables
            .iter()
            .map(|variable| variable.upper_bound - variable.lower_bound)
            .collect();
        Self {
            problem,
            cancel,
            lower,
            width,
            best: RefCell::new(None),
            stop: Rc::new(Cell::new(false)),
        }
    }

    /// The point in the declared box that `unit` names.
    fn denormalised(&self, unit: &[f64]) -> Point {
        unit.iter()
            .zip(&self.lower)
            .zip(&self.width)
            .map(|((u, lower), width)| lower + u * width)
            .collect()
    }

    /// Every constraint's residual at `point`, a fault as `INFINITY` — a
    /// point the solver is to step away from, in `basin`'s own convention.
    fn residuals(&self, point: &Point) -> Vec<f64> {
        self.problem
            .constraints
            .iter()
            .map(|constraint| {
                constraint
                    .compiled
                    .eval_row(point)
                    .map_or(f64::INFINITY, |residual| {
                        if residual.is_finite() {
                            residual
                        } else {
                            f64::INFINITY
                        }
                    })
            })
            .collect()
    }
}

impl CostFunction for &Landing<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, unit: &Vec<f64>) -> Result<f64, Infallible> {
        let point = self.denormalised(unit);
        let residuals = self.residuals(&point);
        let worst = residuals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = residuals.iter().sum();

        // Judged here rather than trusted from the solver: inside the box,
        // every residual `<= 0`, nothing non-finite — the oracle's own rule.
        // The first such point ends the run.
        let inside = unit.iter().all(|u| (0.0..=1.0).contains(u));
        if inside && worst <= 0.0 {
            let mut best = self.best.borrow_mut();
            if best.as_ref().is_none_or(|(deepest, _)| worst < *deepest) {
                *best = Some((worst, point));
            }
            self.stop.set(true);
        }
        if self.cancel.is_requested() {
            self.stop.set(true);
        }
        Ok(total)
    }
}

impl NonlinearInequalityConstraints for &Landing<'_> {
    fn constraints(&self, unit: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        let point = self.denormalised(unit);
        let mut rows = self.residuals(&point);
        // The box, in the cube's own coordinates: `-u <= 0` and `u - 1 <= 0`.
        rows.extend(unit.iter().map(|u| -u));
        rows.extend(unit.iter().map(|u| u - 1.0));
        Ok(rows)
    }

    fn num_constraints(&self) -> usize {
        self.problem.constraints.len() + 2 * self.problem.variables.len()
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::InputVariable;
    use crate::system::tests::system;

    fn rng() -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::seed_from_u64(0x0001_0CA1)
    }

    #[test]
    fn a_half_space_is_seeded_from_the_centre() {
        let system = system(
            vec![
                InputVariable::new("x1", -2.0, 2.0),
                InputVariable::new("x2", -2.0, 2.0),
            ],
            &["2*x1 + x2 < -3"],
        );
        let point = seed(&system, &mut rng(), &Cancellation::never()).expect("a half-space seeds");
        assert!(system.is_feasible(&point, 0.0), "{point:?}");
    }

    #[test]
    fn a_disc_off_centre_is_seeded() {
        let system = system(
            vec![
                InputVariable::new("x", -3.0, 3.0),
                InputVariable::new("y", -3.0, 3.0),
            ],
            &["sqr(x - 2) + sqr(y - 2) < 0.25"],
        );
        let point = seed(&system, &mut rng(), &Cancellation::never()).expect("a disc seeds");
        assert!(system.is_feasible(&point, 0.0), "{point:?}");
    }

    /// The plateau case: every constraint a hair violated at the centre, so a
    /// worst-residual objective would be flat across the initial simplex. The
    /// sum is not, and the centre start lands inside that simplex.
    #[test]
    fn a_corner_of_many_strict_bounds_is_seeded_from_the_centre() {
        let names: Vec<String> = (1..=40).map(|i| format!("x{i}")).collect();
        let inputs = names
            .iter()
            .map(|name| InputVariable::new(name.clone(), 10.0, 11.0))
            .collect();
        let sources: Vec<String> = names.iter().map(|name| format!("{name} > 10.5")).collect();
        let sources: Vec<&str> = sources.iter().map(String::as_str).collect();
        let system = system(inputs, &sources);
        assert!(
            !system.is_feasible(&vec![10.5; 40], 0.0),
            "the centre is on every wall"
        );

        let point = seed(&system, &mut rng(), &Cancellation::never()).expect("the corner seeds");
        assert!(system.is_feasible(&point, 0.0), "{point:?}");
    }

    #[test]
    fn an_empty_region_is_not_seeded() {
        let system = system(
            vec![InputVariable::new("x", 0.0, 10.0)],
            &["x > 8", "x < 2"],
        );
        assert!(seed(&system, &mut rng(), &Cancellation::never()).is_none());
    }

    #[test]
    fn a_faulting_constraint_still_seeds() {
        // `ln` of a negative faults; the fault is a residual the solver steps
        // away from, not an abort.
        let system = system(vec![InputVariable::new("x", -1.0, 3.0)], &["ln(x) > 0"]);
        let point =
            seed(&system, &mut rng(), &Cancellation::never()).expect("the log's domain seeds");
        assert!(system.is_feasible(&point, 0.0), "{point:?}");
    }

    #[test]
    fn the_same_seed_finds_the_same_point() {
        let system = system(
            vec![
                InputVariable::new("x", -3.0, 3.0),
                InputVariable::new("y", -3.0, 3.0),
            ],
            &["sqr(x - 2) + sqr(y - 2) < 0.25", "x + y > 3.9"],
        );
        let first = seed(&system, &mut rng(), &Cancellation::never()).expect("seeds");
        let second = seed(&system, &mut rng(), &Cancellation::never()).expect("seeds");
        assert_eq!(first, second);
    }

    #[test]
    fn a_cancelled_search_does_not_start() {
        let system = system(vec![InputVariable::new("x", 0.0, 10.0)], &["x > 8"]);
        let (sender, receiver) = futures_channel::oneshot::channel::<super::super::Opening>();
        drop(receiver);
        assert!(seed(&system, &mut rng(), &Cancellation::watching(&sender)).is_none());
    }
}
