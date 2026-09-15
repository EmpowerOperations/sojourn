//! Sojourn — constrained random vector generation over a small expression language, babel.
//!
//! Declare a box and the constraints over it, solve, and take samples:
//!
//! ```no_run
//! use sojourn::{ConstraintSystem, InputVariable};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let system = ConstraintSystem::new(
//!     vec![InputVariable::new("x", -2.0, 2.0), InputVariable::new("y", -2.0, 2.0)],
//!     ["x^2 + y^2 < 1", "x + y > 0.5"],
//! )?;
//!
//! // The defaults. For anything else — a pinned seed, a budget, a strategy
//! // list — build the solver yourself: `ConstraintSolver::new().with_seed(42)
//! // .solve(&system)`. Either way the system is borrowed; the search takes a
//! // copy, and this handle keeps its own.
//! let mut region = sojourn::solve(&system).await?;
//!
//! // One column per sample, one row per variable, in the order declared;
//! // `take` waits for them, `try_take` does not.
//! let samples = region.take(256);
//!
//! // A point that is not a sample, brought onto the region at the nearest
//! // feasible point (Euclidean, over box-normalised coordinates), `1e-12`
//! // box widths inside every wall. A function of the system, the point
//! // and the clearance alone.
//! let repaired = region.repair(&[1.5, 1.5], 1e-12)?;
//! # let _ = (samples, repaired);
//! # Ok(())
//! # }
//! ```
//!
//! One expression can also be compiled and evaluated over a batch on its own:
//!
//! ```ignore
//! let compiled = sojourn::compile("x1 + x2 > 20 - x3^2", &["x1", "x2", "x3"])?;
//!
//! // One column per sample, one row per variable, in the order given.
//! let residuals = compiled.eval(samples.as_ref())?;
//! ```
//!
//! Source text goes in; nothing hands back a syntax tree. Two consumers parse
//! it: the evaluator, which [`compile`]s an expression against a list of
//! variable names and runs it over a batch, and the constrained vector
//! generator, which reads the structure of a set of constraints to search for
//! points that satisfy them — a [`ConstraintSystem`] solved by a
//! [`ConstraintSolver`] into a [`FeasibleRegion`], which hands out samples and
//! repairs a point that is not one of them.
//! `src/README.md` has the picture.
//!
//! Boolean expressions evaluate to a scalar whose *sign* carries the truth
//! value: `<= 0` is true, `> 0` is false. That is the canonical `g(x) <= 0`
//! constraint form, so a violated constraint reports how badly it was violated.

// Crate-private while the shape is still settling; goes public when the
// pluggable rewriter needs it.
mod ast;
mod cvg;
pub mod diagnostics;
mod eval;
mod frontend;
mod repair;
mod solve;
mod system;

pub(crate) use eval::Schema;
pub use eval::{CompiledExpression, compile};
pub(crate) use frontend::{Ast, parse};
pub use repair::RepairError;
pub use solve::{
    ConstraintSolver, DEFAULT_GPU_PROPOSAL_BUDGET, DEFAULT_PROPOSAL_BUDGET, DEFAULT_PRUNE_BUDGET,
    FeasibleRegion, GPU_VARIABLE, Infeasibility, Status,
};
pub use system::{ConstraintRef, ConstraintSystem, InputVariable, Point, SystemError};

// Test plumbing: reachable, undocumented, unpromised. Each exists so that a
// fixture in `tests/` can pin one strategy or measure one stage alone.
#[doc(hidden)]
pub use cvg::sampling::fill_box;

#[cfg(feature = "gpu")]
#[doc(hidden)]
pub use cvg::gpu;

#[doc(hidden)]
pub use solve::{DEFAULT_STRATEGIES, Strategy};

/// Whether `name` is a legal Babel variable name.
///
/// Babel accepts Unicode identifiers, so `π`, `测试` and `☕` are all legal.
#[must_use]
pub fn is_legal_variable_name(name: &str) -> bool {
    !name.is_empty() && frontend::parses_as_variable(name)
}

pub async fn solve(system: &ConstraintSystem) -> Result<FeasibleRegion, Infeasibility> {
    ConstraintSolver::new().solve(system).await
}
