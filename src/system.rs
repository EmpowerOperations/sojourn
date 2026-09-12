//! A box and the constraints over it, compiled once and asked everything.
//!
//! The system is the compiled thing. Construction proves every constraint
//! binds by compiling it, and keeps the tape; it reads the equalities for
//! which coordinates are driven; it builds the incidence graph between
//! constraints and coordinates. Every point-level question a strategy asks —
//! is this point feasible, what interval may this coordinate take, put the
//! driven coordinates back on their surface — is answered here. The one thing
//! a solver call needs beyond the system, the SMT-LIB logic, travels with the
//! engine's ladder rather than being folded into the system. That is also what
//! lets [`repair`](crate::repair) be a plain function over a
//! `&ConstraintSystem`: nothing is compiled per call.
//!
//! The vocabulary lives here too: a [`Point`] in the box, an [`InputVariable`]
//! bounding one coordinate of it, and a [`ConstraintRef`] naming one
//! constraint the way a caller reads it.
//!
//! Immutable once built. Strategies borrow it; nothing about it is decided at
//! run time.

use faer::{Mat, MatRef};
use rand::RngExt;
use rand::rngs::Xoshiro256PlusPlus;

use anyhow::Result;

use crate::ast::GlobalId;
use crate::cvg::incidence::{ConstraintId, Incidence, Row};
use crate::cvg::interval::Interval;
use crate::cvg::{classify, interval};
use crate::diagnostics::CompilationFailure;
use crate::solve::{ConstraintSolver, Satisfiability};
use crate::{Ast, CompiledExpression, Schema};

/// A point in the input space, one value per variable in declaration order.
///
/// Positional rather than a name-to-value map: the JVM implementation allocated
/// a hash map per candidate inside a loop that oversamples a hundred to one, and
/// the schema already carries the names. It is also the shape a column-major
/// matrix wants, for when evaluation goes batched.
pub type Point = Vec<f64>;

/// One input variable and the range it may take.
#[derive(Debug, Clone, PartialEq)]
pub struct InputVariable {
    pub name: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

impl InputVariable {
    #[must_use]
    pub fn new(name: impl Into<String>, lower_bound: f64, upper_bound: f64) -> Self {
        Self {
            name: name.into(),
            lower_bound,
            upper_bound,
        }
    }

    #[must_use]
    pub fn contains(&self, value: f64) -> bool {
        (self.lower_bound..=self.upper_bound).contains(&value)
    }
}

/// One constraint, in a form a caller can read.
///
/// Not a syntax tree. A verdict is something a user reads — in a log line, in a UI
/// telling them their formulation conflicts — and handing back a syntax tree
/// makes them render it themselves. The index is there for anyone who wants to
/// find the original in the list they supplied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintRef {
    /// Position in the list given to [`ConstraintSystem::new`].
    pub index: usize,
    /// The constraint as it was written.
    pub source: String,
}

impl std::fmt::Display for ConstraintRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.source)
    }
}

/// One constraint, as written and as compiled.
///
/// Kept as a pair rather than two parallel lists because everything that
/// indexes one indexes the other by the same [`ConstraintId`], and two lists
/// aligned only by the loop that built them are one refactor from silently
/// disagreeing. The AST is what narrowing walks and the emitter renders; the
/// tape is what every feasibility check runs.
#[derive(Debug, Clone)]
pub(crate) struct Constraint {
    pub(crate) written: Ast,
    pub(crate) compiled: CompiledExpression,
}

/// A variable box and the constraints over it, proven to fit together.
///
/// The type exists because these two travelled as parallel slices that nothing
/// validated jointly, and because the properties that matter — how many degrees
/// of freedom are left, which variables another determines — are properties of
/// the *set*, not of any member. "System" is the word for constraints considered
/// together, as in a system of equations.
///
/// Construction is where a constraint naming an undeclared variable is caught,
/// which is what leaves [`solve`](ConstraintSystem::solve)'s `Result` about the
/// search and nothing else.
#[derive(Debug, Clone)]
pub struct ConstraintSystem {
    variables: Vec<InputVariable>,
    /// Every constraint as written and as compiled, at construction. Compiling
    /// is how a constraint is proved to bind, so the tape is kept rather than
    /// made again: a system is the compiled thing, and
    /// [`repair`](crate::repair) can be a plain function over one instead of
    /// a handle that owns a second copy.
    constraints: Vec<Constraint>,
    schema: Schema,
    /// Which coordinates are computed from the others, when any are. See
    /// [`classify`].
    plan: Option<classify::Plan>,
    /// Which constraints name which coordinates, both ways round.
    ///
    /// `slice` narrows one coordinate per move and needs only the constraints
    /// that mention it; without the graph that is a scan of every constraint
    /// each time, which on two hundred variables under two hundred constraints
    /// is forty thousand scans a sweep to do two hundred narrowings.
    incidence: Incidence,
}

/// A system that does not hold together.
#[derive(Debug, Clone, PartialEq)]
pub enum SystemError {
    /// A constraint's text is not a babel expression. Carries every problem the
    /// parser found, with spans, the way [`compile`](crate::compile) would.
    Unparsable {
        constraint: ConstraintRef,
        failure: CompilationFailure,
    },
    /// A constraint names a variable the box does not declare. It could never be
    /// satisfied, and saying so once beats saying it on every evaluation — which
    /// is what the JVM implementation did.
    Unbound {
        constraint: ConstraintRef,
        missing: Vec<String>,
    },
    /// A scalar expression where a constraint was wanted. It has no `<= 0`
    /// reading, so asserting one would invent a constraint nobody wrote.
    NotAConstraint { constraint: ConstraintRef },
    /// `x == sin(x)`, `x2 == x1 + x2/2 - x3/x4` — a variable named on both sides,
    /// so the equality is *implicit* in it: no reading of it yields `v = ...`.
    /// Rows C and F of the equality taxonomy, which turned out to be one thing.
    ///
    /// **A refusal, not a claim that nothing satisfies it.** `sin(x) == x/2` has
    /// three solutions and `x == x*x + 2` is an ordinary quadratic, so
    /// [`Satisfiability::Unsatisfiable`] would be saying something false. What is
    /// true is that nothing here can *drive* such a variable, and a search that
    /// cannot drive it falls back on whatever the sampler manages — which reads
    /// as a capability rather than the gap it is.
    ///
    /// **What is refused is a phrasing.** `x2 == x1 + x2/2` and `x2/2 - x1 == 0`
    /// describe the same set and only the first is implicit, so the message
    /// names the rearrangement rather than only the problem. `cvg_pools::simple_arithmetic`
    /// is the same fixture written the other way round and passes.
    ///
    /// Not to be confused with a *cycle*, which is a mutual dependency between
    /// two equations — `x1 == f(x2)` with `x2 == g(x1)`. `classify::plan` meets
    /// those and drives neither; they are legal, just not reducible.
    ///
    /// Refused at construction because the alternative is worse: a solver call
    /// and several thousand samples before answering `NotFound`, which tells a
    /// caller nothing about what to change.
    Implicit {
        constraint: ConstraintRef,
        variable: String,
    },
    /// `var[3]` against a box that declares two variables.
    ///
    /// Settled the moment a box was declared, and reported here rather than
    /// once per evaluation as `ProblemKind::DynamicIndexOutOfBounds` — which is
    /// where it used to surface, and is a runtime answer to a static question.
    SubscriptOutOfRange {
        constraint: ConstraintRef,
        /// The one-based index the source asked for.
        requested: i64,
        /// How many variables the box declares.
        available: usize,
    },
}

impl std::fmt::Display for SystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unparsable {
                constraint,
                failure,
            } => write!(f, "constraint {constraint} did not parse: {failure}"),
            Self::Unbound {
                constraint,
                missing,
            } => write!(
                f,
                "constraint {constraint} references {} which is not an input variable",
                missing.join(", ")
            ),
            Self::NotAConstraint { constraint } => write!(
                f,
                "{constraint} is a scalar expression, not a constraint: it has no truth value"
            ),
            Self::SubscriptOutOfRange {
                constraint,
                requested,
                available,
            } => write!(
                f,
                "constraint {constraint} reads var[{requested}], and the box                  declares {available} variable(s)"
            ),
            Self::Implicit {
                constraint,
                variable,
            } => write!(
                f,
                "constraint {constraint} is implicit in {variable}: it names \
                 {variable} on both sides, so nothing can solve it for \
                 {variable} without rearranging it first. Write {variable} on \
                 one side only - `a == b + a/2` is `a/2 - b == 0`"
            ),
        }
    }
}

impl std::error::Error for SystemError {}

impl ConstraintSystem {
    /// Parses every constraint, checks that each is one, and that each binds
    /// to the box.
    ///
    /// # Errors
    /// [`SystemError`] for the first constraint that does not fit. One rather
    /// than all: an unbound name is nearly always a typo, and a list of
    /// consequences is less use than the cause.
    pub fn new<S: Into<String>>(
        variables: Vec<InputVariable>,
        constraints: impl IntoIterator<Item = S>,
    ) -> Result<Self, SystemError> {
        let schema = Schema::new(variables.iter().map(|input| input.name.clone()));

        let mut resolved = Vec::new();
        let mut compiled = Vec::new();
        for (index, source) in constraints.into_iter().enumerate() {
            let source: String = source.into();
            let named = ConstraintRef {
                index,
                source: source.clone(),
            };
            let constraint = crate::parse(&source).map_err(|failure| SystemError::Unparsable {
                constraint: named.clone(),
                failure,
            })?;
            if !constraint.is_constraint() {
                return Err(SystemError::NotAConstraint { constraint: named });
            }

            // A schema exists here and nowhere earlier, so this is the first
            // moment `var[1]` can be told which variable it means. Resolving it
            // now is why nothing downstream has to: `smtlib` would resolve it
            // again and `classify` would refuse the whole constraint rather
            // than reason about it.
            let constraint = crate::frontend::rewrite::resolve_subscripts(constraint, &schema)
                .map_err(|out_of_range| SystemError::SubscriptOutOfRange {
                    constraint: named.clone(),
                    requested: out_of_range.requested,
                    available: out_of_range.available,
                })?;

            // Compiling is the binding check, and the tape it produces is the
            // one every strategy evaluates, so it is kept rather than redone.
            let tape = match crate::eval::bind(&constraint, &schema) {
                Ok(tape) => tape,
                Err(unbound) => {
                    return Err(SystemError::Unbound {
                        constraint: named,
                        missing: unbound.missing,
                    });
                }
            };

            // Last, because "you named a variable that does not exist" is a
            // better message than anything about shape when both are true of
            // the same constraint.
            if let classify::Shape::Implicit { variable } = classify::shape(&constraint) {
                return Err(SystemError::Implicit {
                    constraint: named,
                    variable: constraint.symbols()[variable.index()].clone(),
                });
            }
            resolved.push(constraint);
            compiled.push(tape);
        }

        let plan = classify::plan(&resolved, &schema);

        // What counts as affected by *any* move, whichever coordinate it
        // touched. Both entries here are soundness rather than efficiency.
        let mut always: Vec<ConstraintId> = Vec::new();

        // A computed subscript reads a column chosen by the point, so nothing
        // static says which and no symbol list names it. Skipping such a
        // constraint would mean skipping one that *had* changed.
        always.extend(
            resolved
                .iter()
                .enumerate()
                .filter(|(_, constraint)| constraint.contains_dynamic_lookup())
                .map(|(position, _)| ConstraintId(position)),
        );

        // `retract` rewrites every driven coordinate on every move, so whatever
        // names one of those is in play whichever axis was swept. Read off the
        // naming direction, which is why the graph is built before this.
        let incidence = Incidence::of(&resolved, &schema);
        if let Some(plan) = &plan {
            for driven in plan.driven() {
                always.extend_from_slice(incidence.naming(Row(*driven)));
            }
        }
        let incidence = incidence.with_always(&always);

        let constraints = resolved
            .into_iter()
            .zip(compiled)
            .map(|(written, compiled)| Constraint { written, compiled })
            .collect();

        Ok(Self {
            variables,
            constraints,
            schema,
            plan,
            incidence,
        })
    }

    #[must_use]
    pub fn variables(&self) -> &[InputVariable] {
        &self.variables
    }

    /// The constraints as written, in the order they were given.
    pub fn constraints(&self) -> impl ExactSizeIterator<Item = &str> {
        self.constraints
            .iter()
            .map(|constraint| constraint.written.source())
    }

    /// The constraints as parsed, for the emitter — the one consumer that
    /// needs the trees rather than the text.
    pub(crate) fn written(&self) -> impl ExactSizeIterator<Item = &Ast> {
        self.constraints
            .iter()
            .map(|constraint| &constraint.written)
    }

    #[must_use]
    pub(crate) const fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Searches for feasible samples with the default strategies.
    ///
    /// Sugar for [`ConstraintSolver::new().solve(system)`](ConstraintSolver::solve).
    /// Reach for the builder when the randomness or the strategy list has to be
    /// pinned, which is mostly tests.
    ///
    /// # Errors
    /// Anything that went *wrong*, as opposed to anything that was *concluded*.
    /// An unsatisfiable system is a [`Satisfiability`], not an error.
    pub async fn solve(self) -> Result<Satisfiability> {
        ConstraintSolver::new().solve(self).await
    }

    /// The constraint at `index`, as a caller reads it.
    pub(crate) fn named(&self, index: usize) -> ConstraintRef {
        ConstraintRef {
            index,
            source: self.constraints[index].written.source().to_owned(),
        }
    }
}

impl ConstraintSystem {
    /// The interval `coordinate` may take with every other coordinate held at
    /// its value in `point`.
    ///
    /// The declared box, narrowed by each constraint in turn through
    /// [`interval::narrow`]. **A superset of the feasible slice**, so a value
    /// drawn from it is still judged by [`is_feasible`](Self::is_feasible) like
    /// any other candidate — see [`interval`] for why that
    /// leaves the distribution alone.
    ///
    /// Constraints are applied in order and each sees what the previous ones
    /// concluded, so a system narrows further than any one of its constraints
    /// would. Nothing here iterates to a fixpoint: every coordinate but this one
    /// is a point, which leaves a second pass with nothing to tighten.
    pub(crate) fn slice(&self, point: &Point, coordinate: usize) -> Interval {
        self.slice_over(point, coordinate, None)
    }

    /// [`slice`](Self::slice), optionally ignoring constraints that mention a
    /// coordinate whose value is about to change.
    ///
    /// `settled[row]` says the point's value there is one to condition on. A
    /// constraint naming an unsettled coordinate is skipped, because narrowing
    /// against a value that is about to be overwritten conditions on a stale
    /// number — see [`retract`](Self::retract), which is the only caller that
    /// passes anything.
    fn slice_over(&self, point: &Point, coordinate: usize, settled: Option<&[bool]>) -> Interval {
        let input = &self.variables[coordinate];
        let mut interval = Interval::new(input.lower_bound, input.upper_bound);

        let coordinate = Row(coordinate);
        for id in self.incidence.naming(coordinate) {
            let rows = self.incidence.rows_of(*id);
            if let Some(settled) = settled
                && rows
                    .iter()
                    .any(|row| *row != coordinate && !settled[row.index()])
            {
                continue;
            }
            let wanted = rows
                .iter()
                .position(|row| *row == coordinate)
                .expect("`naming` lists only constraints that name the coordinate");

            // The constraint's symbols, in its own order, as intervals: a point
            // for everything held, and the running narrowing for the one asked
            // about.
            let globals: Vec<Interval> = rows
                .iter()
                .enumerate()
                .map(|(symbol, row)| {
                    if symbol == wanted {
                        interval
                    } else {
                        Interval::point(point[row.index()])
                    }
                })
                .collect();

            let wanted = u32::try_from(wanted).expect("fewer than four billion symbols");
            interval = interval.intersect(interval::narrow(
                &self.constraints[id.index()].written,
                &globals,
                GlobalId::from_index(wanted),
            ));
            if interval.is_empty() {
                break;
            }
        }

        interval
    }

    /// The coordinates a search may move, or `None` when nothing is driven.
    pub(crate) fn free_coordinates(&self) -> Option<&[usize]> {
        self.plan.as_ref().map(classify::Plan::free)
    }

    /// Recomputes every driven coordinate from the free ones, in place.
    ///
    /// A point on an equality surface leaves it under almost any move, because
    /// the surface has no volume — which is why a walker that moves every
    /// coordinate independently is reduced to jitter around wherever it started.
    /// Driving is the answer: move what is free, and compute the rest.
    ///
    /// # Why this samples rather than evaluates
    ///
    /// The obvious version assigns `y = f(free)` and is **wrong**, because
    /// babel has no bare equality: `y == f(x) +/- t` admits the whole band, and
    /// collapsing it to its centre line throws away a dimension of the feasible
    /// region. On `xi == 10.75 +/- 0.2` over two hundred variables that is not
    /// subtle — every coordinate pins to `10.75` and the pool returns the same
    /// point two hundred times, which is exactly what it did before this
    /// sampled.
    ///
    /// So it draws uniformly from `f(free) ± t`. That is not a fudge, it is a
    /// **Gibbs step**: with the other coordinates held, the feasible slice for a
    /// driven variable *is* that interval, and drawing uniformly from a
    /// conditional slice is the move that leaves the uniform distribution
    /// invariant. Evaluating to the centre would not.
    ///
    /// The slice comes from [`slice`](Self::slice) rather than from the one
    /// equality that defined the coordinate, so **every** constraint mentioning
    /// it has a say. The band `f(free) ± t` is what that equality contributes
    /// and the intersection can only be tighter, which turns draws that used to
    /// land outside the other constraints and be rejected into draws that
    /// cannot.
    ///
    /// Silent about failure by design. A coordinate whose slice comes back
    /// empty is left alone, and the candidate is judged exactly as an
    /// unretracted one would be. **Feasibility is never assumed from a
    /// successful retraction** — a wrong drive costs rejected moves, not wrong
    /// points, and that is what makes this safe to apply without proving it.
    ///
    /// Order still matters, and for the same reason: a driven coordinate may be
    /// defined in terms of another, and `slice` reads the point as it stands,
    /// so computing them out of order reads a stale value.
    pub(crate) fn retract(&self, point: &mut Point, rng: &mut Xoshiro256PlusPlus) {
        let Some(plan) = &self.plan else {
            return;
        };
        // A driven coordinate holds a value that is about to be replaced, so
        // narrowing against a constraint that mentions one still waiting its
        // turn conditions on a stale number. That is not merely wasteful, it
        // **destroys the freedom driving exists to exploit**: on
        // `y == sin(x) +/- t` with `z == y + 1 +/- t`, conditioning `y` on the
        // current `z` pins it within `t` of `z - 1`, and then `z` is pinned
        // within `t` of the new `y`. The pair shuffles by `t` a sweep instead
        // of travelling, and `two_coupled_equalities_are_traversed` measured it
        // as three occupied cells of eighty where twenty-four are wanted.
        //
        // So a coordinate becomes conditionable only once it has been drawn.
        // Everything free is conditionable from the start.
        let mut settled = vec![true; point.len()];
        for driven in plan.driven() {
            settled[*driven] = false;
        }

        for driven in plan.driven().iter().copied() {
            let slice = self.slice_over(point, driven, Some(&settled));
            if !slice.is_empty() {
                point[driven] = if slice.width() > 0.0 {
                    rng.random_range(slice.lo()..=slice.hi())
                } else {
                    slice.lo()
                };
            }
            settled[driven] = true;
        }
    }

    /// [`retract`](Self::retract) without the draw: every driven coordinate is
    /// clamped into its slice rather than drawn from it.
    ///
    /// The walker must draw, because it is producing a *sample* and the draw is
    /// what keeps the uniform distribution invariant. A repair is producing one
    /// point, near a given one, and must produce the same point every time it
    /// is asked — so it moves each driven coordinate the least distance that
    /// puts it inside its band, and no further. Same order, same settled mask,
    /// same silence on an empty slice, for the same reasons.
    pub(crate) fn settle(&self, point: &mut Point) {
        let Some(plan) = &self.plan else {
            return;
        };
        let mut settled = vec![true; point.len()];
        for driven in plan.driven() {
            settled[*driven] = false;
        }

        for driven in plan.driven().iter().copied() {
            let slice = self.slice_over(point, driven, Some(&settled));
            if !slice.is_empty() {
                point[driven] = point[driven].clamp(slice.lo(), slice.hi());
            }
            settled[driven] = true;
        }
    }

    /// The constraints as compiled, in order: what a backend over the tape
    /// renders.
    #[cfg_attr(
        not(feature = "gpu"),
        allow(dead_code, reason = "the GPU sieve is the only caller")
    )]
    pub(crate) fn compiled(&self) -> impl ExactSizeIterator<Item = &CompiledExpression> {
        self.constraints
            .iter()
            .map(|constraint| &constraint.compiled)
    }

    /// The declared box as `(low, high)` per variable: the shape
    /// [`fill_box`](crate::fill_box) takes.
    pub(crate) fn box_bounds(&self) -> Vec<(f64, f64)> {
        self.variables
            .iter()
            .map(|input| (input.lower_bound, input.upper_bound))
            .collect()
    }

    /// How badly the worst constraint is violated, or `None` if the point is
    /// outside the box or cannot be evaluated.
    ///
    /// [`is_feasible`](Self::is_feasible) asks a yes-or-no question; this asks
    /// *how far*, which is what the `<= 0` convention makes available and what a
    /// repair needs in order to know which way to step.
    pub(crate) fn worst_residual(&self, point: &Point) -> Option<f64> {
        if !self.in_box(point) {
            return None;
        }
        let mut worst = f64::NEG_INFINITY;
        for constraint in &self.constraints {
            worst = worst.max(constraint.compiled.eval_row(point).ok()?);
        }
        Some(worst)
    }

    /// The columns of `candidates` that are inside the box and satisfy every
    /// constraint, in column order, copied out as points.
    ///
    /// The batched twin of [`is_feasible`](Self::is_feasible), for the sources
    /// that propose thousands of independent candidates at once. A candidate
    /// whose evaluation faults — `sqrt` of a negative, a subscript out of
    /// range — is one the constraint does not hold for, exactly as
    /// `is_feasible` treats an `Err` per point.
    pub(crate) fn feasible_columns(&self, candidates: MatRef<'_, f64>) -> Vec<Point> {
        let rows = self.variables.len();
        if candidates.nrows() != rows {
            return Vec::new();
        }
        let columns = candidates.ncols();

        let mut pass: Vec<bool> = (0..columns)
            .map(|column| {
                self.variables
                    .iter()
                    .enumerate()
                    .all(|(row, input)| input.contains(candidates[(row, column)]))
            })
            .collect();

        for constraint in &self.constraints {
            constraint.compiled.holds(candidates, &mut pass).expect(
                "`ConstraintSystem::new` proved every constraint binds, and candidates are shaped by the same box",
            );
        }

        (0..columns)
            .filter(|&column| pass[column])
            .map(|column| (0..rows).map(|row| candidates[(row, column)]).collect())
            .collect()
    }

    /// [`retract`](Self::retract) over every column of a candidate batch.
    ///
    /// Returns immediately when nothing is driven, which is every problem
    /// without an equality — so the batch path pays nothing for this.
    pub(crate) fn retract_columns(&self, candidates: &mut Mat<f64>, rng: &mut Xoshiro256PlusPlus) {
        if self.plan.is_none() {
            return;
        }
        let rows = candidates.nrows();
        for column in 0..candidates.ncols() {
            let mut point: Point = (0..rows).map(|row| candidates[(row, column)]).collect();
            self.retract(&mut point, rng);
            for (row, value) in point.into_iter().enumerate() {
                candidates[(row, column)] = value;
            }
        }
    }

    /// Whether a point is inside the box and satisfies every constraint, with
    /// `clearance` box widths of room to spare in every coordinate direction.
    ///
    /// At `clearance == 0` this is the plain question. Above it, each
    /// constraint must also hold at the point moved `clearance * width` either
    /// way along each variable it names, and the box shrinks by the same step
    /// — see [`holds`](Self::holds). That is exactly "survives any
    /// per-coordinate perturbation smaller than the step", which is what a
    /// caller's own normalise-and-back round trip inflicts, and on a convex
    /// region the axis neighbours span the whole L1 ball of that radius in
    /// box-normalised coordinates: the ball in the metric
    /// [`repair`](crate::repair) measures distance in.
    pub(crate) fn is_feasible(&self, point: &Point, clearance: f64) -> bool {
        if point.len() != self.variables.len() {
            return false;
        }
        if !(0..self.variables.len()).all(|coordinate| self.within(point, coordinate, clearance)) {
            return false;
        }
        // `eval_row` rather than a one-column batch. This question is asked one
        // point at a time by nature — the walker cannot propose its next
        // candidate until it has judged this one — and wrapping each point in a
        // matrix cost five times the evaluation: `p118` ran 32s against 6s.
        (0..self.constraints.len()).all(|index| self.holds(ConstraintId(index), point, clearance))
    }

    /// [`is_feasible`](Self::is_feasible), for a point that differs from a
    /// **feasible** one only in `moved` and in whatever [`retract`](Self::retract)
    /// recomputed.
    ///
    /// # Why this is exact and not an approximation
    ///
    /// A constraint that names none of the coordinates that moved evaluates to
    /// the number it evaluated to before, and that number was `<= 0` — so
    /// re-deriving it is arithmetic nobody reads. Only the constraints in
    /// `affected[moved]` can have changed, and the box only needs re-checking
    /// where the point actually moved. The same holds with a clearance: a
    /// constraint's neighbours are along the variables it names, and those
    /// residuals were judged with the constraint.
    ///
    /// # Why it matters
    ///
    /// The walker judges one candidate per proposal and every axis move touches
    /// one coordinate, so the full check was re-evaluating the whole system to
    /// learn about one column. On `top_corner_200d` that is two hundred
    /// constraints asked in order to consult one, and the walker's cost is
    /// almost entirely this call.
    ///
    /// # The precondition is the caller's
    ///
    /// The point this was derived from **must** have been feasible, with the
    /// same clearance. `advance` holds that by the chain's invariant. Anywhere
    /// that does not, use [`is_feasible`](Self::is_feasible).
    pub(crate) fn is_feasible_after(&self, point: &Point, moved: usize, clearance: f64) -> bool {
        if point.len() != self.variables.len() {
            return false;
        }
        if !self.within(point, moved, clearance) {
            return false;
        }
        if let Some(plan) = &self.plan {
            for driven in plan.driven() {
                if !self.within(point, *driven, clearance) {
                    return false;
                }
            }
        }

        self.incidence
            .affected(Row(moved))
            .iter()
            .all(|id| self.holds(*id, point, clearance))
    }

    /// Whether `point[coordinate]` is inside its bounds with `clearance` box
    /// widths to spare on each side. A zero-width coordinate has no room to
    /// ask for and is judged on containment alone.
    fn within(&self, point: &Point, coordinate: usize, clearance: f64) -> bool {
        let variable = &self.variables[coordinate];
        let value = point[coordinate];
        let step = clearance * (variable.upper_bound - variable.lower_bound);
        if step > 0.0 {
            variable.contains(value - step) && variable.contains(value + step)
        } else {
            variable.contains(value)
        }
    }

    /// Whether one constraint is satisfied at `point`, and — for a positive
    /// clearance — at `point` moved `clearance * width` either way along each
    /// variable the constraint names.
    ///
    /// Babel's boolean rewrite yields a residual whose sign carries the truth
    /// value: `<= 0` is satisfied. A non-finite residual is an `Err` and not a
    /// pass. The neighbours are along the constraint's own variables only,
    /// because moving any other coordinate leaves its residual where it was;
    /// that is what lets [`is_feasible_after`](Self::is_feasible_after) judge
    /// a move with the clearance by consulting the affected constraints alone.
    fn holds(&self, id: ConstraintId, point: &Point, clearance: f64) -> bool {
        let passes = |point: &Point| {
            self.constraints[id.index()]
                .compiled
                .eval_row(point)
                .ok()
                .is_some_and(|residual| residual <= 0.0)
        };
        if !passes(point) {
            return false;
        }
        if clearance == 0.0 {
            return true;
        }
        let mut neighbour = point.clone();
        for row in self.incidence.rows_of(id) {
            let coordinate = row.index();
            let variable = &self.variables[coordinate];
            let step = clearance * (variable.upper_bound - variable.lower_bound);
            if step <= 0.0 {
                continue;
            }
            for sign in [-1.0, 1.0] {
                neighbour[coordinate] = point[coordinate] + sign * step;
                if !passes(&neighbour) {
                    return false;
                }
            }
            neighbour[coordinate] = point[coordinate];
        }
        true
    }

    /// The points among `points` that are feasible, in order.
    ///
    /// What a caller's hints and a solver's witness go through before they
    /// count as points in hand: neither is trusted, only judged.
    pub(crate) fn keep_feasible(&self, points: Vec<Point>) -> Vec<Point> {
        points
            .into_iter()
            .filter(|point| self.is_feasible(point, 0.0))
            .collect()
    }

    /// Nudges a solver's witness back onto the feasible side of `f64`.
    ///
    /// A solver reasons in **exact real arithmetic** and answers with a witness
    /// that is exactly on a boundary — asked for `x == pi +/- 0.001` it
    /// returns exactly `pi - 0.001`, because a boundary is the simplest
    /// solution there is. The pool then re-checks in `f64`, where `pi`, the
    /// tolerance, and the subtraction each round, and the point lands a hair
    /// outside. Discarding it wastes the entire solver call over an error in
    /// the last place.
    ///
    /// This is not a general-purpose repair and does not pretend to be; that
    /// is [`repair`](crate::repair), which starts from anywhere in the box.
    /// This is a bounded coordinate sweep: for each variable, try a step of a
    /// few ulps each way and keep it if the worst residual falls. That reaches
    /// a point which is *barely* outside, which is the only case a solver
    /// witness produces. It will not rescue a point that is genuinely
    /// infeasible, and it should not.
    ///
    /// Returns `None` when the point cannot be brought inside, which is then
    /// the honest answer rather than a silent near-miss.
    pub(crate) fn adjusted(&self, mut point: Point) -> Option<Point> {
        if self.is_feasible(&point, 0.0) {
            return Some(point);
        }

        for sweep in 0..ADJUST_SWEEPS {
            let mut improved = false;

            for index in 0..point.len() {
                let before = self.worst_residual(&point)?;
                let original = point[index];

                // Growing the step across sweeps: an ulp first, because that
                // is what a boundary witness misses by, then wider in case the
                // rounding compounded through a longer expression.
                let step = ulps(original, 1 << (2 * sweep));

                for candidate in [original + step, original - step] {
                    point[index] = candidate;
                    let better = self
                        .worst_residual(&point)
                        .is_some_and(|after| after < before);
                    if better {
                        improved = true;
                        break;
                    }
                    point[index] = original;
                }
            }

            if self.is_feasible(&point, 0.0) {
                return Some(point);
            }
            if !improved {
                break;
            }
        }

        None
    }

    fn in_box(&self, point: &Point) -> bool {
        point.len() == self.variables.len()
            && self
                .variables
                .iter()
                .zip(point)
                .all(|(input, value)| input.contains(*value))
    }
}

/// How many coordinate sweeps an adjustment gets before it gives up.
///
/// A near-miss is a rounding error, so it yields in one or two passes or it was
/// never a near-miss. This is a cap on wasted work rather than a tuning knob.
const ADJUST_SWEEPS: usize = 4;

/// `count` units in the last place of `value`, as a distance.
///
/// Scaled to the value rather than absolute, because a witness near `1e-9` and
/// one near `1e9` miss by wildly different amounts and the same absolute step
/// would be useless for one and enormous for the other.
fn ulps(value: f64, count: u32) -> f64 {
    let magnitude = if value == 0.0 { 1.0 } else { value.abs() };
    f64::from(count) * (magnitude.next_up() - magnitude)
}

#[cfg(test)]
pub(crate) mod tests {
    use faer::Mat;

    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::Xoshiro256PlusPlus;

    use crate::cvg::incidence::{ConstraintId, Row};
    use crate::{ConstraintSystem, InputVariable, Point};

    pub(crate) fn system(inputs: Vec<InputVariable>, sources: &[&str]) -> ConstraintSystem {
        ConstraintSystem::new(inputs, sources.iter().copied()).expect("the fixture binds")
    }

    fn one_variable(source: &str) -> ConstraintSystem {
        system(vec![InputVariable::new("x1", 0.0, 10.0)], &[source])
    }

    /// Points as a matrix, one column each: the shape the batched judge takes.
    fn points_to_matrix(points: &[Point], rows: usize) -> Mat<f64> {
        Mat::from_fn(rows, points.len(), |row, column| points[column][row])
    }

    #[test]
    fn the_restricted_feasibility_check_agrees_with_the_full_one() {
        let fixtures: [(Vec<InputVariable>, &[&str]); 4] = [
            (
                vec![
                    InputVariable::new("x1", -10.0, 10.0),
                    InputVariable::new("x2", -10.0, 10.0),
                    InputVariable::new("x3", -10.0, 10.0),
                ],
                // Separable: each constraint names one coordinate, which is the
                // case the skipping is worth the most on.
                &["x1 < 3", "x2 > -4", "x3 < 8"],
            ),
            (
                vec![
                    InputVariable::new("x1", -10.0, 10.0),
                    InputVariable::new("x2", -10.0, 10.0),
                    InputVariable::new("x3", -10.0, 10.0),
                ],
                // Coupled, so most constraints are in play whatever moved.
                &["x1 + x2 < 3", "x2 * x3 > -20", "x1 - x3 < 6"],
            ),
            (
                vec![
                    InputVariable::new("x", -4.0, 4.0),
                    InputVariable::new("y", -8.0, 8.0),
                    InputVariable::new("z", -8.0, 8.0),
                ],
                // Drives: `retract` moves `y` and `z` whatever was swept, so
                // `affected` has to carry the constraints naming them.
                &["y == sin(x) +/- 0.05", "z == y + 1 +/- 0.05", "x + z < 6"],
            ),
            (
                vec![
                    InputVariable::new("x1", -10.0, 10.0),
                    InputVariable::new("x2", -10.0, 10.0),
                ],
                // A subscript reads a column the expression never names.
                &["var[1] + var[2] < 4"],
            ),
        ];

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x00A6_9EED);
        for (inputs, sources) in fixtures {
            let bounds: Vec<(f64, f64)> = inputs
                .iter()
                .map(|input| (input.lower_bound, input.upper_bound))
                .collect();
            let system = system(inputs, sources);

            let mut compared = 0_usize;
            for _ in 0..20_000 {
                let mut point: Point = bounds
                    .iter()
                    .map(|(low, high)| rng.random_range(*low..=*high))
                    .collect();
                // Retracted first, exactly as the walker reaches a feasible
                // point: on a system of tight equalities the feasible set has
                // no volume, and uniform draws found three of twenty thousand.
                system.retract(&mut point, &mut rng);
                // The precondition: the restricted check is only sound about a
                // point derived from a feasible one.
                if !system.is_feasible(&point, 0.0) {
                    continue;
                }

                let moved = rng.random_range(0..point.len());
                let mut candidate = point.clone();
                candidate[moved] = rng.random_range(bounds[moved].0..=bounds[moved].1);
                system.retract(&mut candidate, &mut rng);

                assert_eq!(
                    system.is_feasible_after(&candidate, moved, 0.0),
                    system.is_feasible(&candidate, 0.0),
                    "{sources:?}: moving coordinate {moved} of {point:?} to \
                     {candidate:?} is judged differently by the two checks"
                );
                compared += 1;
            }
            assert!(
                compared > 100,
                "{sources:?}: only {compared} candidates were derived from a \
                 feasible point, so this asserted almost nothing"
            );
        }
    }

    /// A computed subscript reads a column nothing names statically, so the
    /// constraint holding it can never be skipped.
    ///
    /// This is asserted structurally rather than by sampling, because
    /// `var[n]` needs `n` to land on a whole number and a uniform draw almost
    /// never does — the sampling test above would have found no feasible point
    /// to compare, and reported that rather than this.
    #[test]
    fn a_computed_subscript_is_never_skipped() {
        let system = system(
            vec![
                InputVariable::new("n", 1.0, 2.0),
                InputVariable::new("x2", -10.0, 10.0),
            ],
            &["var[n] < 4", "x2 < 3"],
        );
        assert!(
            system.constraints[0].written.contains_dynamic_lookup(),
            "the fixture stopped exercising a computed subscript"
        );

        for coordinate in 0..2 {
            assert!(
                system
                    .incidence
                    .affected(Row(coordinate))
                    .contains(&ConstraintId(0)),
                "coordinate {coordinate} may move the column `var[n]` reads"
            );
        }
    }

    /// **The invariant `slice` exists to keep.**
    ///
    /// A feasible point satisfies every constraint, so each of its coordinates
    /// is in that coordinate's true feasible slice — and the narrowing is a
    /// *superset* of that slice, so it must contain the coordinate too. If it
    /// ever does not, the walker is being handed an interval that excludes
    /// where it is standing, and the points it cannot propose are gone from the
    /// answer silently.
    ///
    /// This is the check that separates the two ways narrowing can be wrong.
    /// Too wide only costs a rejected proposal; too narrow biases, and shows up
    /// here.
    #[test]
    fn a_feasible_point_is_inside_every_slice_it_sits_in() {
        let fixtures: [(Vec<InputVariable>, &[&str]); 5] = [
            (
                vec![
                    InputVariable::new("x1", -10.0, 10.0),
                    InputVariable::new("x2", -10.0, 10.0),
                ],
                &["x1 + x2 == 3 +/- 0.1"],
            ),
            (
                vec![
                    InputVariable::new("x1", -10.0, 10.0),
                    InputVariable::new("x2", -10.0, 10.0),
                ],
                &["x1 + x2 < 3", "x1 - x2 > -4"],
            ),
            (
                vec![
                    InputVariable::new("x1", 0.5, 10.0),
                    InputVariable::new("x2", 0.5, 10.0),
                ],
                &["x1 / x2 < 4", "ln(x1) < 2"],
            ),
            (
                vec![
                    InputVariable::new("x1", -10.0, 10.0),
                    InputVariable::new("x2", -10.0, 10.0),
                ],
                &["x1 * x2 == 6 +/- 0.5"],
            ),
            (vec![InputVariable::new("x1", 10.0, 11.0)], &["x1 > 10.5"]),
        ];

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0051_1CE5);
        for (inputs, sources) in fixtures {
            let bounds: Vec<(f64, f64)> = inputs
                .iter()
                .map(|input| (input.lower_bound, input.upper_bound))
                .collect();
            let system = system(inputs, sources);

            let mut feasible_seen = 0_usize;
            for _ in 0..20_000 {
                let point: Point = bounds
                    .iter()
                    .map(|(low, high)| rng.random_range(*low..=*high))
                    .collect();
                if !system.is_feasible(&point, 0.0) {
                    continue;
                }
                feasible_seen += 1;
                for coordinate in 0..point.len() {
                    let slice = system.slice(&point, coordinate);
                    assert!(
                        slice.contains(point[coordinate]),
                        "{sources:?}: the feasible point {point:?} has coordinate \
                         {coordinate} narrowed out of [{}, {}]",
                        slice.lo(),
                        slice.hi()
                    );
                }
            }
            assert!(
                feasible_seen > 0,
                "{sources:?}: no feasible point was drawn, so this asserted nothing"
            );
        }
    }

    /// **`slice` can only tighten what the equality already said.**
    ///
    /// `retract` used to draw from `f(free) ± t`, evaluated from the one
    /// equality that defined the coordinate. It now draws from `slice`, which
    /// starts at the declared box and applies every constraint naming that
    /// coordinate — the defining equality among them. So the new interval is
    /// contained in the old one, and the change is a narrowing rather than a
    /// different claim.
    ///
    /// The band is computed here from **Rust's own `sin`** rather than read off
    /// a `Drive`. That is the point: nothing in production carries the
    /// definition any more, so a test that asked production what the band was
    /// would be asking the thing under test. If this ever fails, `slice` is
    /// admitting values the equality forbids.
    const TOLERANCE: f64 = 0.05;

    #[test]
    fn a_slice_is_never_wider_than_the_band_its_equality_defines() {
        let inputs = vec![
            InputVariable::new("x", -4.0, 4.0),
            InputVariable::new("y", -8.0, 8.0),
            InputVariable::new("z", -8.0, 8.0),
        ];
        let bounds: Vec<(f64, f64)> = inputs
            .iter()
            .map(|input| (input.lower_bound, input.upper_bound))
            .collect();
        let system = system(inputs, &["y == sin(x) +/- 0.05", "z == y + 1 +/- 0.05"]);

        /// A driven coordinate and what its equality says its centre is.
        /// `x` is free and has no band, so it is absent.
        type Band = (usize, fn(&Point) -> f64);

        let bands: [Band; 2] = [(1, |point| point[0].sin()), (2, |point| point[1] + 1.0)];

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0000_BA4D);
        for _ in 0..5_000 {
            let point: Point = bounds
                .iter()
                .map(|(low, high)| rng.random_range(*low..=*high))
                .collect();

            for (position, centre_of) in bands {
                let centre = centre_of(&point);
                let slice = system.slice(&point, position);
                if slice.is_empty() {
                    continue;
                }
                // Narrowing pads outward once per operation it passes through,
                // so the interval exceeds the band by a rounding step per node.
                // The slack is *relative* rather than counted in ulps, because
                // an ulp is not a stable unit across a subtraction that
                // cancels: the padding is added while the intermediate is
                // around `centre`, and the endpoint it lands on can be an order
                // of magnitude smaller, where ulps are an order of magnitude
                // finer. Measured in ulps of the endpoint, two ulps of real
                // padding read as nine.
                let slack = 16.0 * f64::EPSILON * (1.0 + centre.abs() + TOLERANCE);
                assert!(
                    slice.lo() >= centre - TOLERANCE - slack
                        && slice.hi() <= centre + TOLERANCE + slack,
                    "coordinate {position} narrowed to [{}, {}], outside the                      band {centre} +/- {TOLERANCE} its equality defines",
                    slice.lo(),
                    slice.hi()
                );
            }
        }
    }

    /// A witness one ulp outside is brought in; one genuinely outside is not.
    ///
    /// The first case is what a solver actually produces. Asked for
    /// `x1 == pi +/- 0.001` Z3 answers with the *boundary* — exactly
    /// `pi - 0.001` — because a boundary is the simplest solution there is. It
    /// reasons in exact reals; the pool re-checks in `f64`, where `pi`, the
    /// tolerance and the subtraction each round, and the point lands a hair
    /// outside. Before this existed the whole solver call was thrown away over
    /// that, and `cvg_pools::constants` passed only because the *previous*
    /// encoding happened to make Z3 pick the other edge, where the rounding
    /// went the other way. Luck, not correctness.
    ///
    /// The second case is the one that matters more: repair must not rescue a
    /// point that is simply infeasible, or `Unsatisfiable` stops meaning
    /// anything.
    #[test]
    fn a_boundary_witness_is_adjusted_and_a_wrong_one_is_not() {
        let system = one_variable("x1 == pi +/- 0.001");

        // The value Z3 actually returns, as a decimal parsed back into f64 —
        // not `PI - 0.001`, which Rust computes to a *different* f64 and which
        // happens to land inside. That difference is the entire bug.
        let edge: f64 = "3.140592653589793".parse().expect("a literal");
        assert!(
            !system.is_feasible(&vec![edge], 0.0),
            "this test is pointless unless the boundary really does miss"
        );
        let adjusted_edge = system
            .adjusted(vec![edge])
            .expect("a near-miss should be adjusted");
        assert!(system.is_feasible(&adjusted_edge, 0.0));
        assert!(
            (adjusted_edge[0] - edge).abs() < 1e-12,
            "the adjustment moved the point {} away from the witness, which is not a nudge",
            (adjusted_edge[0] - edge).abs()
        );

        assert!(
            system.adjusted(vec![7.0]).is_none(),
            "a point nowhere near the band was 'adjusted' into feasibility"
        );
    }

    /// `worst_residual` has to grade, not just judge — a repair steps downhill
    /// and there is no hill in a boolean.
    #[test]
    fn the_worst_residual_is_graded() {
        let system = one_variable("x1 > 4");

        let near = system.worst_residual(&vec![3.9]).expect("inside the box");
        let far = system.worst_residual(&vec![1.0]).expect("inside the box");
        assert!(
            near < far,
            "{near} should be a smaller violation than {far}"
        );
        assert!(system.worst_residual(&vec![5.0]).is_some_and(|r| r <= 0.0));
        assert!(
            system.worst_residual(&vec![99.0]).is_none(),
            "outside the box is not a residual"
        );
    }

    /// A grid of candidates, some deliberately outside the box.
    fn candidates() -> Vec<Point> {
        let mut points = Vec::new();
        for i in 0..40 {
            let x1 = f64::from(i) * 0.3 - 1.0; // -1 .. 10.7, past both ends of 0..10
            let x2 = f64::from(i % 7) - 3.0;
            points.push(vec![x1, x2]);
        }
        points
    }

    #[test]
    fn batched_judging_agrees_with_per_point_is_feasible() {
        let system = system(
            vec![
                InputVariable::new("x1", 0.0, 10.0),
                InputVariable::new("x2", -5.0, 5.0),
            ],
            &["x1 > 4", "ln(x1) < 2", "x2 * x2 < 5"],
        );

        let points = candidates();
        let matrix = points_to_matrix(&points, 2);
        let batched = system.feasible_columns(matrix.as_ref());
        let one_at_a_time = system.keep_feasible(points);

        assert!(
            !batched.is_empty(),
            "the grid should contain feasible points"
        );
        assert_eq!(batched, one_at_a_time);
    }

    /// `sqrt(x1 - 5)` is NaN for every candidate below five, and the front end
    /// leaves it alone because the root is not the whole side of the comparison
    /// (`ln(x1 - 5) < 0` would be inverted into plain bounds and never fault).
    /// Those candidates are infeasible; the batch still returns the ones above.
    #[test]
    fn a_faulting_candidate_is_infeasible_rather_than_fatal() {
        let system = one_variable("sqrt(x1 - 5) + x1 < 6");

        let points: Vec<Point> = (0..100).map(|i| vec![f64::from(i) * 0.1]).collect();
        let matrix = points_to_matrix(&points, 1);

        // The strict evaluator refuses the batch outright: that is what lenient
        // judging exists to get past.
        let strict = crate::compile(
            system.constraints().next().unwrap(),
            system.schema().names(),
        )
        .unwrap();
        assert!(strict.eval(matrix.as_ref()).is_err());

        let feasible = system.feasible_columns(matrix.as_ref());
        assert!(!feasible.is_empty());
        for point in &feasible {
            assert!(point[0] >= 5.0 && point[0] < 6.0, "{point:?}");
        }
        assert_eq!(feasible, system.keep_feasible(points));
    }

    #[test]
    fn a_candidate_matrix_of_the_wrong_height_yields_nothing() {
        let system = one_variable("x1 > 1");
        let wrong = Mat::from_fn(2, 5, |_, _| 5.0);
        assert!(system.feasible_columns(wrong.as_ref()).is_empty());
    }
}
