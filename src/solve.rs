//! The solve: a builder for how hard to search, and the region a search
//! hands back.
//!
//! [`ConstraintSolver`] is every knob — the seed, the budgets, the strategy
//! list — with a default for each, and one call,
//! [`solve`](ConstraintSolver::solve), that runs the engine's opening to a
//! verdict. [`FeasibleRegion`] is what a satisfied search returns: the
//! system it was solved over and the feasible points the opening found, from
//! which a space-filling design is [`sample`](FeasibleRegion::sample)d and
//! any point is [`repair`](FeasibleRegion::repair)ed. The verdicts say what a
//! search concluded, and whether that was a proof or a shrug. The engine
//! itself is `cvg`.

use faer::{Mat, MatRef};
use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;

use crate::cvg;
use crate::cvg::walking::HitAndRunWalker;
use crate::cvg::{Ladder, Opening, local};
use crate::repair::RepairError;
use crate::{ConstraintRef, ConstraintSystem, Point};

/// The seed the reference point's local solve draws its extra starts from,
/// when the opening did not run one.
///
/// A constant, so the reference is a function of the system alone: the same
/// on every solve, on every machine, whatever the solver's own seed found.
/// The first start is the box centre and needs no draw; this seeds the ones
/// after it.
const REFERENCE_SEED: u64 = 0x5E_ED_0F_1D;

/// Why no sample was produced — and whether that is a proof or a shrug.
///
/// The one way [`solve`](crate::solve) fails to return a region: there is
/// none to return, or none could be found. Everything else that can go wrong
/// in a search is a bug in this crate or a failure of the host, and is a
/// panic.
///
/// Kept as two variants rather than a `proved: bool` because they are different
/// sentences to whoever reads the result. *"Your constraints conflict, here are
/// the three involved"* sends someone to rewrite a formulation. *"We found
/// nothing"* sends them to widen a tolerance or wait longer. A flag invites
/// code that ignores it and says the first when it means the second.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Infeasibility {
    /// Interval reasoning proved no point exists, and these are the
    /// constraints its proof used.
    ///
    /// A list rather than one culprit: a contradiction is a *relationship*.
    /// `x > 8` is perfectly satisfiable right up until `x < 2` appears, and
    /// naming either alone would be picking arbitrarily. The list is the
    /// trace of the proof — every constraint that narrowed a coordinate the
    /// emptied one depended on — so it is the constraints actually used
    /// rather than every one present.
    Proved { blamed: Vec<ConstraintRef> },
    /// Sampling found nothing and nothing could be proved. **This is not a
    /// claim that the region is empty.**
    ///
    /// `unexpressed` names the constraints interval reasoning could conclude
    /// nothing from — a computed subscript, say — which is often the reason:
    /// a region defined by something no enclosure can see is found only by
    /// luck. Empty when every constraint said *something* and it still did
    /// not add up to a proof, which is what a contradiction too thin or too
    /// algebraic for intervals looks like.
    NotFound { unexpressed: Vec<ConstraintRef> },
}

/// The sentence each arm is: a conflict names the constraints in it, and a
/// shrug says what was tried and, when some constraint was beyond interval
/// reasoning, which. `Display` by hand because the shrug's second sentence
/// is conditional; the `Error` impl is derived on it.
impl std::fmt::Display for Infeasibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let listed = |constraints: &[ConstraintRef]| {
            constraints
                .iter()
                .map(|constraint| format!("`{}`", constraint.source))
                .collect::<Vec<_>>()
                .join(", ")
        };
        match self {
            Self::Proved { blamed } => write!(
                f,
                "no point satisfies these constraints together: {}",
                listed(blamed)
            ),
            Self::NotFound { unexpressed } => {
                write!(
                    f,
                    "no feasible point was found: nothing proved the region empty and \
                     sampling found nothing"
                )?;
                if !unexpressed.is_empty() {
                    write!(
                        f,
                        "; nothing could be concluded from {}",
                        listed(unexpressed)
                    )?;
                }
                Ok(())
            }
        }
    }
}

/// Which strategies a pool may use.
///
/// Hidden, and hidden deliberately: which strategy delivers is the engine's
/// decision, made per batch — sampling first, the walker for whatever is left
/// — rather than the caller's. This exists so that tests can pin one strategy
/// and measure it alone, because a pool that mixes them cannot say which one
/// produced a bad distribution.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Rejection sampling over the declared box, never narrowed. Uniform over
    /// the feasible region by construction: the probe that decides the route,
    /// the thing that delivers where the probe succeeds, and the fairness
    /// oracle the tests measure against.
    ///
    /// Where the probe lands nothing, and the solver — if configured — could
    /// not settle it either, it is the brute squad: the same proposals, wider,
    /// on every core, for [`ConstraintSolver::with_proposal_budget`]
    /// candidates, to land the seed the walker needs. What it lands is a
    /// function of the seed and the budget, never of the thread count, and
    /// every thread is joined before it returns.
    BruteSquad,
    /// Hit-and-run: walk the chord of the region through the current point.
    /// Converges to the uniform distribution, but needs a feasible point to
    /// start from and crosses between disconnected pieces only by luck.
    /// In the opening it decides only whether the prune budget is spent on
    /// coverage; a design's candidate pool is always walked.
    HitAndRun,
    /// A local solve for a first point when the probe found none: COBYLA,
    /// derivative-free, from the box centre and a few seeded starts, driving
    /// the worst residual down until a point is judged feasible. Finding one
    /// point of a nonlinear system is an ordinary constrained optimisation,
    /// and a local method does it in seconds at two hundred variables where
    /// the solver spends minutes per query and brute force cannot find a
    /// region a millionth of its box. It cannot prove a region empty; a start
    /// that finds nothing only says the basin it fell into held nothing.
    /// Deterministic: a fixed evaluation count per start, and the same seed
    /// gives the same starts.
    LocalSolve,
    /// Interval contraction and bisection — branch-and-prune. The one
    /// strategy that can *prove* a region empty and name the constraints
    /// that conflict, and the one that finds the *pieces* of a region the
    /// walker must be started in, since a chain cannot cross between them.
    /// The declared box is contracted before a single proposal, which is
    /// where a plain contradiction is caught; the box is split and pruned
    /// only when the walker will carry the search or nothing has been found.
    /// Bounded by a count of contractions, never a clock. What it cannot see
    /// — a contradiction every box encloses a little of — is left to brute
    /// force and reported as [`Infeasibility::NotFound`] rather than claimed.
    ///
    /// The one a test leaves out when it must measure sampling alone: a
    /// contraction settles `x1 > 0.999999` at once, which would make a
    /// time-to-first-hit fixture a measurement of the contractor.
    Prune,
}

/// What production uses: plain sampling, the walker for whatever it leaves
/// short, a local solve for a first point where sampling finds none, and
/// branch-and-prune to prove the region empty or find its pieces.
///
/// The strategies are partitioned by role in [`Ladder::new`] rather than by
/// position, so the order here is cosmetic. The actual order of escalation is
/// fixed by [`cvg::open`]: contract, probe, then local solve, then bisect,
/// then brute force.
///
/// Public so that tests measuring "what a caller gets" cannot drift from it. A
/// copy of this list living in the test suite is a copy that goes stale, and did.
#[doc(hidden)]
pub const DEFAULT_STRATEGIES: &[Strategy] = &[
    Strategy::BruteSquad,
    Strategy::LocalSolve,
    Strategy::HitAndRun,
    Strategy::Prune,
];

/// Candidates the brute-force search proposes before giving up, unless
/// [`ConstraintSolver::with_proposal_budget`] says otherwise.
///
/// A billion: a few seconds across a laptop's sixteen threads and a quarter
/// of a minute on one, which reaches a region a hundred-millionth of its box
/// with ten expected hits and gives up on a ten-billionth in a time a caller
/// can wait out. Spent only on what the solver could not decide. A count
/// rather than a duration so that the same seed finds the same point on
/// every machine.
pub const DEFAULT_PROPOSAL_BUDGET: u64 = 1_000_000_000;

/// How many contractions branch-and-prune may spend splitting a box, unless
/// [`ConstraintSolver::with_prune_budget`] says otherwise.
///
/// A count of contractions, so that the same problem answers the same way on
/// every machine. Spent only when the walker will carry the search or
/// nothing has been found; a problem sampling settles pays one contraction
/// of the declared box and nothing more.
///
/// Splitting is a low-dimensional tool: it isolates a piece of a region, or
/// proves a box empty, only where it can split enough coordinates, and 256
/// contractions is eight levels — every coordinate once at eight
/// dimensions. Measured: the two roots of `(x + 2)(x - 1) == 0` and the two
/// branches of `abs(x) == 1` each cost 2; the disc inside a ring it cannot
/// meet is proved empty in 2; a circle's ribbon at `1e-6` spends whatever it
/// is given and settles nothing, which is the shape of a budget spent
/// honestly. On the ten-segment stepped beam — twenty variables, one piece,
/// carried by the walker — every contraction is bought for nothing, and at
/// 4096 that was forty seconds of an unoptimised build; at 256 it is a
/// couple, which is what a problem this tool cannot help pays.
pub const DEFAULT_PRUNE_BUDGET: u32 = 256;

/// Candidates brute force proposes on a GPU before giving up, unless
/// [`ConstraintSolver::with_gpu_proposal_budget`] says otherwise.
///
/// Thirty billion: thirty times the CPU's, because a proposal on the device
/// is ten to a hundred times cheaper. Sized so that a region a ten-billionth
/// of its box is found three times over in expectation rather than being a
/// coin: about fifteen seconds on this laptop's iGPU, which draws and judges
/// two billion candidates a second, and a second or two on a desktop card.
/// Still a count, so that the same seed finds the same point on the same
/// device.
pub const DEFAULT_GPU_PROPOSAL_BUDGET: u64 = 30_000_000_000;

/// The environment variable that picks which GPU the sieve runs on.
///
/// Unset, wgpu's own high-performance preference decides, which on a machine
/// with an iGPU and a discrete card is the card. Set, it is read once per
/// connection: `off` (or `none`) keeps brute force on the CPU; a number is an
/// index into the adapters wgpu enumerates; anything else is a
/// case-insensitive substring of an adapter's name, or the name of a backend
/// (`vulkan`, `dx12`, `metal`). A value that matches nothing is logged at
/// `warn` with the list of what there is, and brute force stays on the CPU —
/// a typo should be noticed, not silently corrected. The diagnostic knob for
/// "which device did it actually use"; the list is logged at `info` whenever
/// the variable is set. Only read by builds with the `gpu` feature.
pub const GPU_VARIABLE: &str = "SOJOURN_GPU";

/// Everything a solve needs beyond the problem itself.
///
/// The randomness, any points the caller already believes in, and which
/// strategies to use — all dependencies, all with defaults. They live here
/// rather than as parameters because there used to be three entry points
/// (`solve`, `solve_with_rng`, `solve_with`) that differed only in how many of
/// these they let you reach, and two of the three existed purely so the tests
/// could get past the first.
///
/// Construction cannot fail: nothing held here can be invalid on its own. What
/// *can* be invalid — a constraint naming a variable the box does not declare —
/// needs the problem, and so is checked in [`ConstraintSolver::solve`].
///
/// ```no_run
/// # use sojourn::{ConstraintSystem, InputVariable};
/// # fn example() -> anyhow::Result<()> {
/// let system = ConstraintSystem::new(vec![InputVariable::new("x", -1.0, 1.0)], ["x > 0"])?;
///
/// let region = sojourn::solve(&system)?;
/// // One column per point, one row per variable — an input matrix as it
/// // stands, no transpose.
/// let design = region.sample(faer::Mat::zeros(1, 0).as_ref(), 16, 42)?;
/// # let _ = design;
/// # Ok(())
/// # }
/// ```
/// Deliberately not `Clone`, even though its generator is: two solvers sharing
/// a stream would silently produce the same "random" points, and a `Clone`
/// here would make that a one-word mistake.
#[derive(Debug)]
pub struct ConstraintSolver {
    rng: Xoshiro256PlusPlus,
    known_feasible: Vec<Point>,
    strategies: Vec<Strategy>,
    budgets: Budgets,
}

/// How much each rung of the ladder may spend before handing over.
///
/// Every one a count rather than a clock, so that the same seed reaches the
/// same verdict on every machine; the thread count changes only how soon.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Budgets {
    /// See [`ConstraintSolver::with_proposal_budget`].
    pub(crate) proposals: u64,
    /// See [`ConstraintSolver::with_threads`].
    pub(crate) threads: usize,
    /// See [`ConstraintSolver::with_prune_budget`].
    pub(crate) prune: u32,
    /// See [`ConstraintSolver::with_gpu`]. Read only when the `gpu` feature
    /// is on; kept in the struct either way so the builder is one API.
    #[cfg_attr(
        not(feature = "gpu"),
        allow(dead_code, reason = "the knob exists without the feature")
    )]
    pub(crate) gpu: bool,
    /// See [`ConstraintSolver::with_gpu_proposal_budget`].
    #[cfg_attr(
        not(feature = "gpu"),
        allow(dead_code, reason = "the knob exists without the feature")
    )]
    pub(crate) gpu_proposals: u64,
}

impl Default for Budgets {
    fn default() -> Self {
        Self {
            proposals: DEFAULT_PROPOSAL_BUDGET,
            threads: std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
            prune: DEFAULT_PRUNE_BUDGET,
            gpu: true,
            gpu_proposals: DEFAULT_GPU_PROPOSAL_BUDGET,
        }
    }
}

impl Default for ConstraintSolver {
    fn default() -> Self {
        Self {
            rng: Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
            known_feasible: Vec::new(),
            strategies: DEFAULT_STRATEGIES.to_vec(),
            budgets: Budgets::default(),
        }
    }
}

impl ConstraintSolver {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Points the caller already believes are feasible.
    ///
    /// A hint, not an assertion: infeasible ones are discarded rather than
    /// trusted. Worth supplying — on a region too tight to sample, a seed is
    /// the difference between an opening that returns at once and one that
    /// spends its budgets.
    #[must_use]
    pub fn with_known_feasible(mut self, points: Vec<Point>) -> Self {
        self.known_feasible = points;
        self
    }

    /// Pins the randomness, so a run is reproducible.
    ///
    /// What a caller reproducing a run supplies. Every point the opening
    /// finds is a function of this seed and the budgets, so two solves of
    /// the same system under the same seed hold the same points. (A design
    /// takes its own seed, [`sample`](FeasibleRegion::sample);
    /// [`repair`](FeasibleRegion::repair) needs none.)
    #[must_use]
    pub fn with_seed(self, seed: u64) -> Self {
        self.with_rng(Xoshiro256PlusPlus::seed_from_u64(seed))
    }

    /// Pins the generator itself, for a test that wants a particular stream.
    #[doc(hidden)]
    #[must_use]
    pub fn with_rng(mut self, rng: Xoshiro256PlusPlus) -> Self {
        self.rng = rng;
        self
    }

    /// Pins the strategy list.
    ///
    /// Hidden along with [`Strategy`] itself: which strategy finds the
    /// region is the engine's decision, not the caller's. Tests use this to
    /// measure one strategy at a time.
    #[doc(hidden)]
    #[must_use]
    pub fn with_strategies(mut self, strategies: Vec<Strategy>) -> Self {
        self.strategies = strategies;
        self
    }

    /// How many candidates brute force may propose before giving up.
    ///
    /// A *proposal* is one random point in the declared box, judged against
    /// every constraint. When the opening probe lands nothing and the solver,
    /// if configured, comes back without a proof or a usable witness, brute
    /// force keeps proposing on every core until a batch lands or this many
    /// have been judged. The default
    /// is [`DEFAULT_PROPOSAL_BUDGET`]; the cost is some seventy million
    /// proposals a second per core on a simple constraint set. Zero skips
    /// brute force on the CPU. A count rather than a duration, so that the
    /// same seed finds the same point on every machine.
    ///
    /// The GPU, when brute force runs there, has its own budget:
    /// [`with_gpu_proposal_budget`](Self::with_gpu_proposal_budget). To skip
    /// brute force altogether, zero both, or zero this and
    /// [`with_gpu(false)`](Self::with_gpu).
    #[must_use]
    pub const fn with_proposal_budget(mut self, proposals: u64) -> Self {
        self.budgets.proposals = proposals;
        self
    }

    /// How many contractions branch-and-prune may spend splitting the box.
    ///
    /// A count, not a clock, so that the same problem gets the same answer on
    /// every machine. The default is [`DEFAULT_PRUNE_BUDGET`]; zero contracts
    /// the declared box once and never splits it, which still catches a plain
    /// contradiction and never finds a second piece. A budget spent without a
    /// conclusion is handled like any other shrug: brute force gets its turn,
    /// and an empty search is [`Infeasibility::NotFound`].
    #[must_use]
    pub const fn with_prune_budget(mut self, contractions: u32) -> Self {
        self.budgets.prune = contractions;
        self
    }

    /// Whether brute force may run on a GPU.
    ///
    /// On by default, and used only when the crate was built with the opt-in
    /// `gpu` feature and an adapter is present; otherwise brute force runs on
    /// the CPU threads and this changes nothing. The device is acquired when
    /// brute force starts and released when it returns. The GPU
    /// proposes and sieves candidates in `f32`, and the CPU re-judges every
    /// survivor exactly, so what it delivers is as feasible as anything else.
    /// What it trades is reproducibility *across machines*: the seed brute
    /// force lands is a function of the seed, the budget, and the device,
    /// where the CPU path is a function of the first two alone. Turn it off
    /// for a run that must reproduce anywhere. Which adapter is used, when
    /// there are several, is [`GPU_VARIABLE`]'s business.
    #[must_use]
    pub const fn with_gpu(mut self, enabled: bool) -> Self {
        self.budgets.gpu = enabled;
        self
    }

    /// How many candidates brute force may propose on a GPU before giving up.
    ///
    /// The GPU's own budget, separate from [`with_proposal_budget`](Self::with_proposal_budget)
    /// because a proposal there costs a tenth to a hundredth of one on the
    /// CPU, so the same wall time buys a wider search. The default is
    /// [`DEFAULT_GPU_PROPOSAL_BUDGET`]. Used only when the sieve is; zero
    /// makes the GPU path give up at once.
    #[must_use]
    pub const fn with_gpu_proposal_budget(mut self, proposals: u64) -> Self {
        self.budgets.gpu_proposals = proposals;
        self
    }

    /// Pins how many threads brute force fans out over.
    ///
    /// Hidden because it never changes what is found — a test uses it to
    /// prove exactly that. Defaults to the available parallelism.
    #[doc(hidden)]
    #[must_use]
    pub const fn with_threads(mut self, threads: usize) -> Self {
        self.budgets.threads = threads;
        self
    }

    /// Finds a feasible region and hands back the points it found there.
    ///
    /// Bounded by its budgets and by nothing else: a count of contractions
    /// for branch-and-prune, of evaluations for the local solve, of
    /// proposals for brute force — never a clock, so the same seed reaches
    /// the same verdict on every machine. That is also the shape of "this
    /// may take a while": the budgets say how much may be spent, and a
    /// search that spends them without a point is [`Infeasibility::NotFound`]
    /// rather than a call that never returns. Milliseconds on most systems,
    /// plus the walker's burn-in — eight chains, `16·d` steps each, twice —
    /// which is the second a solve costs at two hundred variables and what
    /// every design afterwards is spared; brute force on a region it cannot
    /// reach is the default budget's seconds. Runs on the calling thread;
    /// brute force fans out over the cores and joins them before returning,
    /// and nothing runs after.
    ///
    /// # Errors
    /// There is no region: the constraints were proved to conflict, or nothing
    /// could be found and nothing could be proved — [`Infeasibility`]
    /// says which and names the constraints involved. Nothing else is an
    /// error; see [`Infeasibility`] for what is a panic instead.
    pub fn solve(self, system: &ConstraintSystem) -> Result<FeasibleRegion, Infeasibility> {
        let mut ladder = Ladder::new(system, self.rng, &self.strategies, self.budgets);
        let (verdict, progress) = cvg::open(system, &mut ladder, self.known_feasible);

        let name_all = |indices: Vec<usize>| -> Vec<ConstraintRef> {
            indices.into_iter().map(|i| system.named(i)).collect()
        };
        match verdict {
            // The region answers for the system after the search, which is
            // where repair lives; a system is tapes and two small graphs, so
            // the clone is nothing against the search.
            Opening::Satisfied { seeded } => {
                let points = progress.into_points();
                // The reference a flat-constraint repair walks in from: the
                // local solve's point from the box centre — the opening's own
                // where it ran one, else one run here under a fixed seed. A
                // probe hit would do for feasibility but not for this: a
                // random point of Keane's region can stand where the product
                // is already nearly flat, and a chord from there lands the
                // projection on ground it cannot read (1.3 s against 4 ms
                // from the centre, measured). The witness is the fallback
                // where no local solve lands.
                let reference = seeded
                    .or_else(|| {
                        local::find_initial(
                            system,
                            &system.declared(),
                            local::STARTS,
                            &mut Xoshiro256PlusPlus::seed_from_u64(REFERENCE_SEED),
                        )
                    })
                    .unwrap_or_else(|| points[0].clone());
                // The walker's chains, burnt in once here rather than on
                // every design: a function of the region, and the one cost
                // of a solve that is not milliseconds at two hundred
                // variables.
                let mut walker = ladder.into_walker();
                walker.burn_in(&points, system);
                Ok(FeasibleRegion {
                    system: system.clone(),
                    points,
                    reference,
                    walker,
                })
            }
            Opening::Impossible { blamed } => Err(Infeasibility::Proved {
                blamed: name_all(blamed),
            }),
            Opening::Unproven { unexpressed } => Err(Infeasibility::NotFound {
                unexpressed: name_all(unexpressed),
            }),
        }
    }
}

/// Why [`FeasibleRegion::sample`] could not answer.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SampleError {
    /// The region ran out of distinct points before the design was full: the
    /// next choice would have coincided with one already in the design or
    /// among the caller's own. A region a few ulps wide, or a single point —
    /// a fully determined system — does this at once. `found` is the design
    /// as far as it got, every column feasible, in case it is better than
    /// nothing.
    #[error(
        "the region yielded {} distinct points of the {wanted} asked for; it is a point, or near enough to one",
        found.ncols()
    )]
    Degenerate { found: Mat<f64>, wanted: usize },
}

/// A solved region: the system a search found feasible points in, and those
/// points.
///
/// A value, immutable: the engine's ladder and its progress lived for the
/// opening and are gone. What is here is the system, every feasible point the
/// opening ended with (the witness first), the reference a repair walks in
/// from, and the walker with its chains already burnt in. From it a
/// space-filling design is [`sample`](Self::sample)d, and a point that
/// is not one is brought to the region by [`repair`](Self::repair): a region
/// that could not be solved has nothing to repair toward, which is why that
/// lives here and not on the system.
///
/// Slight misnomer: this region is "solved", and may be disjoint
/// (meaning its "feasible regions"),
/// at time of writing it has no mechanism to discover this.
#[derive(Clone)]
pub struct FeasibleRegion {
    system: ConstraintSystem,
    /// Every feasible point the opening ended with, the witness at the front.
    /// Never empty: `Satisfied` means at least one.
    points: Vec<Point>,
    /// A feasible point as far inside the region as a local solve from the
    /// box centre reaches: the reference a repair on a flat constraint walks
    /// in from. A function of the system alone.
    reference: Point,
    /// The walker, its eight chains burnt in from `points` and its shape
    /// fitted, at `solve`. A design clones it and walks.
    walker: HitAndRunWalker,
}

impl std::fmt::Debug for FeasibleRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeasibleRegion")
            .field("points", &self.points.len())
            .finish()
    }
}

impl FeasibleRegion {
    /// The system this region was solved over.
    #[must_use]
    pub const fn system(&self) -> &ConstraintSystem {
        &self.system
    }

    /// The first feasible point the search found: the one every design
    /// starts its walk from.
    #[must_use]
    pub fn witness(&self) -> &[f64] {
        &self.points[0]
    }

    /// Every feasible point the opening ended with, the witness first: the
    /// probe's hits, the local solve's or brute force's seed, the coverage
    /// seeds a bisection found, and any hint the caller supplied that judged
    /// feasible. At most a window of the most recent thousand or so.
    #[must_use]
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// A point that satisfies the system with room to spare, near `point`,
    /// the same every time.
    ///
    /// The answer is a function of this region, the point and the clearance
    /// and of nothing else — not of the designs this region has handed out,
    /// not of anything the caller has seen elsewhere. That is what an
    /// optimizer being repaired needs: a landing that depends on other points
    /// steers the optimizer toward them, and this used to take *anchors* for
    /// exactly that reason and with exactly that effect. "Near" is Euclidean
    /// distance over box-normalised coordinates: each coordinate is clamped
    /// into the interval its constraints leave it, and from there the point
    /// is projected — the feasible point nearest it, by Newton on the KKT
    /// system with the constraints' own gradients, or by a derivative-free
    /// solve where a constraint that bites has no derivative (`floor`,
    /// `ceil`, `sgn`, `%`, a computed subscript) — so a step over a wall is
    /// put back where it stepped from rather than slid along the wall to
    /// wherever one coordinate could reach. A constraint with a jump in it
    /// is sampled around under a fixed seed, and one flat where the point
    /// stands is walked in from a reference point a local solve found from
    /// the box centre — a function of the system alone — so a region that
    /// can be sampled is landed near. Microseconds at fifty
    /// variables in a release build where the gradients apply; the
    /// derivative-free fallbacks are milliseconds to tenths of a second.
    ///
    /// `clearance` is the room kept from every wall, as a fraction of each
    /// variable's box width: the result and each of its `2d` axis neighbours
    /// `clearance * width` away pass [`ConstraintSystem::is_feasible`]. `0.0`
    /// asks for feasibility alone and lands on the bounds. A caller that
    /// normalises points and back wants a few thousand ulps of the unit cube,
    /// `1e-12`: far above what any per-coordinate round trip loses and
    /// invisible to an optimiser. There is no default because the right value
    /// is the caller's own noise floor.
    ///
    /// A point that already has the clearance comes back unchanged, so
    /// `repair(repair(x)) == repair(x)`; a feasible point without it is moved
    /// inward. Otherwise the answer is a judged point with the clearance, no
    /// farther from `point` than clamping reached, and the nearest the
    /// projection found within its evaluation budget. The algorithm is
    /// `src/repair.rs`.
    ///
    /// # Errors
    /// [`RepairError::Stranded`] when nothing feasible was reached at all, and
    /// [`RepairError::Cramped`] when something feasible was but the clearance
    /// could not be had there.
    ///
    /// # Panics
    /// If `point` does not have one entry per variable, or `clearance` is
    /// negative or not finite. That is a caller mixing up systems, not a
    /// verdict about the point.
    pub fn repair(&self, point: &[f64], clearance: f64) -> Result<Point, RepairError> {
        crate::repair::repair(&self.system, &self.reference, point, clearance)
    }

    /// A space-filling design of `count` feasible points, spread away from
    /// `existing` and from each other.
    ///
    /// **One column per point, one row per variable**, in `existing` and in
    /// the result — the shape
    /// [`CompiledExpression::eval`](crate::CompiledExpression::eval) takes, so
    /// a design goes straight back in with no transpose. `existing` is what
    /// the caller already has in its design: not judged, not returned, only
    /// spread away from. An optimizer opening on the box centre repairs it,
    /// hands it in here, and concatenates:
    ///
    /// ```no_run
    /// # use sojourn::{ConstraintSystem, InputVariable};
    /// # fn example() -> anyhow::Result<()> {
    /// # let system = ConstraintSystem::new(vec![InputVariable::new("x", -1.0, 1.0)], ["x > 0"])?;
    /// let region = sojourn::solve(&system)?;
    /// let centre = region.repair(&[0.0], 1e-12)?;
    /// let existing = faer::Mat::from_fn(1, 1, |row, _| centre[row]);
    /// let design = region.sample(existing.as_ref(), 9, 42)?;
    /// // `centre` and the nine columns of `design` are the opening ten.
    /// # let _ = design;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Farthest-first over a pool of candidates: each point chosen is the
    /// candidate whose nearest neighbour among `existing` and the points
    /// chosen so far is farthest, in Euclidean distance over box-normalised
    /// coordinates — the metric [`repair`](Self::repair) lands by. The pool
    /// is the region's own points, a round of uniform proposals, and
    /// hit-and-run from the region's chains, burnt in at `solve`, under
    /// streams derived from `seed`; so the design is a function of this
    /// region, `existing`, `count` and `seed`, and the same call gives the
    /// same matrix. Not a Latin hypercube: a design of fewer points than
    /// variables — the usual case — has no useful stratification, and "far
    /// from what I have" is the whole requirement. Cost is
    /// `O(count² · dimensions)` in the selection plus the walk: milliseconds
    /// at twenty variables, a few seconds at two hundred for a pool of a
    /// hundred and fifty.
    ///
    /// # Errors
    /// [`SampleError::Degenerate`] when the region has fewer than `count`
    /// distinct points to give: it is a single point, or near enough to one
    /// that the walk cannot leave it. The error carries what was found.
    ///
    /// # Panics
    /// If `existing` has columns but not one row per variable. That is a
    /// caller mixing up systems, not a verdict about the points. A matrix
    /// with no columns — `Mat::zeros(0, 0)` will do — is nothing to spread
    /// from.
    pub fn sample(
        &self,
        existing: MatRef<'_, f64>,
        count: usize,
        seed: u64,
    ) -> Result<Mat<f64>, SampleError> {
        let rows = self.system.variables.len();
        assert!(
            existing.ncols() == 0 || existing.nrows() == rows,
            "a design has one row per variable of the system it is sampled from, not {}",
            existing.nrows()
        );
        let anchors: Vec<Point> = (0..existing.ncols())
            .map(|column| (0..rows).map(|row| existing[(row, column)]).collect())
            .collect();
        let chosen = cvg::design(
            &self.system,
            &self.walker,
            &self.points,
            &anchors,
            count,
            seed,
        );
        let found = Mat::from_fn(rows, chosen.len(), |row, column| chosen[column][row]);
        if chosen.len() < count {
            return Err(SampleError::Degenerate {
                found,
                wanted: count,
            });
        }
        Ok(found)
    }
}
