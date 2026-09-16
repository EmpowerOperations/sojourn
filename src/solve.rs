//! The solve: a builder for how hard to search, and the handle a search
//! hands back.
//!
//! [`ConstraintSolver`] is every knob — the seed, the budgets, the strategy
//! list — with a default for each, and one awaitable call,
//! [`solve`](ConstraintSolver::solve), that starts the engine on a worker
//! thread. [`FeasibleRegion`] is what a satisfied search returns: the solved
//! region — a handle to that worker, from which feasible points are taken as
//! a matrix, the system it was solved over, and repair of any point against
//! it. The verdicts say what a search concluded, and whether that was a proof
//! or a shrug. The engine itself is `cvg`.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread::JoinHandle;

use futures_channel::oneshot;
use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;

use faer::Mat;

use crate::cvg;
use crate::cvg::{CHANNEL_CAPACITY, Ladder, Opening};
use crate::repair::RepairError;
use crate::{ConstraintRef, ConstraintSystem, Point};

/// Why no sample was produced — and whether that is a proof or a shrug.
///
/// The one way [`solve`](crate::solve) fails to return a region: there is
/// none to return, or none could be found. Everything else that can go wrong
/// in a search — a thread that cannot be spawned, a worker that dies — is a
/// bug in this crate or a failure of the host, and is a panic, raised on the
/// calling thread.
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
    /// function of the seed and the budget, never of the thread count.
    BruteSquad,
    /// Hit-and-run: walk the chord of the region through the current point.
    /// Converges to the uniform distribution, but needs a feasible point to
    /// start from and crosses between disconnected pieces only by luck.
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
/// fixed by [`open`]: contract, probe, then local solve, then bisect, then
/// brute force, then the walker from whatever seed those produced.
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
/// # async fn example() -> anyhow::Result<()> {
/// let system = ConstraintSystem::new(vec![InputVariable::new("x", -1.0, 1.0)], ["x > 0"])?;
///
/// let mut region = sojourn::solve(&system).await?;
/// // One column per sample, one row per variable — an input matrix as it
/// // stands, no transpose.
/// let batch = region.take(1_000);
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
    /// trusted. Worth supplying — on a region too tight to sample, a seed is the
    /// difference between the walker working and having nothing to start from.
    #[must_use]
    pub fn with_known_feasible(mut self, points: Vec<Point>) -> Self {
        self.known_feasible = points;
        self
    }

    /// Pins the randomness, so a run is reproducible.
    ///
    /// What a caller reproducing a run supplies. Every point the search
    /// delivers is a function of this seed and the budgets, so two solves of
    /// the same system under the same seed hand out the same points in the
    /// same order. ([`repair`](FeasibleRegion::repair) needs no seed: it is a
    /// function of the system, the point and the clearance.)
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
    /// Hidden along with [`Strategy`] itself: which strategy delivers is the
    /// engine's decision, made per batch, not the caller's. Tests use this to
    /// measure one strategy at a time, because a pool that mixes them cannot
    /// say which produced a bad distribution.
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
    /// if configured, comes back without a proof or a usable witness, the pool
    /// keeps proposing on every core until a batch lands or this many have
    /// been judged. The default
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

    /// Finds a feasible region and hands back something that can sample it.
    ///
    /// # Why this is `async`
    ///
    /// There is no bound on how long it takes. A solver can hit exponential
    /// blowup and effectively not finish, so a plain `fn` returning in 45ms or
    /// 45 minutes would be lying about its cost. A future says so in the type.
    ///
    /// No runtime is imposed. [`Future`](std::future::Future) is in `core`; drive
    /// this with tokio, smol, or a bare `block_on` — this crate's own tests use
    /// the last of those, which is the proof that nothing heavier is required.
    ///
    /// The search runs on its own thread and this future waits on the opening
    /// verdict, so a `timeout` around it does fire. **Dropping the future is
    /// how to cancel:** a brute-force search notices between batches and
    /// stops, freeing every core it took, and a bisection notices between
    /// boxes. [`FeasibleRegion::take`] is synchronous by design. Recorded in
    /// `docs/todo.md`.
    ///
    /// # Errors
    /// There is no region: the constraints were proved to conflict, or nothing
    /// could be found and nothing could be proved — [`Infeasibility`]
    /// says which and names the constraints involved. Nothing else is an
    /// error; see [`Infeasibility`] for what is a panic instead.
    ///
    /// # Panics
    /// If the search thread panics, with its panic: the worker's payload is
    /// resumed here rather than caught, so a bug in the engine is a panic on
    /// the calling thread like any other.
    pub async fn solve(self, system: &ConstraintSystem) -> Result<FeasibleRegion, Infeasibility> {
        // The worker owns a clone and the handle another: the thread needs
        // `'static`, and the handle answers for the region after the search,
        // which is where repair lives. A system is tapes and two small graphs,
        // so two clones are nothing against the search.
        let ladder = Ladder::new(system, self.rng, &self.strategies, self.budgets);
        let system = system.clone();

        let (send_batch, batches) = mpsc::sync_channel(CHANNEL_CAPACITY);
        let (send_opening, opening) = oneshot::channel();
        let stop = Arc::new(AtomicBool::new(false));

        let worker_stop = Arc::clone(&stop);
        let known_feasible = self.known_feasible;
        let worker_system = system.clone();
        let worker = std::thread::spawn(move || {
            cvg::serve(
                &worker_system,
                ladder,
                known_feasible,
                send_opening,
                &send_batch,
                &worker_stop,
            );
        });

        let Ok(verdict) = opening.await else {
            // The worker dropped its end without reporting, which only a
            // panic does: join it and raise that panic here, payload intact.
            match worker.join() {
                Err(payload) => std::panic::resume_unwind(payload),
                Ok(()) => unreachable!("the search thread returned without reporting a verdict"),
            }
        };

        // Built even for an unsatisfiable problem, which does not keep it: its
        // `Drop` is what joins the worker.
        let region = FeasibleRegion {
            system,
            batches,
            buffer: VecDeque::new(),
            worker: Some(worker),
            stop,
            exhausted: false,
        };

        let name_all = |indices: Vec<usize>| -> Vec<ConstraintRef> {
            indices
                .into_iter()
                .map(|i| region.system.named(i))
                .collect()
        };

        match verdict {
            // The region is dropped on both unsatisfiable paths, and its `Drop`
            // is what joins the worker.
            Opening::Satisfied => Ok(region),
            Opening::Impossible { blamed } => Err(Infeasibility::Proved {
                blamed: name_all(blamed),
            }),
            Opening::Unproven { unexpressed } => Err(Infeasibility::NotFound {
                unexpressed: name_all(unexpressed),
            }),
        }
    }
}

/// What a pool is doing, when it is not simply handing over points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// Still producing, or at least still trying.
    Filling,
    /// The worker finished. There will be no more points, ever.
    ///
    /// Only ever a legitimate end: a worker that *panicked* is not exhausted,
    /// its panic is resumed on the caller's thread by the `take` that finds
    /// it, so "no more points exist" and "we broke" can never be confused.
    Exhausted,
}

/// A solved region: the system a search found feasible points in, being
/// sampled on a background thread.
///
/// Holds no search state — the engine's ladder and its progress value live
/// on the worker thread and nowhere else. This is the system, a receiving
/// end, a buffer, and the means to stop the worker. It is also where a point
/// that is not a sample is brought to the region, [`repair`](Self::repair):
/// a region that could not be solved has nothing to repair toward, which is
/// why that lives here and not on the system.
///
/// Slight misnomer: this region is "solved", and may be disjoint
/// (meaning its "feasible regions"),
/// at time of writing it has no mechanism to discover this.
pub struct FeasibleRegion {
    system: ConstraintSystem,
    batches: Receiver<Vec<Point>>,
    buffer: VecDeque<Point>,
    worker: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    /// Set once the channel disconnects. The worker is gone and no amount of
    /// waiting will produce more.
    exhausted: bool,
}

impl std::fmt::Debug for FeasibleRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeasibleRegion")
            .field("buffered", &self.buffer.len())
            .field("status", &self.status())
            .finish()
    }
}

impl FeasibleRegion {
    /// The system this region was solved over.
    #[must_use]
    pub const fn system(&self) -> &ConstraintSystem {
        &self.system
    }

    /// A point that satisfies the system with room to spare, near `point`,
    /// the same every time.
    ///
    /// The answer is a function of the system, the point and the clearance
    /// and of nothing else — not of the samples this region has handed out,
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
    /// wherever one coordinate could reach. A constraint flat where the
    /// point stands, or with a jump in it, is walked in from a reference
    /// point found under a fixed seed and sampled around under another, so a
    /// region that can be sampled is landed near. Microseconds at fifty
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
    pub fn repair(&self, point: &[f64], clearance: f64) -> std::result::Result<Point, RepairError> {
        crate::repair::repair(&self.system, point, clearance)
    }

    /// Up to `count` samples, waiting for them.
    ///
    /// **One column per sample, one row per schema variable** — the shape
    /// [`CompiledExpression::eval`](crate::CompiledExpression::eval) takes, so a
    /// batch goes straight back in with no transpose.
    ///
    /// Fewer than `count` means the search is exhausted and no amount of waiting
    /// will produce more. That is a real outcome, not an error: a region can
    /// yield forty points and then nothing, ever, and blocking forever on the
    /// forty-first is the hang this returns short to avoid.
    ///
    /// Blocking rather than `async`, for now. The producer is a thread and the
    /// channel is `std::sync::mpsc`, so waiting here is a real park rather than
    /// a spin; making this `async` honestly means an async-aware channel, which
    /// is a change to the worker and not to this signature. Use
    /// [`try_take`](Self::try_take) from a context that must not block.
    ///
    /// # Panics
    /// With the worker's panic, if it panicked: a bug in the engine is raised
    /// here, on the caller's thread, rather than reported as an end.
    pub fn take(&mut self, count: usize) -> Mat<f64> {
        while self.buffer.len() < count && !self.exhausted {
            match self.batches.recv() {
                Ok(batch) => self.buffer.extend(batch),
                Err(_) => self.exhausted = self.worker_finished(),
            }
        }
        self.drain(count)
    }

    /// Up to `count` samples from what is already buffered. Never waits.
    ///
    /// Named `try_take` and not `poll`: `poll` is the async primitive, and a
    /// method by that name on a type callers `await` around would read as one.
    ///
    /// # Panics
    /// As [`take`](Self::take).
    pub fn try_take(&mut self, count: usize) -> Mat<f64> {
        while self.buffer.len() < count {
            match self.batches.try_recv() {
                Ok(batch) => self.buffer.extend(batch),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.exhausted = self.worker_finished();
                    break;
                }
            }
        }
        self.drain(count)
    }

    /// Joins the worker once its channel has closed: `true` when it ended, a
    /// resumed panic when it panicked. Idempotent, since the handle is taken.
    fn worker_finished(&mut self) -> bool {
        if let Some(handle) = self.worker.take()
            && let Err(payload) = handle.join()
        {
            std::panic::resume_unwind(payload);
        }
        true
    }

    /// How many samples can be had right now without waiting.
    #[must_use]
    pub fn available(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the search has finished. No further sample will ever arrive.
    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Stop producing.
    ///
    /// [`Drop`] does this too; calling it early is for a caller who has enough
    /// and wants the worker's CPU back before the handle goes out of scope.
    pub fn close(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Takes `count` from the buffer as a column-per-sample matrix.
    fn drain(&mut self, count: usize) -> Mat<f64> {
        let taken = count.min(self.buffer.len());
        let rows = self.system.variables.len();
        // `from_fn` visits in the matrix's own order, so the points come out of
        // the buffer by index rather than by draining as it goes.
        let samples = Mat::from_fn(rows, taken, |row, column| self.buffer[column][row]);
        self.buffer.drain(..taken);
        samples
    }

    #[must_use]
    pub const fn status(&self) -> Status {
        if self.exhausted {
            Status::Exhausted
        } else {
            Status::Filling
        }
    }
}

impl Drop for FeasibleRegion {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);

        // Draining is not tidiness, it is the difference between joining and
        // deadlocking. `drop` runs before the fields do, so the receiver is
        // still alive here — and a worker parked on a full channel stays parked
        // until somebody reads. Emptying it lets that last `send` return, at
        // which point the worker sees the stop flag and exits, the sender drops,
        // and `recv` finally errors out of this loop.
        while self.batches.recv().is_ok() {}

        if let Some(handle) = self.worker.take() {
            drop(handle.join());
        }
    }
}
