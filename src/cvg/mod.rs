//! Constrained random vector generation.
//!
//! Given a box of input variables and a set of babel constraints, produce
//! points that satisfy all of them and that cover the feasible region
//! reasonably evenly.
//!
//! Lives inside babel rather than alongside it so that [`crate::ast`] can stay
//! private — the SMT-LIB2 emitter is an internal function over the AST, not a
//! published consumer of it.
//!
//! # Strategy
//!
//! Finding the *first* feasible point is the hard part, and for a tight region
//! it needs a solver. Once there, cheap strategies cover the space quickly. That
//! is why `ConstraintSolver::solve` is the expensive, awaitable call and
//! `FeasibleRegion::take` is not.
//!
//! This module is the engine. The types a caller holds — the system, the
//! solver, the samples handle, the verdicts — are defined at the crate root
//! (`system.rs`, `solve.rs`, `repair.rs`) and this is what they drive.
//!
//! The strategies divide along that line:
//!
//! * **Uniform rejection sampling** — the brute squad — *probes*, and on a
//!   region it reaches often enough it simply delivers: unbiased by
//!   construction, no burn-in, no chain. Where the probe lands nothing and the
//!   solver could not settle it, the same sampler keeps proposing on every
//!   core, for a proposal budget, until one batch lands: a region a millionth
//!   or a hundred-millionth of its box is a matter of milliseconds to seconds,
//!   and the seed it finds is what the walker starts from.
//! * **Hit-and-run** *emits* everywhere the probe did not settle it. It
//!   converges to the uniform distribution over the region, so what a caller
//!   receives is governed by the strategy with a guarantee. It cannot start
//!   without a feasible point, and a seed comes from the probe's own hits,
//!   from the local solve, from the solver, or from brute force.
//! * **The local solve** ([`local`]) *seeds* where the probe found nothing:
//!   COBYLA from the box centre and a few seeded starts, stopped at the first
//!   point the oracle judges feasible. Finding one point of a nonlinear system
//!   is an ordinary constrained optimisation, and a local method does it in
//!   milliseconds at two hundred variables where a decision procedure spends
//!   minutes per query and brute force cannot find a region a millionth of its
//!   box. It cannot prove anything: a start that finds nothing says only that
//!   its basin held nothing.
//!
//! None of those can prove a region *empty*. That is the solver's job, and it
//! comes after the local solve and *before* brute force: when the probe and
//! every local start come back empty *and* [`Strategy::Solver`] is in the
//! list, Z3 is asked. It settles a contradiction in milliseconds where brute
//! force would spend its whole budget, and only it can return
//! [`Infeasibility::Proved`]. What it answers `unknown` on — anything
//! transcendental, anything past its limit — is exactly what brute force then
//! spends the budget on. Without a solver in the list the probe hands straight
//! to brute force, and an empty search is simply [`Infeasibility::NotFound`].

pub(crate) mod classify;
#[cfg(feature = "gpu")]
#[doc(hidden)]
pub mod gpu;
pub(crate) mod incidence;
pub(crate) mod interval;
mod local;
mod progress;
pub(crate) mod sampling;
#[cfg(feature = "gpu")]
mod sieve;
mod smt;
mod smtlib;
mod walking;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;

use futures_channel::oneshot;
use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;

use progress::{Progress, Trial};
use sampling::RandomSampler;
use walking::HitAndRunWalker;

use crate::solve::{Budgets, SmtLogic, Strategy};
use crate::{ConstraintSystem, Point};

/// The hit rate below which the walker is expected to do the delivering.
///
/// A rate, judged on the probe. Below it the delivery batches — a hundred
/// candidates per point asked for — come back short often enough that the
/// walker fills most of every batch: at one in a thousand a batch for 32
/// points expects 3.2 hits; at one in ten thousand it is empty three times in
/// four. The JVM's `EASY_PATH_THRESHOLD_FACTOR` was a tenth of the points
/// *asked for* at hundredfold oversampling, which is the same rate.
///
/// This decides nothing about who runs — every batch is sampled first and
/// walked for the rest, see [`next_batch`]. It decides only whether the
/// opening spends solver calls on *coverage*: chains cannot cross between a
/// region's pieces, so a search the walker will carry needs a seed in every
/// piece, where uniform proposals reach every piece in proportion to its
/// measure and need no help. The probe is one batch and can misjudge the
/// rate either way; the cost of a wrong guess here is coverage of a rare
/// piece, never a stalled stream.
const EASY_PATH_THRESHOLD: f64 = 0.001;

/// How many points the worker produces per round trip through the channel.
///
/// Trades channel overhead against shutdown latency and memory: the stop flag
/// is only checked between batches, so this also bounds how long `drop` waits.
///
/// Small, because read-ahead is not free on the expensive problems. Total
/// look-ahead is this times [`CHANNEL_CAPACITY`], and every point of it is
/// produced whether or not anybody asks: at 200 dimensions a point costs some
/// four hundred walker moves, so 64 points of buffer is about three seconds of
/// work done on spec. Cheap problems never notice either number.
const BATCH_SIZE: usize = 32;

/// How many batches may sit unread before the worker blocks.
///
/// This *is* the high-water mark. A bounded channel parks the producer when it
/// is full and wakes it when the consumer drains — which is the whole of
/// "fill up in the background between requests", with no watermarks, condvars
/// or polling to write.
pub(crate) const CHANNEL_CAPACITY: usize = 2;

/// Consecutive empty batches before the worker concludes there is nothing left.
///
/// More than one because an empty batch is not proof: rejection sampling can
/// miss a whole round by luck on a region it usually reaches. More than a
/// handful would just burn cycles on a region that really is exhausted.
const BARREN_BATCHES: usize = 3;

/// The strategies, holding nothing but their streams and their knobs.
///
/// Lives entirely on the worker thread and is never shared. That is the whole
/// concurrency design — no locks, because there is nothing to lock. What the
/// caller holds is [`FeasibleRegion`], which is a handle to the worker and
/// owns none of this. What the search has *found* is not here either: that is
/// a [`Progress`] value the worker threads through its loop.
pub(crate) struct Ladder {
    /// Uniform rejection sampling over the declared box, if configured. The
    /// probe, the first source every batch draws on, and the brute squad where
    /// the opening finds nothing.
    sampler: Option<RandomSampler>,
    /// Fills whatever sampling left short of a batch, from the points in hand.
    walker: Option<HitAndRunWalker>,
    /// The stream a local solve draws its starts from, when
    /// [`Strategy::LocalSolve`] is configured. Runs once, on the opening,
    /// after the probe and before any solver.
    local: Option<Xoshiro256PlusPlus>,
    /// The solver's resource limit, when [`Strategy::Solver`] is configured.
    /// Not a strategy object: the solver needs the whole system to emit a
    /// document, and it runs once, on the opening, rather than per batch.
    solver: Option<u32>,
    /// The SMT-LIB logic a document is emitted under. Carried here rather
    /// than defaulted at the point of use, so that a document is emitted under
    /// the logic the caller chose and not under whatever the worker thread's
    /// environment happens to say.
    logic: SmtLogic,
}

impl std::fmt::Debug for Ladder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ladder")
            .field("sampler", &self.sampler.is_some())
            .field("walker", &self.walker.is_some())
            .field("local", &self.local.is_some())
            .field("solver", &self.solver)
            .field("logic", &self.logic)
            .finish()
    }
}

impl Ladder {
    pub(crate) fn new(
        system: &ConstraintSystem,
        logic: SmtLogic,
        mut rng: Xoshiro256PlusPlus,
        strategies: &[Strategy],
        budgets: Budgets,
    ) -> Self {
        let mut ladder = Self {
            sampler: None,
            walker: None,
            local: None,
            solver: None,
            logic,
        };
        // Each strategy gets its own stream, derived from the one passed in and
        // drawn in list order, so that adding or removing a strategy does not
        // reseed the ones after it.
        for strategy in strategies {
            let stream = Xoshiro256PlusPlus::from_rng(&mut rng);
            match strategy {
                Strategy::Solver => ladder.solver = Some(budgets.solver_limit),
                Strategy::BruteSquad => {
                    let sampler = RandomSampler::new(
                        &system.variables,
                        stream,
                        budgets.proposals,
                        budgets.threads,
                    );
                    #[cfg(feature = "gpu")]
                    let sampler = sampler.with_gpu(budgets.gpu.then_some(budgets.gpu_proposals));
                    ladder.sampler = Some(sampler);
                }
                Strategy::HitAndRun => ladder.walker = Some(HitAndRunWalker::new(stream)),
                Strategy::LocalSolve => ladder.local = Some(stream),
            }
        }
        ladder
    }

    /// Whether the walker will do most of the delivering, judged on the
    /// probe's hit rate against [`EASY_PATH_THRESHOLD`].
    ///
    /// What the opening asks before spending solver calls on coverage. With no
    /// walker configured the answer is no whatever the rate: there are no
    /// chains to place, so nothing to cover for.
    fn walker_will_carry(&self, probe: &Trial) -> bool {
        if self.walker.is_none() {
            return false;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "sample counts are far below the f64 integer limit"
        )]
        let enough = probe.points.len() as f64 >= EASY_PATH_THRESHOLD * probe.proposed as f64;
        !enough
    }
}

/// What the worker concluded while looking for its first point.
///
/// Sent exactly once, and the thing `ConstraintSolver::solve` awaits. Carries
/// constraint *indices* rather than expressions so the worker never needs a copy
/// of them.
pub(crate) enum Opening {
    /// At least one sample is in hand. The invariant a returned `FeasibleRegion`
    /// rests on, and the reason Z3's `unknown` need not surface: an `unknown`
    /// that still produced a point arrives here like any other success.
    Satisfied,
    /// A solver proved the region empty.
    Impossible { blamed: Vec<usize> },
    /// Nothing found and nothing proven, carrying whatever could not be
    /// expressed — which is usually why.
    Unproven { unexpressed: Vec<usize> },
}

/// How many solver calls a search may spend looking for pieces of its region
/// that it has not reached.
///
/// Also the *resolution* of the search: [`cover_gaps`] halves its reach on every
/// unsatisfiable answer, so the count sets how fine it gets before giving up —
/// sixteen halvings take a unit box down to about `1.5e-5`. That doubles as the
/// floor nothing else has to supply. Below it lies the failure this whole thing
/// exists to avoid, where the solver answers with a point a hair from one it
/// already gave us.
///
/// A count, not the budget. The budget is the solver limit, spent across the
/// whole stage in the solver's own units — see [`cover_gaps`] — and on a
/// problem the solver finds hard it is what ends the stage, long before this.
///
/// At zero this is the behaviour before it existed: one seed, every chain
/// starting from it.
const GAP_QUERIES: usize = 16;

/// Seeds from the parts of the region the search has not reached.
///
/// **Hit-and-run cannot discover a component it was not started in.** A chain
/// samples the piece it began in and nothing else, so on `abs(x1) == 1 +/- 1e-9`
/// a search that starts from one root delivers that root five hundred times and
/// never learns the other exists. Somebody has to go looking *before* the chains
/// are placed, and only a solver can.
///
/// # Coarse to fine, because the gap is the unknown
///
/// Asking the solver again returns the same witness, and asking it to avoid that
/// *point* returns one a fifth of a nanometre away — measured, on the parabola
/// above. The exclusion has to be on the scale of the **gap between
/// components**, which nothing knows in advance.
///
/// So `reach` starts at half the widest declared range — as far as it is
/// meaningful to ask — and halves on every `unsat`. An `unsat` says only "there
/// is nothing this far out", which is a statement about `reach` and not about
/// the region, so the answer is to look closer. A `sat` is a genuinely new piece:
/// it is kept, added to what the next round must avoid, and `reach` stays where
/// it is, since a scale that just worked is the right one to try again.
///
/// The two directions of error are both benign. Too large wastes a call and
/// shrinks. Too small returns something adjacent to a point already held, which
/// is redundant rather than wrong — and the budget bottoms out well above that,
/// which is why no floor constant appears here.
///
/// # Soundness
///
/// The constraints are untouched, so any witness is feasible for the real
/// problem. **Only [`smt::Verdict::Seed`] is read**; an `Impossible` from an
/// exclusion query means "nothing that far from what we hold", never that the
/// problem is unsatisfiable.
///
/// Everything returned has been judged: [`smt::adjusted`] nudges a boundary
/// witness and answers only with a point that passes `is_feasible`, so a bad
/// one costs a solver call and never a wrong point.
///
/// # The stage has one budget, in the solver's units
///
/// `limit` is the budget for the *whole* stage — one seed query's worth of
/// insurance, which is what the caller's solver limit already means for a
/// query — not for each of up to [`GAP_QUERIES`] calls. Each query is given
/// what remains as its own limit, so none can overrun the stage, and what it
/// reports spent ([`smt::Answer::spent`]) comes off the top. The stage ends
/// when the budget does, and at the first `unknown`: that query consumed its
/// allowance, and the next would be the same document with a weaker
/// exclusion, which is not a cheaper question. Either way the coverage risk is
/// logged rather than hidden. A limit of zero is the caller's "no limit" and
/// stays so; the count and the per-call ceiling still bound the stage.
///
/// Why units and not seconds: the 20-segment stepped beam (Artemis's
/// `e06`, `tests/regression_fixture.rs`) spends the full default limit on one
/// gap query at 115 s, and sixteen of those before `solve` returned was the
/// 3.7 CPU-hours the report came with — for a region sampling had in hand in
/// under a second. In units the same budget is the same work on any machine.
fn cover_gaps(
    problem: &ConstraintSystem,
    logic: &SmtLogic,
    found: &VecDeque<Point>,
    limit: u32,
    cancel: &Cancellation<'_>,
) -> Vec<Point> {
    if found.is_empty() {
        return Vec::new();
    }

    // As far out as it is meaningful to ask: any more and the exclusion covers
    // the declared box and every answer is `unsat` by construction.
    let mut reach = problem
        .variables()
        .iter()
        .map(|input| (input.upper_bound - input.lower_bound) / 2.0)
        .fold(0.0f64, f64::max);

    let mut avoid: Vec<Point> = found.iter().cloned().collect();
    let mut seeds = Vec::new();

    let unlimited = limit == 0;
    let mut remaining = limit;
    let mut cut_short = false;
    for _ in 0..GAP_QUERIES {
        // Guards a subnormal reach halving its way to zero, and a NaN from a
        // non-finite box, which no amount of looking closer will fix. A
        // cancelled search would otherwise ask, and interrupt, up to sixteen
        // more times.
        if !reach.is_finite() || reach <= 0.0 || cancel.is_requested() {
            break;
        }
        if !unlimited && remaining == 0 {
            cut_short = true;
            break;
        }
        let answer = smt::seed_away_from(problem, logic, remaining, &avoid, reach, cancel);
        remaining = remaining.saturating_sub(answer.spent);
        match answer.verdict {
            smt::Verdict::Seed { point, .. } => {
                // Avoided whether or not it survives repair: the solver has
                // told us about this piece, and asking again would be told the
                // same thing.
                avoid.push(point.clone());
                if let Some(seed) = smt::adjusted(problem, point) {
                    seeds.push(seed);
                }
            }
            smt::Verdict::Impossible { .. } => {
                // Nothing is out this far; look closer.
                reach /= 2.0;
            }
            smt::Verdict::Inconclusive { .. } => {
                // Undecided within its allowance. A weaker exclusion is not a
                // cheaper question, so the stage ends here.
                cut_short = true;
                break;
            }
        }
    }

    if cut_short {
        tracing::warn!(
            seeds = seeds.len(),
            spent = limit - remaining,
            limit,
            reach,
            "gap coverage cut short; components beyond the seeds in hand may be missed"
        );
    }

    seeds
}

/// How many coordinate sweeps a repair gets before it gives up.
/// The caller's way of saying "never mind".
///
/// Dropping the `ConstraintSolver::solve` future drops the receiving
/// end of the opening channel, and the sending end can see that. Brute force
/// asks between batches, and a solver call asks while it waits on Z3.
pub(crate) struct Cancellation<'a>(Option<&'a oneshot::Sender<Opening>>);

impl Cancellation<'_> {
    /// Requested once the receiving end of `opening` is gone.
    pub(crate) const fn watching(opening: &oneshot::Sender<Opening>) -> Cancellation<'_> {
        Cancellation(Some(opening))
    }

    /// Never requested: for a test driving a solver call with no future
    /// behind it to drop.
    #[cfg(test)]
    pub(crate) const fn never() -> Cancellation<'static> {
        Cancellation(None)
    }

    pub(crate) fn is_requested(&self) -> bool {
        self.0.is_some_and(oneshot::Sender::is_canceled)
    }
}

/// The worker's thread body: open, report the verdict, keep filling.
pub(crate) fn serve(
    problem: &ConstraintSystem,
    mut ladder: Ladder,
    known: Vec<Point>,
    opening: oneshot::Sender<Opening>,
    batches: &SyncSender<Vec<Point>>,
    stop: &AtomicBool,
) {
    // Hints are judged, not trusted, and count as points rather than trials.
    let known = known
        .into_iter()
        .filter(|point| problem.is_feasible(point, 0.0))
        .collect();
    let progress = Progress::empty().extend(known);
    let cancel = Cancellation::watching(&opening);

    let (verdict, progress) = open(problem, &mut ladder, progress, &cancel);
    if cancel.is_requested() {
        // The caller dropped the future mid-search. There is nobody to report
        // to, and brute force stopped for exactly that reason.
        return;
    }
    tracing::debug!(
        points = progress.points().len(),
        proposed = progress.proposed(),
        landed = progress.landed(),
        "opened"
    );

    let deliverable = matches!(verdict, Opening::Satisfied);
    if opening.send(verdict).is_err() || !deliverable {
        // Either the caller gave up before we answered, or there is nothing to
        // deliver. Dropping `batches` on the way out is what tells the pool it
        // is exhausted rather than merely slow.
        return;
    }

    // The points in hand are the first batch: every one is feasible, and they
    // are as good as any that would follow.
    let first: Vec<Point> = progress.points().iter().take(BATCH_SIZE).cloned().collect();
    if batches.send(first).is_err() {
        return;
    }
    keep_filling(problem, &mut ladder, progress, batches, stop);
}

/// The opening: probe, then the solver, then brute force, in that order and
/// with no flags between them. Each rung runs only if the ones before it left
/// nothing in hand.
///
/// The probe is one brute-force batch, tens of microseconds, and settles most
/// problems outright. The solver goes next because it settles a contradiction
/// or an equality ribbon in milliseconds, where brute force would spend its
/// whole budget, and what it answers `unknown` on — anything transcendental,
/// anything past its resource limit — is exactly what brute force is for.
///
/// `Satisfied` means at least one feasible point is in hand, which is what
/// a returned [`FeasibleRegion`](crate::FeasibleRegion) promises.
fn open(
    problem: &ConstraintSystem,
    ladder: &mut Ladder,
    progress: Progress,
    cancel: &Cancellation<'_>,
) -> (Opening, Progress) {
    let (mut progress, walker_will_carry) = match &mut ladder.sampler {
        Some(sampler) => {
            let probe = sampler.probe(problem);
            let carries = ladder.walker_will_carry(&probe);
            (progress.absorb(probe), carries)
        }
        None => (progress, ladder.walker.is_some()),
    };
    // Having points settles the *verdict*, and used to end the opening here.
    // It does not settle **coverage**: hit-and-run cannot discover a component
    // it was not started in, so a search the walker will carry needs to know
    // about the whole region before its chains are placed — however the points
    // it holds were come by. A caller's hint and a brute-force seed are every
    // bit as single-component as a solver's witness, and
    // `parabolic_roots_ribbon` hands in one point at `x = -2` and never learns
    // about the root at 1.
    //
    // A search sampling will carry is exempt, and that is not an oversight:
    // uniform proposals reach every component in proportion to its measure, so
    // discovery would be a solver call bought for nothing.
    if !progress.is_empty() {
        if walker_will_carry && let Some(limit) = ladder.solver {
            let gaps = cover_gaps(problem, &ladder.logic, progress.points(), limit, cancel);
            progress = progress.extend(gaps);
        }
        return (Opening::Satisfied, progress);
    }

    // A local solve before any solver: finding one point is an optimisation,
    // and a local method does it where a decision procedure spends minutes
    // per query. A seed found here is as single-component as a witness, so it
    // takes the same road to coverage.
    if let Some(rng) = &mut ladder.local
        && let Some(point) = local::seed(problem, rng, cancel)
    {
        progress = progress.extend(vec![point]);
        if let Some(limit) = ladder.solver {
            let gaps = cover_gaps(problem, &ladder.logic, progress.points(), limit, cancel);
            progress = progress.extend(gaps);
        }
        return (Opening::Satisfied, progress);
    }

    let unexpressed = match ladder.solver {
        // Every constraint is then "unexpressed" in the sense `NotFound` uses:
        // none was put to anything that could reason about it.
        None => (0..problem.constraints.len()).collect(),
        Some(limit) => {
            match smt::escalate_for_seed(problem, &ladder.logic, limit, cancel).verdict {
                smt::Verdict::Impossible { blamed } => {
                    return (Opening::Impossible { blamed }, progress);
                }
                smt::Verdict::Inconclusive { unexpressed } => unexpressed,
                smt::Verdict::Seed { point, unexpressed } => {
                    // The witness is exact in real arithmetic and need not be in
                    // `f64`. Repairing beats discarding: the solver call that found
                    // it is the expensive part, and the miss is in the last place.
                    // A seed is not a sample either: it satisfies whatever could
                    // be expressed, and it is judged against *everything*; if it
                    // does not survive that, brute force still gets its turn.
                    progress = progress.extend(smt::adjusted(problem, point).into_iter().collect());

                    // With a point in hand the search knows one piece of its region,
                    // so now ask the solver about the rest of the box. A region in
                    // several pieces gets a seed in more than one of them here or
                    // nowhere: a chain cannot cross between them afterwards.
                    if !progress.is_empty() {
                        let gaps =
                            cover_gaps(problem, &ladder.logic, progress.points(), limit, cancel);
                        progress = progress.extend(gaps);
                    }
                    unexpressed
                }
            }
        }
    };

    if progress.is_empty()
        && let Some(sampler) = &mut ladder.sampler
    {
        let trial = sampler.brute_force(problem, cancel);
        tracing::info!(
            proposed = trial.proposed,
            landed = trial.points.len(),
            "brute force"
        );
        progress = progress.absorb(trial);
    }

    if progress.is_empty() {
        (Opening::Unproven { unexpressed }, progress)
    } else {
        (Opening::Satisfied, progress)
    }
}

/// The steady state: one batch per trip through the channel until the caller
/// stops asking or the region runs dry.
///
/// "Runs dry" is [`BARREN_BATCHES`] empty batches in a row, and a batch is
/// empty only when sampling landed nothing *and* the walker had nothing to
/// fill from — see [`next_batch`]. With a walker configured and points in
/// hand that cannot happen, so this is a conclusion only a sampling-only
/// configuration ever reaches.
fn keep_filling(
    problem: &ConstraintSystem,
    ladder: &mut Ladder,
    mut progress: Progress,
    batches: &SyncSender<Vec<Point>>,
    stop: &AtomicBool,
) {
    let mut barren = 0;
    while !stop.load(Ordering::Relaxed) {
        let (batch, next) = next_batch(problem, ladder, progress, BATCH_SIZE, stop);
        progress = next;
        if batch.is_empty() {
            barren += 1;
            if barren >= BARREN_BATCHES {
                return;
            }
            continue;
        }
        barren = 0;
        if batches.send(batch).is_err() {
            return;
        }
    }
}

/// At most `count` feasible points, and the progress that now includes them.
///
/// Sampling goes first and the walker fills whatever it left short. Sampling
/// first because it is unbiased by construction and needs no burn-in, and on
/// a region it reaches it is the whole answer: the measured case is
/// `parabolic_roots_narrowing`, where the walker alone returned 2000 points
/// worth 87 independent ones, because a chain cannot cross the gap between
/// the two bands and the split between them was decided by where each chain
/// happened to start. Plain sampling reaches that region perfectly well and
/// has no such problem. On a region sampling cannot reach, its batch is a few
/// thousand proposals judged in one pass — cheap next to the walk that
/// follows — and the walker does the delivering.
///
/// There is no mode to get wrong. A region the probe misjudged as easy is
/// simply walked where sampling comes up short, and the mixture of two
/// sources that are each uniform over the region is uniform over it.
///
/// Fewer than asked for is normal — the walker may have nothing to walk from
/// yet. Never more, and never an infeasible one.
fn next_batch(
    problem: &ConstraintSystem,
    ladder: &mut Ladder,
    mut progress: Progress,
    count: usize,
    stop: &AtomicBool,
) -> (Vec<Point>, Progress) {
    let mut points = Vec::new();
    if let Some(sampler) = &mut ladder.sampler {
        let trial = sampler.deliver(problem, count);
        points.clone_from(&trial.points);
        progress = progress.absorb(trial);
    }
    if points.len() < count
        && let Some(walker) = &mut ladder.walker
    {
        let walked = walker.extend(problem, progress.points(), count - points.len(), stop);
        let walked: Vec<Point> = walked
            .into_iter()
            .filter(|point| problem.is_feasible(point, 0.0))
            .collect();
        progress = progress.extend(walked.clone());
        points.extend(walked);
    }
    (points, progress)
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::*;
    use crate::{ConstraintSolver, Infeasibility, InputVariable};

    /// A region one millionth of its box: the probe misses, brute force
    /// lands a seed, the walker delivers from it.
    fn one_in_a_million() -> ConstraintSystem {
        ConstraintSystem::new(vec![InputVariable::new("x1", 0.0, 1.0)], ["x1 > 0.999999"])
            .expect("the fixture binds")
    }

    const SEED: u64 = 0x50_50_1E_5E_ED;

    #[pollster::test]
    async fn brute_force_seeds_the_walker_when_the_probe_is_empty() {
        let verdict = ConstraintSolver::new()
            .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
            .with_strategies(vec![Strategy::BruteSquad, Strategy::HitAndRun])
            .with_threads(2)
            .solve(&one_in_a_million())
            .await;

        let mut samples = verdict.expect("brute force should have found the region");
        let delivered = samples.take(10);
        assert_eq!(delivered.ncols(), 10);
        for column in 0..10 {
            assert!(
                delivered[(0, column)] > 0.999_999,
                "{}",
                delivered[(0, column)]
            );
        }
    }

    /// The order of escalation: probe, local solve, solver, brute force. With
    /// no budget at all a region the seeders can reach is still found, because
    /// they go first; a region the solver answers `unknown` on — a
    /// transcendental — is found anyway, because what it cannot decide is
    /// handed to brute force.
    #[pollster::test]
    async fn the_solver_goes_first_and_brute_force_takes_what_it_cannot_decide() {
        let by_solver = ConstraintSolver::new()
            .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
            .with_proposal_budget(0)
            .solve(&one_in_a_million())
            .await;
        assert!(
            by_solver.is_ok(),
            "Z3 should have seeded the region with no brute force at all: {by_solver:?}"
        );

        // `sin` is increasing on `[0, 1]`, so this is `x1 > 0.99999` written
        // so that no solver can be asked about it: one in a hundred thousand,
        // ten expected hits in the budget below.
        let transcendental = ConstraintSystem::new(
            vec![InputVariable::new("x1", 0.0, 1.0)],
            ["sin(x1) > sin(0.99999)"],
        )
        .expect("the fixture binds");
        let by_brute_force = ConstraintSolver::new()
            .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
            .with_proposal_budget(1_000_000)
            .with_threads(2)
            .solve(&transcendental)
            .await;
        let mut samples =
            by_brute_force.expect("brute force should have taken over from Z3's `unknown`");
        let delivered = samples.take(5);
        assert_eq!(delivered.ncols(), 5);
        for column in 0..5 {
            assert!(
                delivered[(0, column)] > 0.99999,
                "{}",
                delivered[(0, column)]
            );
        }
    }

    /// The solver limit reaches Z3 through the builder: an instance Z3 would
    /// grind on comes back `NotFound` promptly, with no brute force to mask it
    /// — on either engine.
    #[pollster::test]
    async fn the_solver_limit_bounds_the_opening() {
        let hard = ConstraintSystem::new(
            vec![
                InputVariable::new("x", 0.0, 100.0),
                InputVariable::new("y", 0.0, 100.0),
                InputVariable::new("z", 0.0, 100.0),
            ],
            [
                "floor(x) * floor(y) == floor(z) * 7 + 3 +/- 0.000000001",
                "x*y*z == 12345.678 +/- 0.000000001",
                "x^2 + y^2 == z^2 + 1 +/- 0.000000001",
            ],
        )
        .expect("the fixture binds");

        let started = std::time::Instant::now();
        let verdict = ConstraintSolver::new()
            .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
            .with_solver_limit(30_000)
            .with_proposal_budget(0)
            .with_gpu(false)
            .solve(&hard)
            .await;
        let took = started.elapsed();
        assert!(
            matches!(verdict, Err(Infeasibility::NotFound { .. })),
            "{verdict:?}"
        );
        assert!(took < std::time::Duration::from_secs(10), "{took:?}");
    }

    /// Pins that the loop is what changed: with no budget on either engine
    /// the pool behaves as it did before step 4 and gives up after the probe.
    #[pollster::test]
    async fn a_zero_budget_is_the_old_behaviour() {
        let verdict = ConstraintSolver::new()
            .with_rng(Xoshiro256PlusPlus::seed_from_u64(SEED))
            .with_strategies(vec![Strategy::BruteSquad, Strategy::HitAndRun])
            .with_proposal_budget(0)
            .with_gpu(false)
            .solve(&one_in_a_million())
            .await;

        assert!(
            matches!(verdict, Err(Infeasibility::NotFound { .. })),
            "{verdict:?}"
        );
    }

    /// The caller dropped the `solve` future — here, the receiving end of the
    /// opening channel — while brute force was grinding on an empty region
    /// with an effectively unlimited budget. The worker must notice and
    /// return, not spend the budget.
    #[test]
    fn a_dropped_future_stops_the_search() {
        let system = ConstraintSystem::new(vec![InputVariable::new("x1", 0.0, 1.0)], ["x1 > 2"])
            .expect("the fixture binds");
        let ladder = Ladder::new(
            &system,
            SmtLogic::default(),
            Xoshiro256PlusPlus::seed_from_u64(SEED),
            &[Strategy::BruteSquad, Strategy::HitAndRun],
            Budgets {
                proposals: u64::MAX,
                threads: 2,
                ..Budgets::default()
            },
        );

        let (send_opening, opening) = oneshot::channel();
        let (send_batch, _batches) = mpsc::sync_channel(CHANNEL_CAPACITY);
        let stop = AtomicBool::new(false);
        drop(opening);

        let started = std::time::Instant::now();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                serve(
                    &system,
                    ladder,
                    Vec::new(),
                    send_opening,
                    &send_batch,
                    &stop,
                )
            });
        });
        let took = started.elapsed();
        // Generous, because under `--features gpu` this shares one device —
        // and one lock per dispatch — with the sieve tests running beside it,
        // and connecting to the adapter is a couple of hundred milliseconds
        // by itself. The budget it must not spend is hours.
        assert!(
            took < std::time::Duration::from_secs(10),
            "the worker ran {took:?} after its caller was gone"
        );
    }
}
