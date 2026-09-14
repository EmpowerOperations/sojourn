//! Handing constraints to an SMT solver.
//!
//! The document itself is [`super::smtlib`]'s job; this is what sends it and reads
//! the answer back.
//!
//! # Why a document rather than a solver's API
//!
//! The JVM version built Z3 AST objects directly and paid for it: a hand-rolled
//! Taylor expansion of `sin` out to 1/11!, definitional axioms for `sqrt`,
//! `cbrt`, `log`, `ln`, `floor` and `ceil`, four axioms for `mod`, and a
//! mechanism that silently *dropped* any constraint making the solver answer
//! UNKNOWN. Roughly a third of that file was fighting the API rather than
//! solving constraints — and because it transcoded straight into Z3 objects, the
//! only way to see what it had asked was to ask Z3.
//!
//! Text is readable, diffable, testable with no solver attached, and portable
//! across solvers. It also makes the backend a small thing rather than the whole
//! integration.
//!
//! # The one hazard worth knowing
//!
//! **`Solver::from_string` returns `()`.** It cannot report a syntax error, and
//! a malformed document leaves an empty solver that then answers `sat` instantly
//! with an empty model — so an emitter bug reads as "solved it". Every verdict
//! here is therefore gated on the assertions actually having arrived. A parse
//! error loses the *whole* document rather than the tail after it, which is what
//! makes that single check sufficient; there is a test pinning exactly that, so
//! it will say so if Z3 ever becomes more forgiving.
//!
//! # The other hazard: Z3 does not always stop
//!
//! `rlimit` makes a hard instance give up deterministically, and usually it
//! does. But `(^ x 1.234)` — a rational exponent, which the emitter refuses
//! precisely because of this — ran `check()` for 119 seconds against a
//! 5-second `timeout` and an rlimit of a hundred thousand, and a first attempt
//! for eight minutes before it was killed. Both budgets are one cancel flag
//! that Z3's loops are supposed to poll, and some loop does not. Where one such
//! hole was found there will be others, so every call goes through
//! [`Z3Backend::solve`]'s leash: Z3 runs on its own thread, is interrupted when
//! the caller cancels or a wall-clock ceiling passes, and is abandoned with an
//! error-level log line if it ignores that. The caller then carries on without
//! it, which is the failure mode a hung process never offers.

#![allow(dead_code)] // `DRealBackend` is a recorded option, not a live one.

use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use super::{Cancellation, smtlib};
use crate::solve::SmtLogic;
use crate::{ConstraintSystem, Point};

/// What a solver concluded about a document.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Outcome {
    /// Satisfiable, with a model — variable name to value.
    Sat(Vec<(String, f64)>),
    /// Provably no solution, and which constraints were involved.
    ///
    /// Several indices rather than one culprit, because a contradiction is a
    /// *relationship*: `x > 8` is perfectly satisfiable until `x < 2` turns up.
    /// Naming one of them would be picking arbitrarily.
    Unsat { blamed: Vec<usize> },
    /// The solver gave up: timeout, unsupported theory, incomplete procedure.
    Unknown,
}

/// What a solver had to say, and what it cost to say it.
///
/// `spent` is in the backend's own resource units — the same units `limit`
/// is given in — and is reported whether or not the call finished, so a
/// caller asking several questions against one budget can keep honest
/// accounts. Not wall clock: a unit is a fixed amount of the solver's work,
/// and the same budget answers the same way on a slower machine. A backend
/// that cannot report reports zero; a call that had to be abandoned reports
/// its whole allowance, since nothing better is known and a budget must not
/// be credited for work that never came back.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Reply {
    pub(crate) outcome: Outcome,
    pub(crate) spent: u32,
}

/// Somewhere to send an SMT-LIB2 document.
///
/// A trait because the deployment story and the theory support pull in opposite
/// directions, and we should be able to measure rather than guess. See the
/// survey below.
///
/// Deliberately **not** `Send + Sync`. It carried those bounds while it was a
/// stub, on the assumption the pool would hold a `Box<dyn SmtBackend>` shared
/// between threads — which is not how it turned out: a backend is constructed
/// where it is used, on the worker thread, and no call site is dynamic. The
/// bounds would also rule out a perfectly reasonable implementation, since Z3's
/// context is thread-local and anything caching one could never be `Sync`.
pub(crate) trait SmtBackend {
    fn name(&self) -> &'static str;

    /// Solve a complete SMT-LIB2 document.
    ///
    /// # Errors
    /// Transport and process failures. A solver *concluding* `unknown` is an
    /// [`Outcome`], not an error.
    /// `limit` is a resource limit in the backend's own units; zero is none.
    /// The [`Reply`] says what the call spent in those units. `cancel` is the
    /// caller's way of saying "never mind" mid-call, which a backend answers
    /// with [`Outcome::Unknown`].
    fn solve(&self, document: &str, limit: u32, cancel: &Cancellation<'_>) -> Reply;
}

/// **dReal** — the best theory fit, and the reason this trait exists.
///
/// A δ-complete solver for nonlinear real arithmetic built on interval
/// constraint propagation. It answers `delta-sat` with a witness box, or
/// `unsat`. Crucially **`sin`, `cos`, `exp`, `log` and `sqrt` are primitives**,
/// so every transcendental workaround in the JVM implementation simply
/// disappears — no Taylor series, no range reduction, no definitional axioms.
///
/// "Satisfiable within δ" rather than exactly is the trade, and for *generating
/// sample points* that is arguably what you want: the job is finding points, not
/// proving theorems. Points get filtered against the real constraints anyway.
///
/// The catch, and the reason this is a stub rather than a dependency: **there is
/// no Rust crate.** dReal is subprocess-only — write the document to stdin, read
/// `delta-sat`/`unsat` and the model back. Which also hands you an OS-level
/// answer to a wedged solver, since `kill(pid)` always works where interrupting
/// a native library inside your own process does not.
pub(crate) struct DRealBackend {
    /// Path to the `dreal` executable.
    pub(crate) executable: std::path::PathBuf,
    /// The δ it is allowed to be wrong by. dReal's own default is `0.001`.
    pub(crate) precision: f64,
}

impl SmtBackend for DRealBackend {
    fn name(&self) -> &'static str {
        "dreal"
    }

    fn solve(&self, _document: &str, _limit: u32, _cancel: &Cancellation<'_>) -> Reply {
        unimplemented!(
            "dReal backend: spawn {} with --precision {}, write the document to \
             stdin, parse `delta-sat`/`unsat` and the witness box from stdout",
            self.executable.display(),
            self.precision
        )
    }
}

/// **Z3** — the incumbent, and the only one that can be linked in-process.
///
/// The `z3` crate's `bundled`/`vendored` features build Z3 from source via
/// `z3-src`, so it links statically and there is no executable to deploy. And
/// `Solver::from_string` parses SMT-LIB2 text, so leaving the *API* does not
/// mean leaving *Z3* — the document approach works either way.
///
/// The cost is that Z3's QF_NRA has no transcendentals, so `sin` and friends
/// need encoding again. That is the whole reason to shop around.
///
/// `cvc5` has in-process bindings too (`cvc5` / `cvc5-sys`) and better
/// transcendental support than Z3; worth measuring against both.
pub(crate) struct Z3Backend;

/// How long the caller waits after interrupting Z3 before giving up on it.
///
/// An interrupt that lands is answered in milliseconds; one that does not land
/// is never answered, so there is nothing to tune here beyond "long enough to
/// be sure".
const INTERRUPT_GRACE: Duration = Duration::from_secs(1);

/// Stack for the Z3 thread. Z3 recurses over terms and the default for a
/// spawned thread is two megabytes; this is what a deep document costs.
const Z3_STACK: usize = 8 << 20;

/// The wall clock a call may run before it is interrupted regardless of its
/// rlimit, and abandoned if it ignores that.
///
/// Deliberately far past honest work, so it never fires on a call that is
/// merely slow and the rlimit stays the thing that decides: twenty times the
/// eight seconds per million units [`DEFAULT_SOLVER_LIMIT`] was measured at,
/// never under a minute, eight minutes at the default limit. A limit of zero
/// is "no limit", and no limit must not mean hung, so it gets an hour.
///
/// The eight seconds is not a property of a unit, which is why the margin is
/// twenty and not two: a unit is a fixed amount of Z3's work, and what that
/// work costs in seconds is the problem's. On the nonlinear 20-segment beam
/// (`tests/regression_fixture.rs`) the default limit is spent in 115 s,
/// about forty microseconds a unit against the eight this was measured at.
/// The ceiling is the last resort against a hung process and never the
/// budget; budgets are in units, where [`Reply::spent`] keeps the accounts.
///
/// [`DEFAULT_SOLVER_LIMIT`]: crate::DEFAULT_SOLVER_LIMIT
fn ceiling(limit: u32) -> Duration {
    const SECONDS_PER_MILLION: f64 = 8.0;
    const MARGIN: f64 = 20.0;
    const FLOOR: Duration = Duration::from_secs(60);
    const UNLIMITED: Duration = Duration::from_secs(60 * 60);

    if limit == 0 {
        return UNLIMITED;
    }
    let expected = f64::from(limit) / 1e6 * SECONDS_PER_MILLION;
    Duration::from_secs_f64(expected * MARGIN).max(FLOOR)
}

/// A Z3 context another thread may interrupt.
///
/// `Z3_interrupt` is the one call Z3 documents as safe from a thread other
/// than the context's own. The pointer stays valid because the thread that
/// owns the context holds it until [`Z3Backend::solve`] has returned, which
/// is after the last interrupt it could send.
struct Interruptible(z3_sys::Z3_context);

// SAFETY: the pointer is only ever passed to `Z3_interrupt`, which Z3
// specifies as callable from another thread, and the owning thread keeps the
// context alive for as long as this handle can be used (see `Z3Backend::solve`).
unsafe impl Send for Interruptible {}

impl Interruptible {
    fn interrupt(&self) {
        // SAFETY: as above.
        unsafe { z3_sys::Z3_interrupt(self.0) }
    }
}

/// Waits for the worker's answer, interrupting when the caller cancels or the
/// ceiling passes, and giving up on the worker if the interrupt goes
/// unanswered for the grace period.
///
/// `None` is "abandoned": the worker is still running and nothing more will
/// be heard from it. A worker that died took its answer with it, so its panic
/// is resumed here on the caller's thread rather than reported as a hang.
fn await_answer(
    answers: &mpsc::Receiver<Reply>,
    worker: std::thread::JoinHandle<()>,
    cancel: &Cancellation<'_>,
    interrupt: impl FnOnce(),
    ceiling: Duration,
    grace: Duration,
) -> Option<Reply> {
    const POLL: Duration = Duration::from_millis(10);

    let started = Instant::now();
    let mut interrupt = Some(interrupt);
    let mut interrupted_at = None;
    loop {
        match answers.recv_timeout(POLL) {
            Ok(answer) => return Some(answer),
            Err(RecvTimeoutError::Disconnected) => match worker.join() {
                Err(payload) => std::panic::resume_unwind(payload),
                Ok(()) => unreachable!("the Z3 thread holds its sender until it has answered"),
            },
            Err(RecvTimeoutError::Timeout) => {}
        }
        match interrupted_at {
            None if cancel.is_requested() || started.elapsed() >= ceiling => {
                if let Some(interrupt) = interrupt.take() {
                    interrupt();
                }
                interrupted_at = Some(Instant::now());
            }
            Some(at) if at.elapsed() >= grace => return None,
            _ => {}
        }
    }
}

impl SmtBackend for Z3Backend {
    fn name(&self) -> &'static str {
        "z3"
    }

    /// The leash. Z3 runs on a thread of its own; this thread waits, and
    /// interrupts it when the caller cancels or [`ceiling`] passes. An
    /// interrupt Z3 answers is an [`Outcome::Unknown`] like any other. One it
    /// ignores — see the module doc for the case that does — is logged at
    /// error level and the thread is left to finish on its own, so that the
    /// search carries on rather than the process hanging.
    ///
    /// The release channel is what makes the interrupt sound: the worker keeps
    /// its context alive until this function returns, so an interrupt can never
    /// reach a context that a worker finishing at the same moment has freed.
    fn solve(&self, document: &str, limit: u32, cancel: &Cancellation<'_>) -> Reply {
        let (send_context, context) = mpsc::channel();
        let (send_answer, answers) = mpsc::channel();
        let (release, hold) = mpsc::channel::<()>();
        let text = document.to_owned();
        let worker = std::thread::Builder::new()
            .name("sojourn-z3".to_owned())
            .stack_size(Z3_STACK)
            .spawn(move || {
                // Fresh per thread, so this is the one every object below uses.
                let handle = Interruptible(z3::Context::thread_local().get_z3_context());
                let _ = send_context.send(handle);
                let _ = send_answer.send(solve_on_this_thread(&text, limit));
                // Errors when the caller has returned, which is the point.
                let _ = hold.recv();
            })
            .expect("the host can spawn a thread for Z3");
        let Ok(handle) = context.recv() else {
            match worker.join() {
                Err(payload) => std::panic::resume_unwind(payload),
                Ok(()) => unreachable!("the Z3 thread sends its context before anything else"),
            }
        };

        let started = Instant::now();
        let answer = await_answer(
            &answers,
            worker,
            cancel,
            || handle.interrupt(),
            ceiling(limit),
            INTERRUPT_GRACE,
        );
        drop(release);
        answer.unwrap_or_else(|| {
            tracing::error!(
                limit,
                cancelled = cancel.is_requested(),
                waited = ?started.elapsed(),
                document_bytes = document.len(),
                "Z3 ignored the interrupt; its thread is abandoned and keeps running"
            );
            tracing::debug!(document = %document, "the document Z3 would not stop on");
            Reply {
                outcome: Outcome::Unknown,
                spent: limit,
            }
        })
    }
}

/// One call to Z3, on the calling thread, with the thread-local context.
fn solve_on_this_thread(document: &str, limit: u32) -> Reply {
    let solver = z3::Solver::new();
    // `rlimit` is Z3's own count of the work it has done, in units of its
    // choosing; it makes the call give up with `unknown` deterministically,
    // where a wall-clock timeout would make the same problem answer
    // differently on a slower machine. Zero is Z3's "no limit".
    let mut params = z3::Params::new();
    params.set_u32("rlimit", limit);
    solver.set_params(&params);
    solver.from_string(document);

    // `from_string` returns `()`. It *cannot* report a syntax error, and a
    // malformed document leaves an empty solver which then answers `sat`
    // instantly with an empty model — so an emitter bug would read as
    // "solved it". Checking the assertions actually arrived is the only
    // defence available, and it is worth more than it looks.
    if solver.get_assertions().is_empty() {
        tracing::error!(document = %document, "the document Z3 parsed nothing from");
        panic!(
            "Z3 parsed no assertions from a {}-byte document: `Solver::from_string` reports a \
             syntax error by silently accepting nothing, so this is an emitter bug — the \
             document is in the trace",
            document.len()
        );
    }

    let result = solver.check();
    // Z3's own count of the work it did, in the units `rlimit` is set in —
    // reported after an `unknown` too, which is what lets a caller budget a
    // series of calls rather than each one.
    let spent = match solver.get_statistics().value("rlimit count") {
        Some(z3::StatisticsValue::UInt(count)) => count,
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a count Z3 reports as a double is still a count, and saturating is the honest rounding"
        )]
        Some(z3::StatisticsValue::Double(count)) => count.max(0.0) as u32,
        None => 0,
    };
    let outcome = match result {
        z3::SatResult::Unsat => Outcome::Unsat {
            blamed: solver
                .get_unsat_core()
                .iter()
                .filter_map(|term| smtlib::core_index(&term.to_string()))
                .collect(),
        },
        z3::SatResult::Unknown => Outcome::Unknown,
        z3::SatResult::Sat => {
            let Some(model) = solver.get_model() else {
                tracing::error!(document = %document, "the document Z3 answered sat on without a model");
                panic!("Z3 answered sat but produced no model; the document is in the trace");
            };

            let mut values = Vec::new();
            for declaration in model.iter() {
                // The `define-fun` prelude helpers are in the model too and
                // they take arguments; `apply(&[])` on one panics inside the
                // binding rather than returning an error.
                if declaration.arity() != 0 {
                    continue;
                }
                let Some(term) = model.get_const_interp(&declaration.apply(&[])) else {
                    continue;
                };
                let Some(real) = term.as_real() else {
                    continue;
                };

                // Z3 hands back rationals, and not in lowest terms or even
                // in the form you wrote: ask about `2.5` and the model says
                // `(/ 5.0 2.0)`. So this is the *ordinary* path, not a
                // special case, and dividing the pair is the whole job.
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "narrowing a model value to f64 is the point of this function"
                )]
                let value = match real.as_rational() {
                    Some((numerator, denominator)) if denominator != 0 => {
                        numerator as f64 / denominator as f64
                    }
                    // Two ways to land here, and the decimal string handles
                    // one of them. `as_rational` is `Z3_get_numeral_small`,
                    // which fails when either half overruns `i64`; and a
                    // nonlinear model can hold an algebraic irrational with
                    // no rational form at all — `sqrt 2` comes back as
                    // `(root-obj (+ (^ x 2) (- 2)) 2)`.
                    //
                    // `approx`'s argument is decimal *places*, not
                    // significant figures, which is the trap: at the
                    // `approx_f64` default of 17 a value of 1e-23 reads back
                    // as a confident `0.0`. 330 covers every magnitude an
                    // `f64` can hold — the smallest subnormal is near
                    // 4.9e-324 — and anything smaller than that is `0.0`
                    // honestly rather than by truncation.
                    _ => match real.approx(330).parse::<f64>() {
                        Ok(approximation) => approximation,
                        // Not a numeral at all: `pi` comes back symbolic,
                        // and `approx_f64` would `unwrap` and panic the
                        // worker. Skipping leaves the variable at its lower
                        // bound and lets the pool's filter judge the point.
                        Err(_) => continue,
                    },
                };
                values.push((declaration.name(), value));
            }
            Outcome::Sat(values)
        }
    };
    Reply { outcome, spent }
}

/// What a solver had to say about a pool that sampling could not crack, and
/// what it spent saying it — [`Reply::spent`], in the solver's own units.
pub(crate) struct Answer {
    pub(crate) verdict: Verdict,
    pub(crate) spent: u32,
}

/// What a solver had to say about a pool that sampling could not crack.
pub(crate) enum Verdict {
    /// A point worth walking out from.
    ///
    /// `unexpressed` is normally empty. When it is not, the point satisfies only
    /// the constraints the emitter could write down — still the best start
    /// available, and the pool filters it against *all* of them anyway, but the
    /// caller must not report the region as understood.
    Seed {
        point: Point,
        unexpressed: Vec<usize>,
    },
    /// No point exists, and these constraints are why. Only ever returned when
    /// the solver was shown the whole problem.
    Impossible { blamed: Vec<usize> },
    /// Nothing usable, carrying whatever the emitter could not express — which
    /// is usually the reason.
    Inconclusive { unexpressed: Vec<usize> },
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
/// is [`FeasibleRegion::repair`](crate::FeasibleRegion::repair), which starts
/// from anywhere in the box. This is a bounded coordinate sweep: for each
/// variable, try a step of a few ulps each way and keep it if the worst
/// residual falls. That reaches a point which is *barely* outside, which is
/// the only case a solver witness produces. It will not rescue a point that
/// is genuinely infeasible, and it should not.
///
/// Returns `None` when the point cannot be brought inside, which is then
/// the honest answer rather than a silent near-miss.
pub(crate) fn adjusted(problem: &ConstraintSystem, mut point: Point) -> Option<Point> {
    if problem.is_feasible(&point, 0.0) {
        return Some(point);
    }

    for sweep in 0..ADJUST_SWEEPS {
        let mut improved = false;

        for index in 0..point.len() {
            let before = problem.worst_residual(&point)?;
            let original = point[index];

            // Growing the step across sweeps: an ulp first, because that
            // is what a boundary witness misses by, then wider in case the
            // rounding compounded through a longer expression.
            let step = ulps(original, 1 << (2 * sweep));

            for candidate in [original + step, original - step] {
                point[index] = candidate;
                let better = problem
                    .worst_residual(&point)
                    .is_some_and(|after| after < before);
                if better {
                    improved = true;
                    break;
                }
                point[index] = original;
            }
        }

        if problem.is_feasible(&point, 0.0) {
            return Some(point);
        }
        if !improved {
            break;
        }
    }

    None
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

/// Asks a solver for a first point, once rejection sampling has failed to find
/// one.
///
/// Sampling coming up empty does not prove a region is empty; only a solver can
/// say that, which is why this is the one path able to produce
/// [`Infeasibility::Proved`](crate::Infeasibility::Proved).
///
/// # Errors
/// Transport and process failures. A solver *concluding* something — including
/// that it cannot decide — is a [`Verdict`], not an error.
pub(crate) fn escalate_for_seed(
    problem: &ConstraintSystem,
    logic: &SmtLogic,
    limit: u32,
    cancel: &Cancellation<'_>,
) -> Answer {
    seed_away_from(problem, logic, limit, &[], 0.0, cancel)
}

/// A point at least `reach` away, on some coordinate, from everything in
/// `avoid`.
///
/// With an empty `avoid` this is [`escalate_for_seed`] — "find a point" and
/// "find a *different* point" are one question asked with nothing and with
/// something to stay away from. See [`smtlib::emit_away_from`] for the shape of
/// the exclusion and why it needs a `reach` at all.
///
/// # The verdicts do not mean what they mean above
///
/// **[`Verdict::Impossible`] is not unsatisfiability** once `avoid` is
/// non-empty. It means nothing satisfies the constraints *and* stands `reach`
/// away from what is held — which is either "the region is covered" or "`reach`
/// is too large", and the caller distinguishes them by shrinking `reach` and
/// asking again. Reporting it as
/// [`Infeasibility::Proved`](crate::Infeasibility::Proved)
/// would be badly wrong.
///
/// # Errors
/// As [`escalate_for_seed`].
pub(crate) fn seed_away_from(
    problem: &ConstraintSystem,
    logic: &SmtLogic,
    limit: u32,
    avoid: &[Point],
    reach: f64,
    cancel: &Cancellation<'_>,
) -> Answer {
    let inputs = problem.variables();
    let written = problem
        .constraints
        .iter()
        .map(|constraint| &constraint.written);
    let document = smtlib::emit_away_from(inputs, written, logic, avoid, reach);
    let unexpressed = document.untranslated;

    let Reply { outcome, spent } = Z3Backend.solve(&document.text, limit, cancel);
    let verdict = match outcome {
        Outcome::Unsat { blamed } if unexpressed.is_empty() => Verdict::Impossible { blamed },

        // "Nothing satisfies the constraints we wrote down" is a much weaker
        // claim than "nothing satisfies the constraints" when some were left
        // out, and the difference is exactly the one worth not eliding.
        Outcome::Unsat { .. } | Outcome::Unknown => Verdict::Inconclusive { unexpressed },

        Outcome::Sat(values) => Verdict::Seed {
            // The model names variables; a `Point` is positional. Anything the
            // solver did not pin — an auxiliary, or a variable left free —
            // simply is not in the model, so fall back to the lower bound and
            // let the pool's filter judge the result.
            point: inputs
                .iter()
                .map(|input| {
                    values
                        .iter()
                        .find(|(name, _)| *name == input.name)
                        .map_or(input.lower_bound, |(_, value)| *value)
                })
                .collect(),
            unexpressed,
        },
    };
    Answer { verdict, spent }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InputVariable;

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
    /// The second case is the one that matters more: the nudge must not rescue
    /// a point that is simply infeasible, or `Unsatisfiable` stops meaning
    /// anything.
    #[test]
    fn a_boundary_witness_is_adjusted_and_a_wrong_one_is_not() {
        let system = crate::system::tests::system(
            vec![InputVariable::new("x1", 0.0, 10.0)],
            &["x1 == pi +/- 0.001"],
        );

        // The value Z3 actually returns, as a decimal parsed back into f64 —
        // not `PI - 0.001`, which Rust computes to a *different* f64 and which
        // happens to land inside. That difference is the entire bug.
        let edge: f64 = "3.140592653589793".parse().expect("a literal");
        assert!(
            !system.is_feasible(&[edge], 0.0),
            "this test is pointless unless the boundary really does miss"
        );
        let adjusted_edge = adjusted(&system, vec![edge]).expect("a near-miss should be adjusted");
        assert!(system.is_feasible(&adjusted_edge, 0.0));
        assert!(
            (adjusted_edge[0] - edge).abs() < 1e-12,
            "the adjustment moved the point {} away from the witness, which is not a nudge",
            (adjusted_edge[0] - edge).abs()
        );

        assert!(
            adjusted(&system, vec![7.0]).is_none(),
            "a point nowhere near the band was 'adjusted' into feasibility"
        );
    }

    /// How many assertions a solver actually took from a document.
    ///
    /// Compared against how many were written. Z3 stops parsing at the first
    /// error, so a malformed *tail* leaves the earlier assertions in place and
    /// the non-empty guard in `solve` never fires. A short count is the only way
    /// that shows up.
    fn assertions_taken(document: &str) -> usize {
        let solver = z3::Solver::new();
        solver.from_string(document);
        solver.get_assertions().len()
    }

    /// A document Z3 has to work on — `floor` mixes integers into a
    /// nonlinear system — comes back `unknown` at the limit instead of
    /// running for however long it takes.
    ///
    /// This is what makes a dropped `solve` future cost a bounded amount of
    /// solver time, and what the calibration of `DEFAULT_SOLVER_LIMIT` rests
    /// on: this instance runs about eight seconds per million units on the
    /// laptop, and past ten million it ran for over ten minutes before it was
    /// killed, so the scaling is not to be trusted at the high end.
    #[test]
    fn a_hard_instance_gives_up_at_the_limit_instead_of_hanging() {
        let inputs = vec![
            InputVariable::new("x", 0.0, 100.0),
            InputVariable::new("y", 0.0, 100.0),
            InputVariable::new("z", 0.0, 100.0),
        ];
        let constraints: Vec<crate::Ast> = [
            "floor(x) * floor(y) == floor(z) * 7 + 3 +/- 0.000000001",
            "x*y*z == 12345.678 +/- 0.000000001",
            "x^2 + y^2 == z^2 + 1 +/- 0.000000001",
        ]
        .iter()
        .map(|s| crate::parse(s).expect("fixture should parse"))
        .collect();
        let document = smtlib::emit_away_from(
            &inputs,
            &constraints,
            &crate::cvg::SmtLogic::default(),
            &[],
            0.0,
        );

        let started = std::time::Instant::now();
        let reply = Z3Backend.solve(&document.text, 30_000, &Cancellation::never());
        let took = started.elapsed();
        assert_eq!(reply.outcome, Outcome::Unknown, "{reply:?}");
        assert!(
            reply.spent >= 30_000,
            "a call that hit its limit should report at least the limit spent, not {}",
            reply.spent
        );
        assert!(
            took < std::time::Duration::from_secs(5),
            "thirty thousand units took {took:?}; the limit is not being applied"
        );
    }

    /// The hard fixture with no rlimit at all, cancelled two hundred
    /// milliseconds in: the interrupt lands and the call answers `unknown`
    /// well inside the grace period, where before it ran to its limit on an
    /// abandoned thread.
    #[test]
    fn a_cancelled_call_is_interrupted_within_the_grace() {
        let inputs = vec![
            InputVariable::new("x", 0.0, 100.0),
            InputVariable::new("y", 0.0, 100.0),
            InputVariable::new("z", 0.0, 100.0),
        ];
        let constraints: Vec<crate::Ast> = [
            "floor(x) * floor(y) == floor(z) * 7 + 3 +/- 0.000000001",
            "x*y*z == 12345.678 +/- 0.000000001",
            "x^2 + y^2 == z^2 + 1 +/- 0.000000001",
        ]
        .iter()
        .map(|s| crate::parse(s).expect("fixture should parse"))
        .collect();
        let document = smtlib::emit_away_from(
            &inputs,
            &constraints,
            &crate::cvg::SmtLogic::default(),
            &[],
            0.0,
        );

        let (sender, receiver) = futures_channel::oneshot::channel::<super::super::Opening>();
        let timer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            drop(receiver);
        });
        let started = Instant::now();
        let reply = Z3Backend.solve(&document.text, 0, &Cancellation::watching(&sender));
        let took = started.elapsed();
        timer.join().expect("the timer thread joins");

        assert_eq!(reply.outcome, Outcome::Unknown, "{reply:?}");
        assert!(
            took < Duration::from_millis(200) + INTERRUPT_GRACE,
            "cancelled at 200ms and answered after {took:?}; the interrupt did not land"
        );
    }

    /// The abandon branch, without Z3: a worker that never answers is given
    /// the ceiling, then the interrupt, then the grace, and then given up on.
    /// Z3 is never used here because a genuinely ignored interrupt leaves a
    /// thread running inside Z3 past the end of the test, which on Linux can
    /// crash static teardown after `main` has returned.
    #[test]
    fn an_unanswered_interrupt_is_abandoned_after_the_grace() {
        let (sender, answers) = mpsc::channel::<Reply>();
        // A worker that holds its sender and never answers, until released.
        let (release, hold) = mpsc::channel::<()>();
        let worker = std::thread::spawn(move || {
            let _sender = sender;
            let _ = hold.recv();
        });
        let interrupts = std::cell::Cell::new(0_u32);
        let started = Instant::now();
        let answer = await_answer(
            &answers,
            worker,
            &Cancellation::never(),
            || interrupts.set(interrupts.get() + 1),
            Duration::from_millis(100),
            Duration::from_millis(200),
        );
        let took = started.elapsed();
        drop(release);

        assert!(answer.is_none(), "{answer:?}");
        assert_eq!(interrupts.get(), 1, "the interrupt is sent exactly once");
        assert!(
            took >= Duration::from_millis(300) && took < Duration::from_secs(2),
            "ceiling and grace should add up to about 300ms, took {took:?}"
        );
    }

    /// A worker that dies is its own panic on the caller's thread, not a hang
    /// and not an abandonment.
    #[test]
    #[should_panic(expected = "the worker fell over")]
    fn a_dead_worker_is_its_panic_rather_than_a_wait() {
        let (sender, answers) = mpsc::channel::<Reply>();
        let worker = std::thread::spawn(move || {
            let _sender = sender;
            panic!("the worker fell over");
        });
        let answer = await_answer(
            &answers,
            worker,
            &Cancellation::never(),
            || panic!("nothing to interrupt"),
            Duration::from_secs(60),
            Duration::from_secs(60),
        );
        // Not reached: the disconnected channel resumes the worker's panic.
        drop(answer);
    }

    /// The ceiling is far past honest work and never absent.
    #[test]
    fn the_ceiling_is_generous_and_never_unbounded() {
        assert_eq!(
            ceiling(crate::DEFAULT_SOLVER_LIMIT),
            Duration::from_secs(480)
        );
        assert_eq!(ceiling(30_000), Duration::from_secs(60), "the floor");
        assert_eq!(
            ceiling(0),
            Duration::from_secs(3600),
            "no limit is not no ceiling"
        );
    }

    /// Every shape the emitter can produce, run past Z3 to see if it parses.
    ///
    /// This is the closest thing to a linter available without writing one, and
    /// it is worth more than a parenthesis counter: Z3 checks sorts, arities and
    /// scoping too, so a `Real` where an `Int` belongs or an auxiliary used
    /// before its declaration fails here rather than silently.
    ///
    /// It only works because [`Z3Backend::solve`] refuses a document that
    /// produced no assertions. `Solver::from_string` reports a syntax error by
    /// accepting nothing and then answering `sat`, so without that guard this
    /// test would pass on garbage.
    #[test]
    fn every_document_the_emitter_can_build_parses() {
        let box_of = |names: &[&str]| -> Vec<InputVariable> {
            names
                .iter()
                .map(|name| InputVariable::new(*name, -8.0, 8.0))
                .collect()
        };

        let cases: Vec<(Vec<InputVariable>, &str)> = vec![
            // literals, including the shapes `real` has to special-case
            (box_of(&["x"]), "x > 4"),
            (box_of(&["x"]), "x >= -1.5"),
            (box_of(&["x"]), "x < 0.00001"),
            (box_of(&["x"]), "x > pi"),
            (box_of(&["x"]), "x > e"),
            // every unary that translates
            (box_of(&["x"]), "abs(x) > 1"),
            (box_of(&["x"]), "sqr(x) > 1"),
            (box_of(&["x"]), "cube(x) > 1"),
            (box_of(&["x"]), "sgn(x) > 0"),
            (box_of(&["x"]), "sqrt(x) > 1"),
            (box_of(&["x"]), "cbrt(x) > 1"),
            (box_of(&["x"]), "-x > 1"),
            // every binary that translates
            (box_of(&["x", "y"]), "x + y > 1"),
            (box_of(&["x", "y"]), "x - y > 1"),
            (box_of(&["x", "y"]), "x * y > 1"),
            (box_of(&["x", "y"]), "x / y > 1"),
            (box_of(&["x", "y"]), "max(x, y) > 1"),
            (box_of(&["x", "y"]), "min(x, y) > 1"),
            (box_of(&["x"]), "x^3 > 1"),
            (box_of(&["x"]), "x^-2 > 1"),
            (box_of(&["x"]), "x^0 > 1"),
            // structure: folds, blocks, subscripts, equality-with-tolerance
            (box_of(&["a", "b", "c"]), "sum(1, 3, i -> var[i]) > 2"),
            (box_of(&["a", "b", "c"]), "prod(1, 3, i -> var[i]) > 2"),
            (box_of(&["x"]), "var a = x * 2; var b = a + 1; b > 3"),
            (box_of(&["x", "y"]), "x == y +/- 0.001"),
            // nesting deep enough that a stray parenthesis would show
            (box_of(&["x", "y"]), "abs(sqrt(abs(x)) - cbrt(y / 2)) < 1"),
            // a name the lexer allows and SMT-LIB needs quoting for
            (box_of(&["λ"]), "λ > 0.5"),
        ];

        for (inputs, source) in cases {
            let constraint = crate::parse(source).expect("test constraint should compile");
            let document = smtlib::emit_away_from(
                &inputs,
                std::slice::from_ref(&constraint),
                &crate::cvg::SmtLogic::default(),
                &[],
                0.0,
            );
            assert!(
                document.untranslated.is_empty(),
                "{source:?} is not meant to be beyond the emitter"
            );

            // The parse guard must not fire on a document the emitter wrote.
            drop(Z3Backend.solve(&document.text, 0, &Cancellation::never()));

            // Z3 stopping partway is not an error, just a shorter document.
            let written = document.text.matches("(assert ").count();
            assert_eq!(
                assertions_taken(&document.text),
                written,
                "Z3 took fewer assertions than were written for {source:?}, so it stopped parsing partway:\n{}",
                document.text
            );
        }
    }

    /// What Z3 can and cannot be asked, measured rather than assumed.
    ///
    /// This is the evidence behind [`super::smtlib`] refusing the transcendentals
    /// outright, and it is deliberately a *canary*: it asserts a negative
    /// capability, so the day a Z3 upgrade grows one of these, this test fails
    /// and tells us the refusal is now costing something.
    ///
    /// The summary, as of Z3 4.15:
    ///
    /// | asked | answered |
    /// |---|---|
    /// | `sin`, `cos` | parse, then `unknown` on anything narrow |
    /// | `ln`, `log`, `exp`, `sqrt` | not in the grammar at all |
    /// | `^` with a real exponent | works — and on `(^ x 1.234)` runs minutes past `rlimit` and `timeout` |
    ///
    /// The middle row is why emitting transcendentals would buy nothing, and the
    /// first row is why it would be worse than nothing: `unknown` on precisely
    /// the narrow regions a solver is wanted for, at the cost of the search time
    /// spent finding that out.
    #[test]
    fn z3_still_cannot_help_with_transcendentals() {
        let declare = "(declare-const x Real)(declare-const y Real)";

        // Names Z3's parser does not know: it takes no assertion from a
        // document using one, which is the emitter's refusal vindicated.
        for unknown_name in [
            "(> (ln x) 2.0)",
            "(> (log x) 2.0)",
            "(> (exp x) 2.0)",
            "(= (sqrt x) 3.0)",
        ] {
            assert_eq!(
                assertions_taken(&format!("{declare}(assert {unknown_name})")),
                0,
                "Z3 has learned {unknown_name} — the emitter could now emit it"
            );
        }

        // `sin` parses, which is the trap: it looks supported right up until the
        // problem is one worth solving.
        let narrow = format!(
            "{declare}(assert (and (>= x 0.0) (<= x 3.0)))(assert (= y (sin x)))(assert (> y 0.99))"
        );
        assert_eq!(
            Z3Backend.solve(&narrow, 0, &Cancellation::never()).outcome,
            Outcome::Unknown,
            "Z3 has learned to decide narrow trigonometry"
        );

        // `^` with a real exponent does work — so `sqrt` could have been
        // `(^ x 0.5)` rather than an auxiliary variable. The auxiliary stays,
        // because it is standard SMT-LIB where `^` is a Z3 extension.
        let root = format!(
            "{declare}(assert (and (>= x 0.0) (<= x 100.0)))(assert (= y (^ x 0.5)))(assert (> y 3.0))"
        );
        assert!(matches!(
            Z3Backend.solve(&root, 0, &Cancellation::never()).outcome,
            Outcome::Sat(_)
        ));
    }

    #[test]
    fn a_model_value_too_big_for_i64_still_reads_back() {
        // Z3 answers in rationals — `2.5` comes back as `(/ 5.0 2.0)` — and
        // `as_rational` is `Z3_get_numeral_small`, which fails outright when
        // either half overruns `i64`. The fallback then has to be right, and at
        // `approx_f64`'s 17 decimal *places* it was not: this value is about
        // 1.01e-23 and used to read back as a confident 0.0.
        let document = "(declare-const x Real)
                        (assert (= x (/ 1.0 98765432109876543210987.0)))
";
        let Outcome::Sat(values) = Z3Backend.solve(document, 0, &Cancellation::never()).outcome
        else {
            panic!("a pinned value should be satisfiable");
        };
        let (_, value) = values
            .iter()
            .find(|(n, _)| n == "x")
            .expect("x is in the model");
        assert!(
            (value - 1.012_499_999_886_093_7e-23).abs() < 1e-31,
            "tiny rational came back as {value}"
        );
    }

    #[test]
    fn a_model_value_that_is_not_a_numeral_does_not_panic() {
        // `Real::approx_f64` is `parse().unwrap()`, and not every model value is
        // a decimal — asking Z3 about `pi` hands back something symbolic, which
        // used to take the worker thread down with it. Skipping the variable
        // leaves it at its lower bound and lets the pool filter the point.
        let reply = Z3Backend.solve(
            "(declare-const x Real)(assert (> x pi))",
            0,
            &Cancellation::never(),
        );
        assert!(matches!(reply.outcome, Outcome::Sat(_)));
        assert!(
            reply.spent > 0,
            "a finished call should report the work it did"
        );
    }

    #[test]
    #[should_panic(expected = "parsed no assertions")]
    fn the_parse_guard_actually_catches_a_bad_document() {
        // The guard above is the whole reason the previous test means anything,
        // so it needs its own proof that it fires.
        let broken = "(set-logic QF_NRA)
(declare-const x Real)
(assert (this is not smtlib))
";
        drop(Z3Backend.solve(broken, 0, &Cancellation::never()));
    }

    #[test]
    #[should_panic(expected = "parsed no assertions")]
    fn a_parse_error_anywhere_loses_the_whole_document() {
        // This decides whether the non-empty guard in `solve` is *sufficient*.
        // The worry was a malformed tail: earlier assertions parse, the solver
        // is not empty, and the guard waves it through with the last constraint
        // silently missing. Z3 turns out not to work that way — it keeps nothing
        // at all — so the guard covers a bad tail as well as a bad head.
        //
        // Pinned rather than deleted: if Z3 ever became more forgiving, this is
        // the test that fails, and the guard would have to start counting.
        let truncated = concat!(
            "(set-logic QF_NRA)\n",
            "(declare-const x Real)\n",
            "(assert (>= x 0.0))\n",
            "(assert (<= x ((((("
        );
        assert_eq!(
            assertions_taken(truncated),
            0,
            "Z3 kept some assertions from a document with a syntax error in it"
        );
        // And the guard fires on it, which is the `should_panic`.
        drop(Z3Backend.solve(truncated, 0, &Cancellation::never()));
    }
}
