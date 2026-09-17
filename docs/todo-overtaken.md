# Todos overtaken by events

Entries that were open in `todo.md` on 2026-09-16 and are moot: overtaken by the
synchronous `solve`, built in another form, or duplicated. Moved here verbatim, each
with a line on why, so `todo.md` lists only what is live. Nothing here is a plan.

## Overtaken by the synchronous `solve` (2026-09-16)

- [x] **Hanging is the worst failure mode; a `panic!` beats a spin.** Three hedges were
      considered on 2026-09-13 against bugs not yet written; none is adopted here, and the
      third is a discipline recorded in `AGENTS.md`.
      1. *Classify every loop as a budget or an invariant, and guard the invariants with a cap
         that panics.* Rejected: it depends on the author correctly deciding a loop is safe,
         which is exactly the judgement that fails when it fails.
      2. *A heartbeat the worker bumps in every loop and a watchdog on the handle side:*
         `take_within(Duration)`, a `Drop` that joins with a deadline and abandons the thread
         with an error-level trace when it passes (the Z3 leash's shape), and
         `Status::Failed("no progress for 30 s in <stage>")`. Converts unknown future hangs into
         a fact the caller can read, changes no result. Practical, not yet committed to: the
         `&& ticker.tick_and_check()` in every loop predicate may make the code structurally
         worse, and it is not clear the crate has a loop that needs it — every loop in the
         engine today is bounded by a named count or a finite set (`GAP_QUERIES`,
         `SHRINK_LIMIT`, `CLAMP_SWEEPS`, `CHORD_BITS`, `LANDING_LADDER`, `ADJUST_SWEEPS`, the
         starts and evaluations of the local solve, burn-in and thinning by dimension, the
         proposal budget; `classify::plan`'s ordering consumes a finite set; `narrow` is a
         single pass by design). The two places the failure mode actually lives are foreign
         code (Z3, leashed; COBYLA, budgeted per evaluation but not per iteration) and the
         handle waiting on the worker (`take` unbounded, `Drop` joins unconditionally — the
         burn-in that ignored the stop flag was found by exactly that).
      3. *Every foreign call and every API with a remote chance of combinatorial explosion gets
         a resource limit, or failing that a wall-clock ceiling set so conservatively that
         hitting it means a bug, and hitting it panics rather than waits.* Adopted as a
         discipline; the leash on Z3 was the model until Z3 left, and with it went the last
         wall-clock ceiling in the crate — every loop that remains is a count, branch-and-prune
         included (a contraction budget, a visit cap, a progress threshold). COBYLA's callback
         is the one place today that lacks a per-iteration bound.
      *Why moot:* The worker, its handle and `take_within` are gone: `solve` runs on the calling thread and returns when its count budgets are spent. The one discipline this entry kept — a `panic!` beats a spin — is in `AGENTS.md`.

- [x] **Concurrent coverage: start the walker from the first hit, cover gaps beside it.**
      Settled in discussion on 2026-09-12 as the follow-up to the budget above. Cases: (A)
      sampling carries — coverage is already skipped; (B) the solver finds the seed and the
      walker carries — today it waits for coverage, up to one limit's worth of Z3, before a
      point flows; (C) a second component exists that hit-and-run cannot reach — coverage is
      what finds it. Running coverage concurrently and merging a new component's seed by
      burning a chain in under the frozen preconditioner makes B pay nothing, and C gains its
      component later rather than never. The price is exact and worth writing down before it
      is paid: in case C the census becomes a function of *when* the seed arrived relative to
      the walker's batches, so the same seed gives different streams on different runs, which
      the two-run oracle in `cvg_benchmarks` and Artemis's same-seed test both rest on. In A
      and B the output is identical either way. To be chosen, not drifted into.

      The split that made it work: **one-shot and ongoing are different asynchronies.** "Crack the
      first point" has a completion and belongs to a `Future`; "keep filling between requests" has
      none and belongs to a worker. Trying to make one `Future` carry both is why it previously fit
      neither.
      Today's `ConstraintPool` became `Generator`, private to the worker and never shared, so there
      is nothing to lock. The public `ConstraintPool` is a handle: a `Receiver`, a buffer, and a
      stop flag. `std::sync::mpsc::sync_channel` supplies the rest for free — a bounded channel
      *is* the high-water mark, and `TryRecvError`'s `Empty`/`Disconnected` *is* the
      starved-versus-exhausted distinction that decides whether a caller waits or gives up.
      `generate` blocks until it has the count or the pool is exhausted, which cannot hang because
      exhaustion is detectable, and `status` covers the rest. `found()` and `try_generate` were
      both deleted: the first had no callers, and the second was speculative — letting every call
      block is simpler and the blocking is bounded anyway. `Status::Failed` carries the panic
      message rather than sitting beside a separate `failure()` accessor.
      *Why moot:* Nothing runs after `solve` returns, so there is nothing to run coverage concurrently with; coverage is part of the opening, and a design is walked from the region's burnt-in chains on demand.

- [x] **Suite runtime is 31 seconds**, nearly all of it `top_corner_200d` at 26 — which was
      already 23 before the worker landed, because burn-in for a 200-dimensional chain is genuinely
      expensive. The extra three seconds are read-ahead: `BATCH_SIZE * CHANNEL_CAPACITY` points get
      produced whether or not anybody asks for them, and at 200 dimensions a point is about four
      hundred walker moves. Both constants are deliberately small for that reason.
      Plus about four minutes on a cold build for Z3's C++, which caches.
      `.config/nextest.toml` sets a 60s slow-timeout with `terminate-after`, so a stuck test reports
      instead of stalling CI. The threshold has to clear the slowest *honest* test by a wide margin
      — an earlier 30s sat close enough to flag `top_corner_200d` as slow, which just teaches people
      to ignore the warning. Each emitted point costs about
      one feasibility evaluation per shrink, times thinning, and the distribution oracles each cost
      a whole extra solve. `TopCorner200D` alone is ~10s. Tolerable now; worth watching.
      *Why moot:* Read-ahead does not exist. The suite is ~45 s in debug, most of it the walker's burn-in on every `solve`; `tests/profiling.rs` has the release numbers.

- [x] **`Solution` wants an accessor.** Getting the pool out means writing
      `Solution::Satisfied(pool) | Solution::Unknown { pool, .. } => pool` at every call site. The
      Kotlin had a `Worthwhile` supertype over exactly those two cases; a `fn pool(&mut self)` or
      `fn into_pool(self)` is the Rust equivalent and removes the repetition.
      *Why moot:* `Solution` became `Result<FeasibleRegion, Infeasibility>`; `points()` and `witness()` are the accessors.

## Built, in another form

- [x] **A typed `SolveError`.** `solve` still returns `anyhow::Result`, which is
      right for a binary and loose for a library.
      *Why moot:* `ConstraintSolver::solve` returns `Result<FeasibleRegion, Infeasibility>`.

- [x] **Reverse-mode autodiff over the tape.** Roughly 100 lines once the tape exists, and probably
      worth more to the expensive-constraint / penalty-function work than raw throughput.
      *Why moot:* `src/eval/differentiate.rs`: reverse mode over the single-assignment tape, one rule per instruction; what Newton-KKT repair reads.

- [x] **GPU** via wgpu or CubeCL, once the batch tape shows the shape is right.
      *Why moot:* `src/cvg/sieve.rs`: wgpu, one compute shader per problem, candidates drawn on the device.

- [x] **`UNSAT` as a babel diagnostic, not just a sampling verdict.** Separate from everything
      above, and probably the higher-value half of wiring up a solver. A user who writes two
      constraints that cannot both hold currently gets silence — the pool searches, finds nothing,
      and reports having found nothing, which looks identical to a region that is merely hard. A
      solver's `check()` distinguishes "no point exists" from "we did not find one", and that is a
      *compile-time* answer: report it through `ProblemKind` alongside the syntax errors, where the
      user is already looking.
      **Most of the machinery now exists**: the emitter names its assertions, `Z3Backend` reads
      unsat cores back, and `Solution::Unsatisfiable` carries the conflicting constraints. What is
      missing is only the *entry point* — `compile()` sees one expression and no input variables, so
      reporting this through `ProblemKind` needs somewhere new for a caller to hand over a whole
      constraint set. Notably it needs no pool, no sampling and no strategy selection. It also
      applies to constraint sets that sample perfectly well, which is the case the
      "solver only when sampling is hard" framing misses entirely. Nothing in sojourn-CVG does
      this today; it is new work this repo is now positioned for.
      *Why moot:* `Infeasibility::Unsatisfiable` names the conflicting constraints when branch-and-prune proves the system empty, and `cvg_pools::contradictory_constraints_are_reported_as_unsatisfiable` pins it; Z3 is gone. What was not built is the framing — a compile-time answer with no solve — and it is not wanted: the proof is a contraction of the whole system, which is a `solve` with a prune budget.

- [x] **Translation is currently infallible, and its `Result` is a lie.** Every error path in the
      translator was an `unsupported` problem; with those gone there is no reachable `Err`, and
      `SemanticTranslator` has no fields left — a unit struct with `&self` methods.
      **Deliberately left alone.** The semantic checks above make it genuinely fallible again and
      give the struct real state back (it needs the source to build problems). Revisit once they
      land: if they do not materialise, collapse the `Result` and turn the methods into free
      functions.
      *Why moot:* the revisit happened by itself. `SemanticTranslator { source }` has its field back and two reachable `Err`s — a non-finite or non-positive tolerance on `a == b +/- t` (`src/frontend/parse.rs`) — so the `Result` is true again. All `pub(crate)`; nothing here was ever public.

- [x] **Split `Display` on the `{}` / `{:#}` boundary.** `Problem` always renders the full
      caret block, which is wrong for a log line. Plain `{}` should be the one-line summary and
      `{:#}` the block with source and caret.
      Note this does not recover Kotlin's `abbreviatedProblemText`, which elided lambda bodies
      (`sum(0/0,20,i->...)`) using the parse tree. `Display` has only source and span, so it
      renders `source[span]`. Whitespace collapsing is reachable; lambda elision is not.
      *Why moot:* built. `impl Display for Problem` (`src/diagnostics.rs`) renders the one-liner under `{}` and the caret block under `{:#}`, every wrapper forwards the flag, and `compile_errors::plain_display_is_a_single_line` / `alternate_display_renders_a_caret_block` pin both. The lambda-elision caveat stands.

## Duplicates

- [x] **The JVM tree does not compile on this branch**, so the Kotlin benchmark cannot be run in
      place. Commit `db9add8` ("POrting to rust") commented out four `locals [...]` declarations in
      `BabelParser.g4` — `availability`, `closedValue`, `value` — which `rewriters.kt` still
      depends on, giving nine unresolved references. Presumably the Rust ANTLR codegen would not
      accept them.
      The numbers above came from a throwaway `git worktree` at `db9add8^`, the last commit where
      it built; the worktree has been removed. `ThroughputBenchmarks.kt` is committed to the real
      tree and will run the moment the grammar is restored — or it can be deleted along with the
      rest of the JVM tree, which is already on this list.
      *Why moot:* The standing entry is *The JVM tree does not compile* under "Wave 3"; this was the older copy, kept for its detail on which grammar rules went and where the ledger rows came from.
