# Sojourn — notes for agents

Sojourn is a constrained random vector generator: given a box and a set of constraints
it produces random points that satisfy them, by sampling, by walking, and by asking Z3.
Constraints are written in babel, a small expression language: `x1 + x2 * cos(x3)^2`
is a transform, `x1 < x2 + x3` is a constraint. Babel began as Kotlin/ANTLR on the JVM
(EmpowerOps' optimizer used it) and was ported to Rust here; the generator began as the
university project sojourn-CVG and is `crate::cvg`.

Read these before changing anything, in this order:

1. [`src/README.md`](src/README.md) — the architecture:
   one `Ast`, a meaning-preserving front end, two backends (`eval`, `cvg`).
2. [`docs/todo.md`](docs/todo.md) — the roadmap *and* the reasoning: measurements, dead ends,
   and the decisions that are not recoverable from the code. Part two is long on
   purpose. Add to it when you learn something the code cannot say.
3. [`performance-records/README.md`](performance-records/README.md)
   — how to read and write a throughput number honestly.
4. [`docs/brute-squad.md`](docs/brute-squad.md) — the plan for wide-batch
   sampling (IR tape, CPU vectorisation, wgpu). Owns the "sample harder" tier;
   `docs/todo.md` owns the solver and equality-constraint side.

## Layout

| path | what | status |
|---|---|---|
| `Cargo.toml`, `src/`, `tests/`, `templates/` | the Rust crate, at the repository root. One package and no workspace; when a second crate appears (an FFI `cdylib`, say) it gets a sibling directory and the root `Cargo.toml` gains a `[workspace]` table. | live |
| `src/lib.rs`, `src/system.rs`, `src/solve.rs`, `src/repair.rs` | the public API, as files: `compile` for one expression, the validated system, how to solve it into a `FeasibleRegion`, and the repair the region offers. Source text goes in everywhere and no syntax tree comes out; `lib.rs` re-exports exactly this surface and nothing from the directories below. | live |
| `src/cvg/` | the search engine — private. Strategies, the ladder, the worker, the local solve (`local.rs`, COBYLA from `basin`), the SMT emitter, the GPU sieve. Reachable from `tests/` only through the `#[doc(hidden)]` re-exports in `lib.rs`. | live |
| `grammar/*.g4` | the ANTLR grammar. `build.rs` regenerates the lexer and parser from it into `OUT_DIR`. | live |
| `performance-records/` | throughput ledgers, written by the benchmarks; see its README | live |
| `docs/sojourn/` | notes and statement of intent from the original CVG project, whose code became `crate::cvg` | reference |
| `Justfile`, `.github/workflows/rust.yml` | CI is exactly `just ci` | live |

`src/frontend/generated.rs` is ANTLR output; never hand-edit it.

The JVM implementation this crate was ported from was deleted in 1c26ed9. The
Kotlin fixtures are the spec the Rust tests were ported from — `corpus.rs` ←
`BabelExpressionFixture.kt`, `cvg_pools.rs` ← `Z3SolvingPoolFixture.kt`, etc. —
and live at `git show 6813e0d:src/test/kotlin/com/empowerops/babel/`.

## Build and test

Everything runs from the repository root (the Justfile uses `pwsh`).

```
just build          cargo fmt, then cargo build --all-targets   (also regenerates the parser)
just test-compile   cargo test --no-run         MUST stay green
just test           cargo nextest run --no-fail-fast
just lint           clippy -D warnings, check-only; formatting is build's job
just bench          release-mode throughput, writes performance-records/*.csv
just brute          time-to-first-hit rungs + checks/s, release, machine otherwise idle
```

- Use **nextest**, not `cargo test`: the AST is recursive and a stack overflow in one
  test must not take the binary with it. `.config/nextest.toml` sets a 60 s
  slow-timeout, overridden to 300 s for `cvg_benchmarks` — every problem there runs
  ten seeds, so `top_corner_200d` legitimately takes ~150 s and the equality twin ~350 s.
- The `z3` crate is built with `bundled`, so a cold build compiles Z3 from source and
  needs CMake plus a C++ toolchain (MSVC on Windows). Slow the first time, cached after.
- `antlr-rust-codegen` pulls in RustPython; the lockfile currently wants a recent
  stable rustc. If `cargo build` complains about `requires rustc 1.9x`, update the
  toolchain rather than downgrading dependencies.
- The environment variable `SOJOURN_SMT_LOGIC` overrides the SMT-LIB logic (default
  `QF_NIRA`).

## How to work here

**TDD.** The port was driven test-first and the tests are the spec. A new behaviour
starts as a failing test in `tests/` (integration, public API) or a
`#[cfg(test)]` module beside the code (unit). Red tests are acceptable on a
feature branch; tests that fail to *compile* are not — that is an incomplete API.

**A `panic!` beats a spin.** A hang is the worst failure mode this crate has —
the 20-segment beam ran 3.7 CPU-hours without returning before anyone knew why —
so every foreign call, and every API of ours with even a remote chance of
combinatorial explosion or exponential backoff, carries a budget that ends it:
a resource limit in the callee's own units where one exists (`rlimit` for Z3,
an evaluation count for COBYLA), and otherwise a wall-clock ceiling set so
conservatively that reaching it means a bug, not a slow case — the five-sigma
use. Reaching it must fail loudly (an error, a `tracing::error!` and
abandonment, a `panic!`) and never wait. `smt::Z3Backend::solve` is the model:
rlimit first, a ceiling twenty times past honest work, interrupt, grace,
abandon-and-report. A budget is a count and decides the answer deterministically;
a ceiling is a watchdog and only decides that something is broken. Keep them
distinct, and never let a ceiling become the thing that decides a result.
The full inventory of the engine's loops and the hedges considered is in
`docs/todo.md` under "Hanging is the worst failure mode".

**Call functions by their module.** Import modules and types; call functions
qualified — `eval::bind(..)`, `cvg::serve(..)`, `ast::to_index(..)` — rather
than importing the bare name. A four-letter verb says nothing about which part
of the system is speaking, and the qualifier is what makes a call site readable
to someone who has not memorised the tree. Re-exports in `lib.rs` are the one
exception, since they *define* the surface.

**Assertions are exact by default.** Only cases that route through libm carry a
tolerance. Do not add blanket tolerances to make something pass.

**The tape is the only evaluator, and the tests are its spec.** `eval/` lowers
the AST to a three-address tape and runs it tiled or per lane. It was held to
the tree-walker it replaced on a few thousand random and adversarial rows, then
the walker was deleted. The spec is `tests/corpus.rs`, `tests/runtime_errors.rs`
and `tests/special_values.rs`: plain tests with hand-written expectations. Add
cases there; never a recorded-output file. The CPU tape checks for non-finite
values on every instruction; only a future GPU sieve is allowed to be coarse,
and it must re-run an offending column through the tape for the span rather
than be "fixed" to match (see src/README.md).

**Neither backend's lowering is visible to the other.** The front end produces the
canonical form of what the author wrote and nothing more. If a pass makes the tree
easier to *analyse*, it belongs in `frontend::rewrite`; if it makes it faster to
*run*, it belongs in `eval`; if it makes it *emittable* to a solver, in `cvg::smtlib`.
The `<= 0 is true` residual convention is `eval`'s, not the language's.

`a == b +/- t` is the worked example of that seam. `eval::lower` desugars it into
`b - t`, `b + t`, a comparison against each, and a `Worst` fold — `Compare::Gte`
is `right - left` and `Compare::Lte` is `left - right`, so the result is
`max((b - t) - a, a - (b + t))` node for node, and `corpus.rs` pins that by
value. That deleted the fused instruction from the tape, the scalar executor, the
SIMD kernel, the tiled path and WGSL. **It stops there deliberately**: doing it in
`frontend::rewrite` would leave `cvg::classify` looking at two comparisons, and
an equality does not merely bound a variable, it *determines* one — which is a
dependency claim no pair of inequalities makes, and the thing driving is built
on. Recovering it would mean re-pairing two `Compare` nodes by structure, which
`fold_constants` or `invert_monotone` can disturb on one side and not the other.

**SIMD is explicit.** The tile executor's kernels live in `eval/simd.rs`, built
on `pulp` with the instruction set picked at run time. Every operator is either
a named vector kernel or a named `*_scalar` one; do not rely on auto-vectorisation
anywhere. Never use pulp's `mul_add` (fused on every backend) or its `max`/`min`
(x86 semantics, not NaN-propagating). The crate has no `unsafe`; keep it that way.

**The pool's ladder is probe, local solve, solver, brute force.** `cvg`'s
uniform sampler (`Strategy::BruteSquad`) probes with one brute-force batch —
tens of microseconds — and delivers where that lands often enough. Where it
lands nothing, a local solve goes first (`Strategy::LocalSolve`, `cvg/local.rs`):
COBYLA from the box centre and a few seeded starts, stopped at the first point
the oracle judges feasible — 24 evaluations and 150 ms on the 100-segment
stepped beam, where Z3 needs minutes per query. Only where every start fails is
Z3 asked, under a resource limit (`with_solver_limit`, in Z3's own units so the
answer is machine-independent) — it settles a contradiction in milliseconds,
which is now its role: the proof that there is nothing to find. It answers
`unknown` on anything transcendental or past the limit, and only what it could
not decide gets brute force: the same sampler on every core for a proposal
budget (`with_proposal_budget`, default a billion). What brute force finds is a function of the seed and the budget, never
of the thread count — keep it that way (the batch is the unit of randomness).
`Strategy` is a test-only configuration, not a user-facing one; the fairness
oracles in `tests/cvg_benchmarks.rs` measure against the same sampler. Pool
tests run with `common::PROPOSAL_BUDGET`, a million under debug, because the
default takes minutes on an unoptimised tape. The pool's state is a value:
`cvg::progress::Progress`, threaded through `serve` → `open` → `keep_filling`
and folded with `absorb`/`extend`, never a field. `ConstraintSystem` is
immutable and compiled once; `Ladder` holds only the strategies' streams and
knobs, and the SMT logic a document is emitted under. Keep it that way — the
only `&mut` in the search is an RNG or a walker's chain.

**An equality is read before it is searched.** `cvg::classify` reads
`a == b +/- t` and answers what can be concluded: `Pinned`, `Driven`, `Implicit`
or `Opaque`. A `Driven` variable is one the walker *computes* rather than
searches, which is what lets it move along a measure-zero surface instead of
jittering beside it — `classify::plan` turns a system into the schema positions
the walker moves and the ones it computes, in evaluation order, and `classify::retract`
applies it. Three rules hold the whole thing up:

- **Driving is a Gibbs draw, not an evaluation.** `y == f(x) +/- t` admits the
  whole band, so `retract` draws uniformly from `f(free) ± t`. Assigning
  `y = f(free)` collapses the band to its centre line and throws away a
  dimension — on two hundred pinned variables it returned the same point two
  hundred times. Drawing from a conditional slice is the move that leaves the
  uniform distribution invariant; evaluating to the centre is not.
- **A drive is a proposal, never a rewrite.** Feasibility is re-checked against
  every constraint afterwards, and `retract` skips a coordinate whose definition
  will not evaluate. So a wrong or over-narrow isolation costs rejected moves and
  never a wrong point — which is why the isolation table can be aggressive.
  Refusing to drive is always safe; no constraint is ever dropped.
- **A variable named on both sides makes the equality *implicit* in it**, and
  `ConstraintSystem::new` refuses it, naming the rearrangement (`a == b + a/2` is
  `a/2 - b == 0`). Not "cyclic" — a cycle is a mutual dependency *between*
  equations, which `plan` meets and handles by driving neither. Two narrower
  rules were tried and discarded; both are written up in todo.md.

`classify::reaches` answers whether the operators between an equality's root and
a variable that occurs **exactly once** (*linear* in it, in the term-rewriting
sense) can all be undone, so `x1 + x2 == 3` drives `x1`. It answers *whether*,
not *what*: it used to build the rearrangement `3 - x2` for `retract` to
evaluate, and `interval::slice` derives that band by narrowing the constraint
itself, so the expression lost its consumer and the walk down the path is all
that survives. The arithmetic those rules encoded lives in
`interval::invert_binary`, tested there against the same cases — the two arms
where operands do not commute (`a - u == c`, `a / u == c`) are still where a swap
is silently wrong.

**What can be reached is exactly what `interval` can invert**, and the two are
held together by `interval::invertible_unary` / `invertible_binary` with a test
that the predicates match the tables. Claiming a coordinate is driven and then
handing the walker its whole box for it is the one combination that *stalls*,
where an honest refusal only wastes a proposal.

That set is wider than the old `isolate` allowed, and the reason is worth
keeping: `abs`, `sqr` and `cosh` are not injective, so a **symbolic** inverse
would have to choose a branch and be silently wrong half the time — which is why
`isolate`, which built an expression, refused them. Narrowing does not choose; it
intersects both branches with what the argument can already be. `^`, `%`, `max`,
`min` and the periodic functions still decline.

Two holes are red on purpose. Driving assumes the feasible set is a **graph**
over the free coordinates, so `x1 * x2 == 0` is a cross with one arm unreachable.
And where two equations would drive the *same* variable, `plan` refuses to choose
and drives neither — that is the bipartite matching todo.md carries as unbuilt.

`var[i]` is resolved at `ConstraintSystem::new` — the first moment a schema
exists, since `parse` has none and `Kind::Global` indexes the expression's own
symbols while `var[i]` indexes the schema. After that
`Ast::contains_dynamic_lookup` means "a subscript nothing could resolve" rather
than "a subscript", and nothing downstream special-cases one.

**A constraint says what interval a coordinate may take.** `cvg::interval` is
HC4-revise: evaluate an expression forward over a box, then push the requirement
that the constraint be *true* back down through each operator's inverse.
`interval::slice` intersects that across every constraint naming a coordinate, and
both the walker's axis moves and `retract` draw from it.

**Every interval is a superset of what it models, and that asymmetry is the
whole design.** A value drawn from a superset and then judged by `is_feasible`
is, conditioned on acceptance, distributed exactly as one drawn from the true
slice — so an interval that is too wide costs a rejected proposal and one that is
too narrow removes reachable points and biases the answer silently. Everything
follows: anything without an inverse answers `ENTIRE` and degrades to the
behaviour that already shipped, which is what let this land in stages. The
type is hand-rolled rather than `inari` because padding a few ulps outward buys
soundness without rounding-mode control, and tightness is a dial rather than a
requirement.

The boolean at a constraint's root is the **only** place the kind of comparison
is read — it supplies a target interval, and everything below is one uniform
backward pass. `a == b +/- t` gives `[-t, t]`; the two bounds it desugars to
would give `(-inf, t]` intersected with `[-t, inf)`, which is the same interval.

**A coordinate about to be recomputed is not one to condition on.** `retract`
marks a driven coordinate settled only once it has been drawn, and skips any
constraint naming an unsettled one. Conditioning `y` on a `z` that is itself
about to be recomputed from `y` pins the pair within a tolerance of each other,
and they shuffle by `t` a sweep instead of travelling — three occupied cells of
eighty, where twenty-four are wanted. This is why `classify::Plan`'s topological
order cannot be replaced by a per-coordinate Gibbs sweep, and the attempt is
written up in todo.md.

**Every candidate is judged by the whole system.** There used to be a
restricted check for an axis move — only the constraints naming the moved
coordinate, via `Incidence::affected` — with the precondition that the point it
was derived from had been feasible. Re-measured on 2026-09-12 it bought nothing
outside noise (270 s against 255 s on `top_corner_200d`), so it went, and with
it a precondition a caller could get wrong. `Incidence::affected` still serves
the repair clamp's per-coordinate filter, where "which walls can this landing
see" is the actual question.

**`ConstraintSystem` is data, and every strategy borrows one.**
`ConstraintSystem::new` (in `src/system.rs`) compiles every constraint to prove
it binds and keeps the tape beside the AST as one `Constraint`, along with the
drive plan and the incidence graph, both derived while proving the set fits
together. Its fields are crate-visible and it answers the simple questions
only: `is_feasible` (with a clearance) and `worst_residual`. The moves are the
engine's, as free functions over `&ConstraintSystem` in the module that owns
the idea — `classify::retract` and `classify::settle` for the plan,
`interval::slice` for the narrowing question, `smt::adjusted` for the ulp
nudge of a solver's witness that landed a hair outside in `f64`. There is no
wrapper type around the system: the one thing a solver call needs beyond it,
the SMT logic, is a field of `Ladder` and a parameter of `cvg::smt`.

**`FeasibleRegion` is the solved system.** `ConstraintSolver::solve` takes the
system by reference and clones it twice, once for the worker thread and once
for the region, so the region can answer for the system after the search:
`system()`, `take` for samples, and `repair` — which lives here rather than
on the system because a region that could not be solved has nothing to repair
toward. `repair` draws no randomness — not a seed, not a step — and lands a
coordinate at the caller's clearance inside its bound, *on* the bound at zero
clearance; the design and the alternatives it displaced are in `docs/todo.md`
under *Repair for Artemis*.

`cvg::incidence` is the bipartite graph of constraints and coordinates, kept in
both directions because the walker traverses it both ways. Its indices are
newtypes — `Row` for a schema position, `ConstraintId` for a position in the
constraint list — because both directions are lists of `usize` that mean
different things, and a transpose built the wrong way round reads identically as
bare integers. Three such swaps were tried against the newtypes and all three
are now compile errors.

Two things belong in `affected` that a naive reading of the constraint's symbols
misses, and both are soundness rather than efficiency: every constraint naming a
**driven** coordinate, because retraction moves those whatever was swept; and
every constraint carrying an **unresolved `var[i]`**, because it reads a column
chosen by the point and no symbol list names it. Skipping either accepts a point
the full check would reject.

**Every benchmark runs on ten seeds and requires all ten.** One seed cannot tell
a real change from a lucky draw: `top_corner_200d` was failing on about half of
all seeds and passing on the committed one, so a green tick was reporting the
seed rather than the sampler. `cvg_benchmarks::run` drives `REPLICATES`
independent trials through `catch_unwind` and names every seed that failed. It
costs a tenfold runtime, which is why that binary has its own timeout in
`.config/nextest.toml`.

**A KS test needs the effective sample size, and that is pooled across
coordinates.** The walker emits round-robin across its chains, so successive
points come from *different* chains and the autocorrelation sits at lag
`CHAIN_COUNT` rather than lag one. Two things follow, and both were learned the
hard way. Sokal's window closes at `WINDOW_FACTOR * tau`, which with `tau` near
one is lag five — **before** lag eight, so `autocorrelation_time` forces the
window past the interleave before that rule may close it. And per coordinate the
signal sits at its own noise floor, so estimating from one column is a coin
flip; the interleave belongs to the *emission*, shared by every column, so
`rho_k` is averaged over all of them. Getting this wrong reported `tau = 1.33`
where the truth is 1.85, made the threshold 36% too tight, and read as the
sampler being broken. Any new statistic compared here needs the same treatment —
never `values.len()`.

**Another language is never built with a string builder.** WGSL and SMT-LIB
both go through askama templates under `templates/`, compiled at
build time against views in `eval/wgsl.rs`, `cvg/sieve.rs` and `cvg/smtlib.rs`.
The semantics — which helper, which guard, what is refused — stay in Rust; the
syntax lives in files that read as the language they produce, with one macro
arm per operator, and the operator types the templates match over are the
subsets the target language can spell, so a missing arm is a compile error. A
`format!` that writes a brace, a parenthesis, an operator or a keyword of
another language is the smell to refuse. Template output is validated (naga,
Z3's parser), checked for the substrings that matter and for balance, and never
recorded to a file.

**The GPU is a sieve and never a judge.** Behind the opt-in `gpu` feature
(`just brute`, `just bench` and `just test-gpu` turn it on), brute force runs
on whatever wgpu adapter is present: the tape is rendered as
WGSL through the templates, candidates are drawn and judged on the device in `f32`
with a slack, and *every survivor is re-judged exactly on the CPU*. A false
negative costs hit rate; a false positive costs a CPU check; neither changes an
answer. Shader compilers assume no NaNs, so the emitter guards every operator's
domain with a comparison rather than relying on NaN propagation — keep it that
way. Shader text is validated with naga and compared against the CPU, never
recorded to a file. The GPU path is deterministic per device, not across
machines; `with_gpu(false)` is the reproducible path, and `SOJOURN_GPU` picks the
adapter (`off`, an index, or a name) and logs the list when set. Every wait on the device
has a timeout, and the device is held only while a brute-force search is
using it — a `Weak` in the module, an `Arc` in each live sieve — never for the
life of the process. The default build has no wgpu in it and must stay that
way; `test-compile` builds with every feature so the GPU code cannot rot.

**`sum` and `prod` bounds are constants.** Both are unrolled at compile time; a
bound that depends on a variable is a compile error, not a loop. That feature was
dropped deliberately (todo.md, "Dropped features") — do not reintroduce a
run-time aggregate without reading why.

**Nothing non-finite travels.** NaN/inf is a compile error where provable
(`ProblemKind::NonFiniteConstant`) and a runtime error otherwise
(`ProblemKind::NonFiniteValue`), reported against the innermost span.

**Measure before claiming a speedup.** Run-to-run noise on throughput is ~30%.
Compare medians of several runs, in one sitting, with an untouched case as a control,
against the parent commit. Benchmarks are release-only; a debug number is meaningless
and under upsert would overwrite a good row.

**Z3 is the solver, and its limits are known.** No logarithms, no `e`, `sin`/`cos`
parse but answer `unknown`, `^` with a variable exponent answers `unknown`, `^` with
a negative base and fractional exponent is unsound for babel's `cbrt`. The rewrite
pass `invert_monotone` and each backend's own lowering of a whole exponent to
multiplication (`Expr::whole_exponent`) route around this; the metric that matters is
`Document::untranslated`. cvc5 and dReal were evaluated and rejected;
the table is in todo.md under "The solver question, settled". Do not re-shop for a
solver without a new fact.

**`Solver::from_string` returns `()`.** A malformed SMT-LIB document leaves an empty
solver that answers `sat` with an empty model. Every verdict in `cvg::smt` is gated
on the assertions having arrived; keep it that way.

## Style

- Doc comments explain *why* and record what was measured; the code says what.
  Match that register — the module headers are the model.
- Prefer a type that makes the mistake unrepresentable (`Progress`, the `Slot`
  binding table, `SmtUnary`) over a
  check that reports it.
- Public API is batch-only: `CompiledExpression::eval(MatRef) -> Col<f64>`, one
  column per sample, one row per schema variable. `eval_row` is crate-private for
  the walker and is the same tape through the per-lane executor, not a second
  implementation.
- Non-obvious decisions go in `todo.md` part two, with the measurement that
  justified them.
