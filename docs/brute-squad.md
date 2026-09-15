# I am the brute squad

*2026-09-14: a record. `Strategy::Solver` and the Z3 it named are gone; the proof and the
gap coverage this plan escalated to are `Strategy::Prune`, interval branch-and-prune, and
the ladder is contract, probe, local solve, bisect, brute force — see `docs/todo.md`,
"Z3's fate". Nothing below was rewritten to match.*

*"Sample harder."* A plan for the regime where the feasible region is a
1e-6 to 1e-9 fraction of the box, the SMT solver cannot help — usually because
the constraint contains a transcendental — and the honest answer is to bring
more hardware to the party until a first feasible point lands. Once one has
landed, the existing land-and-expand machinery (adaptive seeding, hit-and-run)
takes over and this work is done.

Companion to [`todo.md`](todo.md), which owns the solver and equality-constraint
work. **Nothing here touches the Z3 or AST-rewrite side.** That work is in
flight on the other machine.

---

## The regimes

| feasible fraction | what works | who is responsible |
|---|---|---|
| > 1e-2 | plain rejection sampling, then the walker | the pool as it is today |
| 1e-2 .. 1e-6 | adaptive sampling seeds, the walker emits | the pool as it is today |
| **1e-6 .. 1e-9** | **wide-batch sampling on every core and any GPU present** | **this plan** |
| < 1e-9 | a solver, or a smarter search. No amount of silicon helps. | `todo.md` |

Time to first hit is geometric: expected proposals is one over the hit rate,
and the tail is heavy — a quarter of runs need 1.4x the expectation, one in
twenty needs 3x.

| hit rate | expected proposals | at 1M checks/s | at 40M checks/s |
|---|---|---|---|
| 1e-4 | 10 thousand | instant | instant |
| 1e-6 | 1 million | 1 s | 25 ms |
| 1e-8 | 100 million | 100 s | 2.5 s |
| 1e-10 | 10 billion | 3 hours | 4 min |
| 6e-61 (`top_corner_200d`) | never | never | never |

So the window hardware buys is about three orders of magnitude. That is still
where users' awkward-but-not-impossible constraints live, and it is the window
in which Z3's `unknown` on `sin` currently leaves them with nothing.

**Targets.** North of 1,000,000 constraint checks per second on a realistic
constraint, on the CPU alone. The number to beat from LGO is 40k, and the
tree-walker already does ~37M/s on `x1 + x2` at batch 256 on BATOU — so the
CPU target is a *pipeline* problem, not an evaluator problem. The GPU is
headroom for heavy constraints, not the thing that gets us over the line.

**Where this sits in the pool.** Today: probe with the fair sampler; ≥10% hit
rate routes to plain sampling, otherwise adaptive seeding plus the walker; Z3
is consulted only when the first batch is empty. The brute squad is the tier
between "first batch empty" and "ask Z3": blast a wide batch for a time budget,
and only then escalate. Eventually Z3 and the sampler should run concurrently
under one deadline — Z3 is single-threaded, the machine has sixteen threads,
whichever lands a point first wins — but the deadline and cancellation story
belongs with the solver work.

---

## The steps

### 0. Red tests first — done 2026-09-02

`tests/brute_squad.rs`, run with `just brute`. What landed, and what was decided
on the way:

- **The solver became `Strategy::Solver`**, last in `DEFAULT_STRATEGIES`. A
  strategy list alone could not keep Z3 out — the worker escalated whenever the
  first batch was empty — and a builder flag would have been a second knob that
  drifts from the list tests already pin. The list is now the escalation
  ladder, in order, and step 4's tier is one more variant before `Solver`.
- **The budget is enforced by polling `solve()` by hand** with a no-op waker and
  dropping the future at the deadline. Its body never yields and the work is
  already on a worker thread, so nothing in the library had to learn about
  deadlines. It leaks that worker until its current batch returns — milliseconds
  today, up to a budget once a tier loops — so a case stops at two misses rather
  than running a third attempt into a degraded machine. Step 4's deadline
  removes the leak.
- **Two of three attempts, and the B/E ≥ 3 rule.** With `m = exp(-B/E)` a green
  rung fails with probability `3m² - 2m³`: 4.9% at B/E = 2, 0.7% at 3. A tier
  turns a rung green only when its expected first-hit time is at most a third
  of the budget.
- **Ignored in debug, release only.** `just test` runs a p = 1e-2 smoke, the
  ladder pin and the checks/s count check; the twelve budgeted rungs need
  `just brute`, which is `--no-fail-fast` because red rungs are the point.
- **The trimmed grid**: 1e-4/1e-6/1e-8 for all three families, 1e-10 for the
  corner and the sine corner, 1e-12 for the corner only.

First run, on the laptop (DESKTOP-KH0S3BP, Ryzen 7 PRO 4750U), the whole file
in 8 s:

| rung | result | how |
|---|---|---|
| 1e-4, all three families | green | found in ~18 ms; seed 1 misses the corner and sine corner, seeds 2 and 3 hit. Deterministic. |
| 1e-6 and below, all nine | **red** | `gave up after ~4 ms` — the pool proposes ~13,200 candidates and stops |

And the checks/s ledger, first row, tree-walker as it stands:

| family | eval-only | pipeline | RNG cost |
|---|---|---|---|
| corner | 12.8M/s | 11.8M/s | 8% |
| ball | 11.6M/s | 10.3M/s | 11% |
| sine corner | 8.2M/s | 7.5M/s | 9% |

Two things this says before any tier is built. The evaluator is already twelve
times past the one-million target on one laptop core, so **the 1e-6 rung is red
purely because the pool gives up, not because checking is slow** — step 4's
loop alone should turn it green, and probably 1e-8 with all cores. And the RNG
is about a tenth of the pipeline, so fusing it into the batch path is worth
having but is not where the order of magnitude lives.

Original plan follows.

Two fixtures, both red until the rest lands, both passing a strategy list with
**no solver in it** — Z3 answers `x1 > 0.999999` instantly and would make the
whole fixture a lie. `DEFAULT_STRATEGIES` is already `pub`, so no environment
variable or config switch is needed.

- **Throughput.** Constraint checks per second on a fixed set of constraints,
  amortised over several batches. Seconds, not minutes. Records into
  `../performance-records/` alongside the evaluator ledgers, same format, same
  caveats: release only, ~30% noise floor, compare medians in one sitting.
- **Time to first hit**, over a family of constraints with a tunable feasible
  fraction: 1e-4, 1e-6, 1e-8, and a **permanently red 1e-10** so we know what
  we are dealing with. Every case has a time budget and asserts a first
  feasible point arrives inside it. A test that never finishes hits the 60 s
  nextest timeout and burns two minutes per run; a budgeted one fails fast
  and reports how far it got.

The family should include at least one transcendental so the case Z3 refuses
is the case being measured.

### 1. The IR tape and a naive evaluator — done 2026-09-02

Measured on the laptop against the step-0 commit (`23d735c`, the tree-walker),
five rounds interleaved old/new in one sitting, medians. Evaluator ledgers in
points per millisecond:

| case | width | walker | tape | gain |
|---|---|---|---|---|
| `x1 + x2` | 1 | 5552 | 5299 | 0.95x |
| `x1 + x2` | 256 | 41412 | 227184 | **5.5x** |
| `x1 + x2 > 20 - x3^2` | 1 | 4536 | 3775 | 0.83x |
| `x1 + x2 > 20 - x3^2` | 256 | 16128 | 131127 | **8.1x** |
| `sin*cos+sqrt*abs` | 1 | 3940 | 3455 | 0.88x |
| `sin*cos+sqrt*abs` | 256 | 10374 | 35595 | **3.4x** |
| deep arithmetic | 1 | 2067 | 2011 | 0.97x |
| deep arithmetic | 256 | 3155 | 56734 | **18x** |
| `sum(1, 200, i -> var[i]^2 - 3.0)` | 1 | 93 | 112 | 1.2x |
| `sum(1, 200, i -> var[i]^2 - 3.0)` | 256 | 94 | 1796 | **19x** |

Checks per second through the brute-squad fixture, 1024-wide batches:

| family | eval-only | pipeline |
|---|---|---|
| corner | 14.0M → 105.6M (7.5x) | 10.7M → 31.4M (2.9x) |
| ball | 12.4M → 126.5M (10.2x) | 9.2M → 31.9M (3.5x) |
| sine corner | 8.5M → 38.9M (4.6x) | 7.0M → 20.4M (2.9x) |

The ranges did not overlap on any batch-256 row, so none of this is drift.
What it says:

- **Batch of one is flat to slightly worse**, as predicted: a register file and
  a fault table per call against the walker's two buffers. The per-lane path
  the sampler actually uses (`eval_row`) is the 200-variable row's 1.2x —
  a flat tape beats a recursive walk even one row at a time.
- **The transcendental case gained 3.4x from dispatch alone.** libm is still
  scalar; the gain is the tree walk that used to surround each call.
- **The RNG is now the bottleneck.** `eval-only` is 105M/s on the corner
  family but `pipeline` is 31M/s: the evaluator went 7.5x and the pipeline
  3x, because filling 3072 doubles from `StdRng` per batch costs more than
  judging them now does. That is step 2's first item, ahead of any SIMD work:
  a faster generator (xoshiro or a counter-based one) and filling the
  matrix in bulk.
- No `multiversion`, no SIMD crate, no `target-cpu`. This is auto-vectorised
  SSE2 and the shape of the loops; AVX2 dispatch is still on the table.

Original sketch follows.

"Obvious code to an obvious baseline." Lower `Ast` → `Tape`, then interpret it.

**Three-address form, not a stack.** A stack machine is a register machine
with a fixed implicit allocation, and the implicitness is what hurts: every
target wants names.

- Emission to WGSL is one `let t7 = t3 * t4;` per instruction.
- Batched CPU execution runs each instruction as one loop over the batch —
  `regs[dst][..] = op(regs[a][..], regs[b][..])`. The dispatch cost is
  amortised over the batch width, which is the entire win over the tree walk.
- Register reuse falls out of a trivial liveness pass because the code is
  straight-line. Fewer live registers means a smaller working set:
  `registers × batch_width × 8 bytes`, which is what picks the tile size.

Sketch:

```rust
struct Reg(u16);

enum Insn {
    Load   { dst: Reg, input: u16 },                    // schema row → register
    Unary  { dst: Reg, op: UnaryOp, a: Reg },
    Binary { dst: Reg, op: BinaryOp, a: Reg, b: Reg },
    // Compare / NearEq / And lower to Sub, Max and a Const epsilon.
    // Fold lowers to a chain of Binary.
}

struct Tape {
    consts: Vec<f64>,   // preloaded as broadcast registers 0..k
    insns: Vec<Insn>,
    registers: u16,
    result: Reg,
}
```

Constants live out of the instruction stream so every instruction packs into
eight bytes.

**Everything lowers.** As built (2026-09-02): `var[expr]` becomes a `Gather`; a
literal subscript that names a real row becomes a plain `Load`, so the
200-variable case spends no constant registers on its indices. Run-time-bounded
aggregates were first given loop instructions and a per-lane path, then dropped
outright on 2026-09-03: a feature nobody uses that silently cost the batch
path was worse than a compile error. Every tape is straight-line.

**Two things decided differently from the sketch above:**

- The sampler restructure is its own step, after this one, measured by
  checks/s and the 1e-6 rungs rather than by the evaluator ledgers.
- The tape reuses `UnaryOp::apply` / `BinaryOp::apply`, Java NaN semantics
  and all, rather than a branch-free `max`. The finite test is fused into each
  instruction's loop as an or-reduction and each lane records its first fault,
  so the tape is as eager as the walker was at no happy-path cost. Only a GPU
  sieve gets to be coarse.

**Test:** a one-off differential, then plain tests. While the tree-walker still
existed the tape was held to it over 121 expressions on seeded random and
adversarial rows, exact bits or fault kind and span; then the walker went, and
so did the recorded file — a recorded-output oracle is the tape checking the
tape. What remains is `tests/corpus.rs`, `tests/runtime_errors.rs` and
`tests/special_values.rs` (signed zeros to the bit, operations that go
non-finite faulting at their own span, every fault kind planted in a batch
naming its column), and the two executors held to each other on random rows.

### 2. Explicit SIMD, a fast generator, batched judging — done 2026-09-03

What landed, against the sketch below:

- **Explicit kernels via `pulp`** (`eval/simd.rs`), dispatched once per tile
  onto AVX2 or the scalar backend. Every operator is a named vector kernel or a
  named `*_scalar` one; nothing relies on auto-vectorisation any more. Scalar
  by name: libm, `%`, `pow`, `log(a, b)`, `ceil`/`floor` (pulp's generic trait
  has no rounding), `sgn`, `Load`, `Gather`. Two pulp traps found on the way:
  its `max`/`min` are the x86 instructions and do not propagate NaN, and its
  any-lane helper is wrong on the scalar backend (a mask is a `bool` there and
  the lane count works out to zero). Both are worked around and pinned.
- **`f64::max` on signed zeros is not even self-consistent on this toolchain**:
  constant-folded in release it answers `+0.0` for both orders, at run time it
  answers the second operand. Babel's `apply` now spells out Java's rule —
  `max(-0.0, 0.0)` is `0.0`, `min` is `-0.0`, either way round — and the
  kernels reproduce it with a bitwise and/or on the equal case.
- **Xoshiro256++** replaces ChaCha12 everywhere in the pool, with `fill_box`
  filling a candidate matrix in bulk from the top 53 bits of each draw,
  half-open. `PointSource::generate` returns a matrix, and `Search::produce`
  judges it in one call through a crate-private `CompiledExpression::holds`,
  a per-column predicate on which a faulting candidate simply does not hold —
  the same verdict `is_feasible` gave a faulting point, with no NaN-as-value
  anywhere. A candidate is no longer a `Vec` each.

Measured on the laptop against the step-1 binaries, five interleaved rounds,
medians. Evaluator ledgers, points per millisecond:

| case | width | step 1 | step 2 | gain |
|---|---|---|---|---|
| `x1 + x2` | 256 | 222343 | 215625 | 0.97x |
| `x1 + x2 > 20 - x3^2` | 256 | 127637 | 135228 | 1.06x |
| `sin*cos+sqrt*abs` | 256 | 34975 | 39612 | 1.13x |
| deep arithmetic | 256 | 56376 | 70114 | 1.24x |
| `sum(1, 200, i -> var[i]^2 - 3.0)` | 256 | 1796 | 2322 | 1.29x |
| every case | 1 | | | 0.84x–0.91x |

Checks per second, 1024-wide batches:

| family | eval-only | pipeline |
|---|---|---|
| corner | 105.9M → 95.0M (0.90x) | 31.6M → **69.4M (2.2x)** |
| ball | 125.1M → 134.9M (1.08x) | 31.5M → **85.7M (2.7x)** |
| sine corner | 38.9M → 38.0M (0.98x) | 20.2M → **33.4M (1.65x)** |

What it says:

- **The generator was the win.** Pipeline throughput 2.2x to 2.7x on the
  arithmetic families, exactly the bottleneck step 1 named.
- **Explicit SIMD bought little at the evaluator.** Batch-256 moved 0.97x to
  1.29x, the most on the heaviest tapes. The auto-vectorised SSE2 loops were
  already within a whisker of this, which says the tile executor is bound by
  memory traffic — two kilobytes read and written per operand per instruction —
  not by lane count. Twice the lanes cannot help a loop that is waiting on L1.
  The next evaluator gain is fusing instructions so intermediates stay in
  registers, not wider instructions; that is a step for later, and the
  measurement is what says so.
- **Batch-1 lost about 10%**: the dispatch and the four slice splits per
  instruction are pure overhead on a one-lane tile. The sampler never runs a
  one-lane tile now, so it is the ledger's degenerate case and nobody's path.
- **The value of explicitness is not in this table.** It is that
  `dispatch_unary!` is an exhaustive match where "scalar" is a word you can
  grep for, and that a loop cannot quietly fall back on a compiler upgrade.

**Every seeded verdict re-rolled**, as predicted. `parabolic_roots_ribbon`
landed with no point in its far band and is red; `todo.md` says why and what
the honest fixes are, and the assertion was not loosened. Two of the three
1e-4 brute-squad rungs re-rolled red too — the corner and the sine corner now
land on one seed of three where they landed on two — which is the fixture's
own arithmetic (E[hits] ≈ 1.3 per first batch) and not a regression. Step 4's
loop is what makes those rungs stop being a coin.

Original sketch follows.

Faer will not vectorise these loops; its SIMD lives inside its own kernels
and elementwise ops are not among them. But if step 1's inner loops are zipped
slice iterators with no early exit, LLVM auto-vectorises `+ - * /`, `sqrt`,
`abs`, `max`, `min`, `floor`, `ceil` on its own. Transcendentals and `%` stay
scalar libm calls, which is fine for a sieve.

The catches:

- **The default x86-64 target is SSE2**, two lanes. AVX2 needs runtime
  dispatch: the `multiversion` crate does it with one attribute and no SIMD
  API. Reach for `pulp` only if the assembly says auto-vectorisation failed.
- **Batch width is a cache decision**, not a convenience. Pick it so
  `registers × width × 8` fits L1 or L2.

Transcendentals on CPU, if they ever matter enough: range reduction plus a
minimax polynomial, which is FMA chains and vectorises. Not tables — a gather
is the weak SIMD instruction. Sieve tier only; the exact evaluator judges.

Steps 1 and 2 may collapse into one.

### 3. GPU via wgpu, kernels as generated WGSL — done 2026-09-04

What landed, against the sketch below:

- **`eval/wgsl.rs`**: the tape as one WGSL function per constraint, over a
  pointer to an `array<f32, N>`, returning the residual; the third backend
  over the tape. Constants inlined as f32 literals; `Compare` and `NearEq`
  carry a relative slack of `1e-4`; every operator with a domain is guarded
  by a comparison and returns a fault residual outside it. Validated with
  naga on any machine; no recorded shader text anywhere.
- **`cvg/sieve.rs`**: one process-wide adapter and device, one shader per
  problem with two entry points — `sieve_generated` draws candidates on the
  device from a 32-bit hash of `(base, batch, lane, dimension)`, `sieve_given`
  takes a batch the CPU drew — an explicit bind-group layout, an error scope
  and a 30 s timeout around every dispatch, and survivors sorted by lane so
  the atomic append order does not leak. Behind the opt-in `gpu` feature, and the
  device is held only while a search is using it — `Weak` in the module, `Arc` in the
  sieve — so a library does not keep a host's GPU open after it is done.
- **Brute force chooses its engine**: `RandomSampler::brute_force` runs the
  sequential GPU loop when a sieve exists and falls back to the CPU loop,
  from the same base, if the device stops answering.
  `ConstraintSolver::with_gpu(false)` is the reproducible path.
- **Two things the sketch did not know.** Shader compilers assume no NaNs —
  this driver evaluates `NaN <= 0` as true, so `sqrt` of a negative *passed*
  the sieve until the domain guards went in. And an auto-derived bind-group
  layout rejects the other entry point's bind group silently; explicit layout,
  error scopes.

Measured on the laptop's iGPU (AMD Radeon Graphics, Vega class, over Vulkan),
release, `just brute`:

| | CPU, one thread | CPU, sixteen threads | iGPU |
|---|---|---|---|
| candidates judged per second, corner | 76M (pipeline) | ~450M | **2.0G** (drawn and sieved on the device) |
| candidates judged per second, sine corner | 34M | — | **2.0G** |
| uploaded batch, sieved | — | — | 51M (upload-bound) |

The device is twenty-five times one thread and four times all sixteen, and
the sine corner costs it nothing extra — the transcendentals really are the
special function units' problem. The upload path is the wrong shape for the
device and exists for the tests and the dial.

The rungs: 1e-4 through 1e-8 green on every family, in about 200 ms each — of
which the search is a few milliseconds and the rest is connecting to the
adapter and compiling the shader, paid per solve because the device is
released when brute force returns. 1e-10 spends the
thirty-billion budget in about fifteen seconds here, past the rung's ten-second
wall, and reports `timed out`; the rung is sized for a desktop card, where the
budget should take a second or two. BATOU's RX 7800 XT numbers go here when it
has run this. 1e-12 permanently red.

A first dial, before the survivor readback was moved off the measured path,
read 380M for the device — the CPU parsing forty thousand survivor rows per
dispatch at one in a hundred, not the GPU. The `generated` rate is now measured
at one in a million, where brute force actually runs.

Original sketch follows.


The dumbest path is the right one, because the kernel comes from a *user's
expression at run time*. Every "compile Rust to GPU" project (rust-gpu,
rust-cuda, CubeCL) compiles functions known at build time; that is not this
problem. Every GPU driver already ships a JIT for shader source. So: the tape
becomes WGSL text, one line per instruction, and wgpu hands it to whatever
adapter is present — Vulkan, DX12 or Metal, on AMD, NVIDIA, Intel or an iGPU.

- **This is the oneAPI dream in miniature.** The tape is the one
  implementation; the CPU interpreter and the WGSL emitter are two small
  backends over it, structurally the same as the SMT-LIB emitter.
- **f32.** WGSL has f32 natively; f64 is a non-standard naga extension and f16
  has too few digits to rank residuals on. The GPU is a *sieve*: keep anything
  whose f32 residual is below a small positive slack, then re-evaluate
  survivors exactly on the CPU. It never changes an answer, only throughput.
- **Transcendentals are NVIDIA's problem.** WGSL has `sin cos tan asin acos
  atan sinh cosh tanh log log2 exp pow sqrt abs floor ceil sign max min %`,
  all on the special function units. `ln` is `log`, `log10` is `log(x)/ln10`,
  `cbrt` is `sign(x) * pow(abs(x), 1/3)`. Throughput drops on a heavy
  constraint and that is fine; optimism is the complexity reducer.
- **Optimistic connect.** Request an adapter; if none, or the shader fails to
  compile, run the CPU tape. A few lines.
- Development on this laptop's Vega iGPU (wgpu over Vulkan); real numbers from
  BATOU's RX 7800 XT.

Repair proper — nudging a near-miss onto the surface — stays parked with the
equality work in `todo.md`. The sieve only needs the exact re-check.

### 4. Wire it into the pool — done 2026-09-03

Done before step 3, because the loop is what the step-0 fixtures were
actually missing: the CPU pipeline was already judging eighty-odd million
candidates a second per core and the pool gave up after ten thousand.

What landed:

- **Same rung, renamed.** `Strategy::UniformSampling` is `Strategy::BruteSquad`.
  The uniform sampler still probes with 10,000 candidates and delivers where
  that lands often enough; where it lands nothing, and Z3 — if configured —
  could not settle it, it keeps proposing on every core until a batch lands
  or a proposal budget is spent. No fourth variant, no seeder concept: `AdaptiveSampling` went
  the same day. The probe and the delivery streams are untouched, so no
  seeded verdict moved.
- **A proposal-count budget**, `ConstraintSolver::with_proposal_budget`,
  default a billion. A count rather than a duration so the same seed finds
  the same point on every machine. Zero disables the loop.
- **The batch is the unit of randomness.** Batch `k` is seeded from
  `(base, k)`; every thread walks its own arithmetic progression of batch
  numbers and stops once past a known winner; the lowest-numbered batch with a
  hit is the answer. One thread and eight land bit-identical points — a test
  pins it. The only thing the thread count changes is `proposed`, by however
  many batches the other threads had in flight.
- **Dropping the `solve` future cancels.** The worker polls
  `oneshot::Sender::is_canceled` between batches on the calling thread and
  stops within one; the fixture's leaked worker is gone. A Z3 call in
  progress runs on to its resource limit — `with_solver_limit`, a count of
  Z3's own work rather than a clock, default three million units, about
  twenty-five seconds here on a hard mixed-integer instance — and then
  answers `unknown`, which hands over to brute force like any other.
- **The probe is one brute-force batch**, not ten thousand candidates: under
  three thousand for three variables, tens of microseconds. The easy-path
  threshold is the hit rate it always was, one in a thousand, below which
  the hundredfold delivery batches go empty often enough to trip the
  exhaustion check.

The rungs, `just brute`, release, sixteen threads, three seeds each:

| rung | corner | ball | sine corner |
|---|---|---|---|
| 1e-4 | 32–42 ms | 16–18 ms | 46–48 ms |
| 1e-6 | 51–60 ms | 23–30 ms | 38–72 ms |
| 1e-8 | 0.11–0.59 s | 80–134 ms | 0.09–0.70 s |
| 1e-10 | gave up, ~3.5 s | — | gave up, ~6 s |
| 1e-12 | timed out at 1 s | — | — |

Thirteen of sixteen green; the three red are the GPU sieve's rungs, as
step 0 said. The billion-proposal budget costs 3.3 s on the plain corner and
about twice that on the sine corner, sixteen threads. The rate is a cache
number before it is an evaluator one: at a 4 MiB batch sixteen threads
managed 137M proposals a second and were slower than eight, because two
hyperthreads' batches did not fit one core's L2; at 64 KiB they manage 450M.
One thread does 75M at any batch size.

**Refactored the same day.** `Search` — one struct with the problem, the
strategies and mutable `found`/`route` state — became three things with
different lifetimes: an immutable `Problem` (constraints compiled once), a
`Ladder` of strategies holding only their streams and knobs, and a `Progress`
value threaded through the worker and folded, never mutated in place. The
worker is `serve` → `open` (probe, solver, brute force, straight-line) →
`keep_filling`; the route is pinned by the first trial and recorded in the
value. `todo.md` has the two behaviour changes; no verdict moved.

**Solver before brute force.** The first cut ran them the other way round,
and every problem the probe cannot land — an equality ribbon, a
contradiction, `top_corner_200d` — spent the whole budget, three to seven
seconds, before Z3 settled it in milliseconds. So the order is probe, Z3,
brute force: what Z3 proves or seeds is done, and what it answers `unknown`
to — the transcendental regime this plan is about — gets the budget. Tests
that leave `Strategy::Solver` out go straight from the probe to brute force,
which is how the rungs above still measure sampling alone. The budget is
minutes on an unoptimised tape, which is why the pool tests run with
`common::PROPOSAL_BUDGET`, a million under debug.

Also worth its own verdict while in there: **found one point, cannot find a
second**. The degenerate feasible set — a fully determined equality system,
an ulp-width band, a repaired witness sitting exactly on a boundary. The
walker's chord through such a point has zero length, every proposal fails,
and today it looks like exhaustion.

---

## Shelved, with reasons

- **Interval constraint propagation** over the AST (forward interval
  evaluation, backward projection through each operator, contract the box
  before sampling). It nests fine and handles `sin` and `ln` where Z3 refuses,
  but it is dReal's core algorithm with dReal's failure modes — the dependency
  problem, multi-branch preimages — and it is solver-side work. `todo.md`
  material.
- **Table-interpolated transcendentals.** Slower than the SFU on GPU, a gather
  on CPU. Polynomials if anything.
- **fp16 on the GPU.** Three significant digits; a residual on `x > 10.5` over
  `10..11` carries no ranking information at that precision.
- **`+/- 0.0` and negative tolerances as compile errors.** Done 2026-09-05:
  `ProblemKind::DegenerateTolerance`, judged on the value in the parser, so
  `0`, `0.0`, `-0.0`, `0.0e1` and every negative are one rule; a non-finite
  tolerance gets the same `NonFiniteConstant` the folding pass gives
  everywhere else. Two users typed `x1 == f(x2) +/- 0.0` and got an ulp
  game. A tiny *positive* tolerance below the ulp of the values compared is
  the same pathology and is not catchable at parse time; that is the
  "degenerate region" verdict still under discussion.

## The GPU landscape, so it need not be re-shopped

Surveyed 2026-09-01. wgpu is mature and vendor-neutral; CubeCL JITs `#[cube]`
Rust to CUDA/HIP/WGSL/SPIR-V/CPU-SIMD but is built for compile-time kernels
and admits rough edges; rust-gpu moved to the Rust-GPU org and rust-cuda is
active, both nightly and ahead-of-time; Mojo now targets AMD RDNA3/4 and
NVIDIA with an open-source GPU toolchain but is a language, not a library;
SYCL/oneAPI (Khronos's post-OpenCL answer; DPC++ and AdaptiveCpp) is C++
single-source, ahead-of-time, adopted in HPC and on Intel, marginal elsewhere.
OpenCL is shipped by everyone and started by nobody. None of the clever ones
solve *runtime codegen from a user expression*; shader text does.
