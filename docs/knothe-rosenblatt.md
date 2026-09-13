# The triangular decoder — mapping the nominal cube onto the region

*Plan, written 2026-09-12, not yet started. Named by the idea as `docs/brute-squad.md` is. The
reasoning and the measurements go into `docs/todo.md` part two, beside "Repairing a point rather
than discarding it", as the work lands.*

## Context

Artemis consumes `FeasibleRegion::repair` as its whole constraint mechanism: every point it
proposes that fails the oracle comes back for a projection, thousands of times a run, each up
to eight clamp sweeps and eight sixty-bit chords, and the optimizer learns nothing about the
region's shape from any of them. It works and it performs badly.

The alternative is a **decoder** (Michalewicz & Schoenauer 1996's taxonomy: penalty, repair,
decoder, feasibility-preserving operators): a map from a cube onto the region, so what the
optimizer searches over is feasible by construction wherever the map is exact. The
construction is the **Knothe–Rosenblatt rearrangement** (Rosenblatt 1952, Knothe 1957), a
triangular map: fix a variable order; map the first cube coordinate onto the first variable's
range; the second onto the second's range *given* the first; and so on. Each step is one
dimension, the inverse is the same loop read backwards, and the conditional range is what
`interval::slice_conditioned` already computes. The textbook version uses conditional
quantiles and preserves measure; this one maps each coordinate linearly onto an interval
enclosure, giving up uniformity (Artemis draws its initial design from `take`, which is
uniform, and encodes it) and keeping totality, determinism and a cheap inverse.

The radial alternative (Koziel & Michalewicz 1999, in the roadmap under "Literature") stays in
reserve: it reads only the oracle, so transcendentals cost it nothing, but it cannot reach
parts of a region invisible from its centre, is hopeless on slabs, and warps on tubes. The
triangular map fails by landing outside the region where narrowing is loose — the one failure
that is visible and recoverable, by the repair that exists.

**Outcome:** a public `KnotheRosenblattTransformer` built from a solved region; an opaque
`Nominal` so a cube point can never reach the oracle or the simulation by accident; one engine
change in `interval.rs`; a decoder-efficiency number per system that says how much of the cube
lands feasible. Composition with repair stays with the caller.

## The shape, as agreed

- **The nominal space is the signed unit cube** `[-1, 1]^d`, one entry per user variable.
  Artemis's cube point *is* the `Nominal`, bitwise; no affine layer sits in front of decode or
  behind encode. Sojourn learns only that the nominal space is the cube — a convention of the
  transformer, not knowledge of Artemis. `InputVariable` bounds stay in user units everywhere.
- **`decode(&Nominal) -> Point`**: always inside the box; feasible wherever narrowing is exact.
  Driven coordinates ignore their nominal entry and sit at the middle of their band. Entries
  outside `[-1, 1]` are clamped, so the map is total.
- **`encode(&[f64]) -> Nominal`**: the inverse on the region. Contract: the input is feasible.
  Driven entries encode as `0.0`. Not clamped: at clearance zero every feasible point encodes
  inside the cube (a theorem — the true slice is inside the enclosure), and an infeasible one
  can land outside, which is the signal. At positive clearance a feasible point within a step
  of a wall encodes just past `±1`.
- **Not a projection.** Repair is the identity on the feasible set; decode moves feasible
  points too, since their coordinates are read as slice positions. A separate type, not a
  mode of repair.
- **`Nominal` is opaque**: a newtype over `Vec<f64>` with `new` (asserts every entry finite),
  `coordinates()`, `len`, `is_empty`; deliberately no `Deref`, `AsRef<[f64]>`, or
  `From<Nominal> for Vec<f64>`. The one mistake the compiler can catch is a nominal reaching
  something that takes user-space points, and those omissions are what catch it. The image
  side stays `Point`, which already means "user space" throughout the crate.
- **Own type, not `FeasibleRegion`.** `KnotheRosenblattTransformer::new(&FeasibleRegion,
  clearance)` clones the system out of the region (the region holds a clone; a system is tapes
  and two small graphs) and is independent afterwards. Built from the region so "it was
  solved" is the precondition, as for repair. `solve.rs` is untouched. No trait over decoders
  until the radial one exists.

## The engine change: unsettled coordinates are conditioned on their box

`interval::slice_conditioned` (src/cvg/interval.rs:512) skips any constraint that names an
unsettled coordinate (the `continue` at :532-538). Skipping contributes `ENTIRE`. Conditioning
on the coordinate's **declared box** instead is sound — its eventual value lies in the box, and
`narrow` (:593, HC4-revise) takes an interval for every symbol and returns a superset — and at
least as tight as skipping. `retract` and `settle` gain from it: a driven coordinate is no
longer drawn where a later driven coordinate's band would be empty. The skip was added against
conditioning on a *stale value* (todo.md:292-307); the box is not stale.

Refactor the body into one general function and make the entry points thin:

```rust
/// The interval `coordinate` may take with every row at the interval `rows` gives it, and
/// `rows[coordinate]` as the starting interval. A superset, as everything here is.
pub(crate) fn slice_over(system: &ConstraintSystem, rows: &[Interval], coordinate: usize) -> Interval
```

`slice` = every row a point, `rows[i]` the declared box. `slice_conditioned` = settled rows
points, unsettled rows their declared box. The general form exists now because the escalation
below needs to pass narrowed intervals, not boxes. Docs to rewrite with it: the function's own
(:512-519, "unsettled coordinates are conditioned on their box, not their stale value"),
`slice`'s "nothing here iterates to a fixpoint" (:504-507, qualify: without a mask), the
`retract`/`settle` comments (classify.rs:170-183, 203-210), the bold paragraph at
AGENTS.md:221-227, and a dated italic amendment at todo.md:303-305 in the house style.

No test pins the mask today. Add `interval::tests::an_unsettled_coordinate_is_conditioned_on_its_box`:
`x ∈ [0,1]`, `y ∈ [0,10]`, `z ∈ [0,1]`, `y == x + z +/- 0.01`, point `[0.5, 7.0, 0.9]`, mask
`[true, false, false]`, slice of `y`: was the box `[0, 10]`, becomes `[0.49, 1.51]` within
`1e-12`; with `None` it is `[1.39, 1.41]`, unchanged.

Regression coverage for the changed semantics, all existing: every driven case in
`tests/cvg_equalities.rs` (`two_coupled_equalities_are_traversed` is the case the skip was
introduced for), `cvg_pools` (`roots`, `power`, `the_same_seed_delivers_the_same_points`),
`cvg_benchmarks::top_corner_200d_as_equalities`, `regression_fixture`, and
`cvg_repair::a_driven_coordinate_is_not_privileged`. No seeded verdict is expected to move: a
verdict can move only where a constraint names two driven coordinates and the box of the
unsettled one tightens the wanted one beyond its other constraints, and the coupled fixtures'
boxes do not.

## The algorithm

Construction fixes the system, the `clearance`, the **order** (the chosen free order, then the
driven coordinates in plan order), the `free` list, and the efficiency. `new` asserts every
bound finite and `lower <= upper` (`InputVariable` is unvalidated and an infinite width makes
the step `inf` or `NaN`), the clearance finite and non-negative, and `nominal.len() ==
variables.len()` on every call, in the style of repair.rs:193-207.

`decode`, per coordinate `i` in order, with `settled` marking what is placed:

1. `slice = interval::slice_conditioned(system, &x, i, Some(&settled))`.
2. Shrink by `step_i = clearance * width_i` at each end. Where `lo + step > hi - step`, aim for
   the middle (repair's rule at repair.rs:397-403). Where the slice itself is empty — the
   settled prefix admits no completion the enclosure can see, or a driven band's dead arm —
   fall back to the box; the point will miss and the oracle will say so, the same silence as
   `retract`.
3. Free `i`: `x[i] = lo' + (u[i].clamp(-1, 1) + 1) / 2 * (hi' - lo')`, then `.clamp(lo', hi')`
   so "inside the box" is a guarantee and not a rounding accident. Driven `i`: the middle.
   Zero-width: `lo'`.

`encode` is the same loop with step 3 read backwards: `u[i] = 2 (x[i] - lo') / (hi' - lo') - 1`
for a free coordinate, `0.0` for a driven one or a zero-width slice. Round trip: decode
produces the prefix values encode conditions on, bitwise, so `encode(decode(u)) == u` to the
affine rounding. `encode_columns(MatRef) -> Vec<Nominal>` reads columns as repair.rs:251 does.

One private method `room(&self, point, settled, i) -> (f64, f64)` carries steps 1–2 for both
loops. Nothing else private for the map; the affine step is inlined in each loop.

**What clearance does and does not guarantee.** Narrowing is existential — the slice holds
every `x[i]` for which *some* value of the other rows satisfies the constraint — so the shrunk
slice is where to aim and never a proof, exactly as in repair's `clamped`. The oracle's
clearance perturbs *every* variable a constraint names; the shrink moves this coordinate's
edge by its own step and knows no coefficient. On `2*x1 + x2 < 1` at `1e-3` the corner
`decode([1, 1]) = (1.496, -1.996)` passes the plain oracle and fails `x1 + 0.004`; about a
third of a percent of that image lacks the clearance. Not fixed by shrinking more: the fix is
the robust counterpart of each constraint, which HC4 cannot express, and the caller's repair
from a feasible point is the cheap "step off the walls" path. Consequence for tests: exact
`efficiency() == 1.0` only at clearance zero; floors at positive clearance.

**Faces and walls.** At clearance zero a slice's edge sits a hair *outside* a strict wall (the
enclosure is padded outward), so `u = ±1` lands on or past it and a strict comparison fails
there. That is the cube's faces mapping to the walls, and a positive clearance is what keeps
the image off them; Artemis never runs at zero. Pin it as a deliberate `!is_feasible` at the
corner rather than paper over it.

## Order selection

Three candidates, scored, best kept, ties to the earliest:

- **(a)** schema order; **(b)** reverse schema order;
- **(c) readability precedence.** A coordinate is *readable* through a constraint when every
  occurrence of it lies on a path from the root that `narrow`'s backward pass can invert:
  `Compare`/`NearEq`/`And` at the root, the `invertible_unary` / `invertible_binary` tables
  (interval.rs:772/788), and a whole-exponent power through its base — the same knowledge
  `classify::reaches` (classify.rs:330) uses for one occurrence. Add
  `pub(crate) fn readable(constraint: &Ast, variable: GlobalId) -> bool` beside `reaches`,
  descending into every child that mentions the variable and answering false the moment the
  variable sits under an operator narrowing declines. Structural, not "narrow at the box and
  compare": that test calls `x2` in `x2 > cos(x3)^2` unreadable whenever `x2`'s box already
  starts at zero, and calls every coordinate of a linear inequality over a wide box
  unreadable. For each constraint, every unreadable coordinate precedes every readable one, so
  `cos(x3)^2 < x2` places `x3` first and the cosine is a constant by the time `x2` is narrowed.
  Kahn's algorithm over the free coordinates, lowest schema index first among the ready, a
  cycle broken by taking the lowest unplaced index. On a purely linear system there are no
  edges and (c) equals (a); harmless, and say so in the doc so nobody deduplicates.
  Driven coordinates always last, in plan order.

**Efficiency**: `ORDER_TRIALS = 256` nominals uniform in `[-1, 1)` from
`Xoshiro256PlusPlus::seed_from_u64(ORDER_SEED)` via `cvg::sampling::fill_box`, decoded, the
fraction passing `is_feasible(x, clearance)`. A fixed seed rather than the region's: the order
is a property of the system and the clearance, and two transformers over the same region must
agree bit for bit whatever seeded the census. The winning score is `efficiency()`, the
instrument for everything below. `new` builds the struct once with schema order and reassigns
`order` per candidate, so there is one decode implementation and no extra system clones.

Cost at construction: `3 × 256 × d` slices and `768` oracle calls; per decode, `d` slices and
no oracle call. A slice costs the constraints naming the coordinate.

## What is deliberately not here

- **No fixpoint over the unsettled rows, in the first cut.** The escalation, which P118 is
  expected to force (see its test), is hull consistency proper: before reading the wanted
  slice, sweep every unsettled row through `slice_over` with the current rows, replace its
  interval by the result, repeat until no row narrows by more than a sliver of its width or a
  small fixed number of sweeps is spent, then read the wanted slice against the narrowed rows.
  Same superset invariant, same `narrow`, no new type; `O(d²)` slices per decode at worst,
  fine at fifteen variables, to be measured at two hundred where the chains that need it do
  not occur.
- **No union-of-intervals.** A slice that is really two bands is answered by its hull and the
  gap is where the map misses. Threading a disjoint-union type through the two thousand lines
  of `interval.rs` is the one expensive item in reach of this design: the decision point, not
  the starting point.
- No quantile weighting from the census: the map is not uniform and does not need to be.
- No composition with repair inside the transformer: the caller checks and repairs, so the
  miss rate stays observable. No bijection claim.
- No adaptive order or clearance: any change to the map mid-run turns every row in Artemis's
  table into a point it can no longer locate. The order is not exposed.
- No consulting of `affected` over `naming`: a constraint with a computed subscript is not
  seen when narrowing a coordinate it does not statically name. Sound (fewer constraints is a
  wider superset) and no loss (such a constraint narrows nothing); it is decoded blind and
  repaired.

## Files

| file | change |
|---|---|
| `src/decoder.rs` (new) | `Nominal`; `KnotheRosenblattTransformer` (`system`, `order`, `driven: Vec<bool>`, `free`, `clearance`, `efficiency`); `new`, `decode`, `encode`, `encode_columns`, `free`, `efficiency`, `clearance`; private `room`, `score`, and the free fn `readability_order`. Module doc in the register of `repair.rs`: what it is for, the construction, which order and how chosen, what it guarantees and does not, what is deliberately not here, pointer to `docs/todo.md`. |
| `src/cvg/interval.rs` | `slice_over`; `slice`/`slice_conditioned` as wrappers; docs; the mask unit test. |
| `src/cvg/classify.rs` | `pub(crate) fn readable` with unit tests beside `reaches`'s; `retract`/`settle` comments. |
| `src/lib.rs` | `mod decoder;` (alphabetical), `pub use decoder::{KnotheRosenblattTransformer, Nominal};` — public surface, not hidden. |
| `tests/cvg_decoder.rs` (new) | the fixtures below, own five-line helpers (`system`, `region`, `variables`, `nominal`), `SEED` and `CLEARANCE = 1e-3` as in `cvg_repair.rs`. |
| `tests/cvg_benchmarks.rs` | extract `fn p118_problem() -> Problem` so the twenty-nine constraints stay transcribed once; `p118()` calls `run(p118_problem())`; add `p118_decodes_efficiently`. |
| `docs/knothe-rosenblatt.md` (new) | this document, in the structure of `brute-squad.md`. |
| `docs/todo.md` | part one: one item beside the repair items (~:364) pointing at the doc; part two: `## The nominal cube: a triangular decoder for Artemis` before "Where CVG stands" with the reasoning, the measurements table as they land, the `slice_conditioned` amendment, and the work items below; the dated amendment at :303-305. |
| `src/README.md`, `README.md`, `AGENTS.md` | files-table row (:188) gains `decoder.rs`; one sentence in the crate README after the `FeasibleRegion` sentence; AGENTS: reading-order item 5 for the doc, the layout row for the API files, the :221-227 rewrite, and a short bold paragraph after :259: "the decoder is a coordinate chart, not a sampler". |

## Tests — `tests/cvg_decoder.rs`

Hand-computed expectations; `1e-12` wherever a constraint narrows the coordinate, because the
enclosure is padded by a few ulps on purpose (say so in a comment); exact only where nothing
narrows. The judge is `ConstraintSystem::is_feasible`. Derivations follow `narrow` step by step.

| test | fixture | expectation |
|---|---|---|
| `a_half_space_is_decoded_in_schema_order` | `2*x1 + x2 < 1`, `[-2,2]²`, clearance 0 | both readable, no edges: schema. `x1` with `x2` at box: `2*x1 ≤ 3` → `[-2, 1.5]`; `x2` given `x1 = p`: `[-2, min(2, 1 - 2p)]`. `free() == [0, 1]`; `decode([0,0]) = (-0.25, -0.25)`; `decode([0.5, -0.5]) = (0.625, -1.5625)`; round trip within `1e-12`; `efficiency() == 1.0`. The `(0,0)` value pins the tie rule: reverse order is exact too and would answer `(-0.75, 0)`. |
| `a_cube_face_lands_on_the_wall` | same | clearance 0: `decode([1,1])` in box, `!is_feasible(.., 0.0)`. `CLEARANCE`: `decode([1,1]) = (1.496, -1.996)` within `1e-12`, feasible plain, `!is_feasible(.., CLEARANCE)`, and `region.repair(Mat::zeros(2,0), &x, CLEARANCE)` is `Ok` — the composition the caller does. |
| `a_disc_is_decoded_by_conditioning_on_the_first_coordinate` | `sqr(x)+sqr(y) < 1` and `x^2 + y^2 < 1`, `[-2,2]²`, clearance 0 | `x` with `y` at box: `sqr(x) ∈ [0,1]`, both branches live → `[-1, 1]`; `y` given `x = p`: `±sqrt(1 - p²)`. `decode([0,0]) = (0,0)`; `decode([0.6, 0.5]) = (0.6, 0.4)` within `1e-12`; round trip; `efficiency() == 1.0`; identical numbers for both spellings. |
| `a_driven_coordinate_takes_the_middle_of_its_band` | **`x2 == 3 - 2*x1 +/- 0.001`**, `[-5,5]²`, clearance 0 — spelled with the bare side, because `2*x1 + x2 == 3` drives `x1` (shape walks symbols in order and `x1` is reachable through `Add` then `Mul`; cvg_repair.rs:234's comment is narrative) | `free() == [0]`; `x1` with `x2` at box: `[-1.0005, 4.0005]`; `decode([0.5, 0.9]) = (2.75025, -2.5005)` within `1e-12`; `decode([0.5, -0.9])` bitwise equal (driven entry ignored); `encode` gives `(0.5, 0)`; `efficiency() == 1.0`. Same fixture at `CLEARANCE`: the step `0.01` exceeds the half-width, `efficiency() == 0.0` — one assertion. |
| `two_hundred_bounds_have_a_closed_form` | `xi > 10.5`, `[10,11]^200`, `CLEARANCE` | shrunk slice `[10.501, 10.999]`; `x_i = 10.501 + (u_i + 1)/2 · 0.498` within `1e-12` for `u_i = -1 + 2i/199`; round trip; `free().len() == 200`; `efficiency() == 1.0` (one coordinate per constraint, so the clearance argument above does not bite). |
| `an_unconstrained_box_is_the_affine_denormalise` | two variables, no constraints (`new` accepts an empty list; the probe returns `Satisfied`) | clearance 0: `decode([-0.5, 0.25]) == vec![-1.0, 6.25]` and `decode([0,0]) == vec![0.0, 5.0]` **exactly**; `CLEARANCE`: `decode([0,0])` within `1e-15`; `free() == [0, 1]`; `efficiency() == 1.0`. |
| `an_out_of_cube_nominal_is_clamped` | the unconstrained box | `decode([3.0, -7.0]) == decode([1.0, -1.0]) == vec![2.0, 0.0]`, exact. |
| `an_order_the_schema_gets_wrong_is_found` | `x1 ∈ [0,1]` uncoupled, `x2 ∈ [0,2]`, `x3 ∈ [0,3]`, `x2 > cos(x3)^2` | schema places `x2` on its box then `x3` on its box (`cos` declines) and misses about a quarter of the cube; readability places `x3` before `x2`. `efficiency() >= 0.99`; 64 seeded decodes all feasible. |
| `a_fan_needs_the_readability_order` | `x1 ∈ [-1,2]`, `x2 ∈ [0,3]`, `x3 ∈ [-1,2]`, `x1 > cos(x2)^2`, `x3 > cos(x2)^2` | edges `x2 → x1`, `x2 → x3`; neither schema nor reverse satisfies both; Kahn gives `(x2, x1, x3)`. `efficiency() >= 0.99` where the other two candidates sit near `0.76`. Goes red if the precedence sort is deleted. |
| `a_gap_landing_is_infeasible_and_repair_recovers_it` | `(x + 2) * (x - 1) == 0 +/- 0.001`, `[-5,5]`, clearance 0 | `x` occurs twice, nothing driven; the product declines (both factors straddle zero) so the slice is the box. `decode([0]) == vec![0.0]` exactly; `!is_feasible`; `efficiency() < 0.05`; `region.repair(anchors [-2],[1], &[0.0], 0.0)` within `0.001` of `1.0`. Documents the known miss. |
| `decode_and_encode_hold_their_contract_over_a_polytope` (`#[pollster::test]`) | the five-variable polytope of `cvg_repair.rs:377-387`; 64 nominals from `Xoshiro256PlusPlus::seed_from_u64(SEED)`; transformers at clearance 0 and at `CLEARANCE` | per trial: in box; `encode(decode(u))` within `1e-12` on free entries; `decode` twice bitwise equal; at `CLEARANCE`, `is_feasible(x, CLEARANCE) \|\| region.repair(anchors, &x, CLEARANCE).is_ok()`. Once, at clearance 0: `encode_columns(region.take(256))` every entry in `[-1, 1]`; `efficiency() == 1.0`. |
| `p118_decodes_efficiently` (in `cvg_benchmarks.rs`) | `p118_problem()`, solved with `with_known_feasible(seeds)` so the opening is immediate; transformers at clearance 0 and `1e-6` | **measure first.** First commit asserts `efficiency() > 0.0`, prints both, and asserts every miss among 256 seeded decodes repairs `Ok` against `take(256)`. After one `--no-capture` run, pin `>= floor` at roughly `0.8 ×` the measurement and record number, date and host in `docs/todo.md`. **Expect it low on one pass**: hand-propagating the tiers, `x4` is placed against `x7..x9` at their boxes, so nothing stops `x4 ≈ 0` when `x1 = 0`, and three tiers down `x7 + x8 + x9 > 70` is unreachable under the bands — `x9`'s slice comes back empty. The fixpoint escalation is what this test is expected to force. |

Unit tests: the mask test in `interval.rs` (above); `readable` in `classify.rs`: `x2 == 3 - 2*x1`
reads both; `x2 > cos(x3)^2` reads `x2` not `x3`; `x1 + sin(x1) < 2` does not read `x1`
(one occurrence under `sin`, the other on an invertible path); `x^2 + y^2 < 1` reads both.

Non-goals: a compile-fail test that `Nominal` lacks `Deref` (no `trybuild`; a review
property); a timing fixture (a debug-mode number is meaningless).

## Steps, each with the test that goes green

1. Copy this plan to `docs/knothe-rosenblatt.md`; the part-one item, the part-two section
   with the work items, the README/AGENTS pointers.
2. `interval::slice_over`; the wrappers; docs; the mask unit test; the `retract`/`settle`
   comments and the AGENTS.md :221-227 rewrite. *Green:* the mask test; no movement in
   `cvg_equalities`, `cvg_pools`, `cvg_repair`, `regression_fixture`.
3. `classify::readable` and its unit tests.
4. `src/decoder.rs`: `Nominal`; the transformer with schema order only and `efficiency`;
   `lib.rs`. Write the closed-form tests first, red, then green; `just test-compile` green
   before anything else compiles against it.
5. Order candidates, `readability_order`, `score`. *Green:* the two order fixtures; the
   half-space `(0,0)` value still holds (tie rule).
6. `encode`, `encode_columns`. *Green:* the polytope property, the gap fixture.
7. `p118_problem()`, `p118_decodes_efficiently`: measure, record, pin.
8. `just fmt-check`, `just lint`, `just test`; the part-two section gets the numbers.

## Work items, in the roadmap's format

- [ ] **`slice_over`, and boxes for the unsettled.** `Driven by:`
      `interval::tests::an_unsettled_coordinate_is_conditioned_on_its_box`; the driven-equality
      suite unchanged.
- [ ] **`classify::readable`.** `Driven by:` unit tests beside `reaches`.
- [ ] **`Nominal` and `KnotheRosenblattTransformer`.** `Driven by:` the closed forms in
      `tests/cvg_decoder.rs`.
- [ ] **Order selection by measured efficiency.** `Driven by:`
      `an_order_the_schema_gets_wrong_is_found`, `a_fan_needs_the_readability_order`.
- [ ] **Encode and the census.** `Driven by:` the polytope property test.
- [ ] **P118 efficiency, measured then pinned.** `Driven by:`
      `cvg_benchmarks::p118_decodes_efficiently`, floor set after the first measurement.
- [ ] **The composition with the oracle and `repair` is Artemis's.** `Driven by:` nothing
      here; the seam is documented, not tested.

## Verification

```
just fmt-check
just lint
just test-compile
cargo nextest run --lib -E 'test(interval) | test(classify)'
cargo nextest run --test cvg_decoder
cargo nextest run -E 'binary(cvg_equalities) | binary(cvg_pools) | binary(cvg_repair) | binary(regression_fixture)'
cargo nextest run -E 'test(p118_decodes_efficiently)' --no-capture
just test
```

Clippy will want `#[must_use]` on the accessors, `is_empty` beside `len`, `f64::from` on a
`u32` hit counter rather than a cast, and backticks in docs. `cvg_benchmarks` legitimately
takes minutes; `.config/nextest.toml` gives it 300 s.

## Stopping rule

Ship at step 8 whatever P118 says; the efficiency number is the deliverable as much as the
map. If the polytopes come back near perfect and only the band fixtures are poor, stop —
repair absorbs the bands. If P118 is poor, the fixpoint over unsettled rows is the next item
and its test is what it must move. The union-of-intervals type is the point to stop and
decide, not to start.
