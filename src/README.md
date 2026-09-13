# One AST, two backends

Sojourn has two consumers with genuinely different needs, and the crate is laid out
to say so. In the middle is an `Ast`; on either side is a backend that takes it
somewhere.

```
                                   +-- eval --> CompiledExpression   runs a batch
   source -->  frontend  -->  Ast -+
                                   +-- cvg  --> FeasibleRegion       searches for points
```

**`frontend` lowers as little as it can.** Everything it does is
meaning-preserving: it produces the canonical form of what the author wrote and
nothing more. A pass that makes the tree easier to *analyse* belongs there.

**`eval` lowers as hard as it can**, in the name of speed. It flattens the tree
to a three-address tape (`eval/tape.rs`), packs the temporaries into registers,
and runs it a tile of 256 samples at a time, each instruction one kernel across
the lanes. The kernels (`eval/simd.rs`) are explicit SIMD through `pulp`, with
AVX2 chosen at run time and a scalar backend otherwise; the operators that have
no vector form — libm, `%`, `pow`, rounding — run in kernels named `*_scalar`,
so a loop that is not vectorised says so in the code rather than being left to
the compiler. The type is opaque, so what runs the tape can change without
an API change.

**`cvg` keeps the tree open**, because its whole job is reading structure: which
constraints a solver can be asked about, which variables another determines,
which comparison can be inverted into a bound.

neither backend's lowering is visible to the other**. 

## The front end

Four passes. `parse` in [`frontend/mod.rs`](frontend/mod.rs) is the whole pipeline
and reads top to bottom. It is crate-private: a caller hands source text to
`compile` or to `ConstraintSystem::new`, and the tree between is nobody's
business but the two backends'.

```
              fold_constants   invert_monotone   unroll_aggregates
 source ─►  ──────────────►  ──────────────►  ──────────────►  Ast
      translate (fallible)                        (fallible)
```

The output is an `Ast`, not an evaluable thing: turning one into something that
runs is `eval::bind`'s job (`compile` is parse then bind), and one directory over
is `cvg`, which never lowers it at all.

| phase | entry point | what it does |
|---|---|---|
| parse & lower | `frontend::translate` | ANTLR parse tree to `ast::Program`; resolves names to `GlobalId`/`LocalSlot`, records `is_constraint` |
| fold constants | `rewrite::fold_constants` | every subtree made only of literals becomes one `Kind::Literal` |
| invert monotone | `rewrite::invert_monotone` | `f(u) op c` becomes `u op' c'` for the strictly monotone `f` |
| unroll aggregates | `rewrite::unroll_aggregates` | every `Kind::Aggregate` becomes `Kind::Fold`; a bound that is not a constant, or a span past the cap, is a compile error |

The order is not arbitrary. Folding runs first because it makes *"is this
constant?"* stop being a question anywhere else — afterwards a statically known
value **is** a `Kind::Literal`, which is why inversion and unrolling can both
pattern-match instead of carrying evaluators of their own. Inversion has to see
`Kind::Compare`, which it does, because nothing eliminates one any more.

One rewrite that used to be here is deliberately not: `x ^ n` for a whole `n`
reaches every backend as written. The tape, the shader rendered from it and the
SMT emitter each lower it to repeated multiplication themselves, keyed on one
rule (`Expr::whole_exponent`), and interval narrowing inverts the node through
its root. Expanding it in the tree cost a compound base `n` evaluations and left
a product fold nothing could invert.

Two passes are fallible, and both refuse rather than defer:

- `fold_constants` rejects a constant subexpression that works out to NaN or an
  infinity — `sqrt(-1)`, `1/0`, and the literal `1.0e400`, which babel's grammar
  admits and `f64` cannot hold.
- `unroll_aggregates` rejects a bound that is not a constant, one that is a
  constant but not a usable index (`sum(1, 20/3, …)`), and an aggregate wider
  than its cap. `sum` is big-sigma over a fixed index set, not a loop; after
  this pass no aggregate exists, and neither backend knows one ever did.

## Nothing non-finite travels

One rule, enforced wherever it can be seen:

| phase   | where                                                                                          | catches                                                    |
|---------|------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| compile | `rewrite::fold_constants` → `ProblemKind::NonFiniteConstant`                                   | what is provable: `sqrt(-1)`, `1/0`, `1.0e400`             |
| runtime | every checked instruction in `eval/tile.rs` and `eval/lane.rs` → `ProblemKind::NonFiniteValue` | the rest: `ln(x)` at `x = 0`, overflow, a non-finite input |

The runtime check is on **every instruction**, not only the operations that can
produce a non-finite value, so the error names the innermost subexpression that
went wrong rather than the whole constraint — instruction order is post-order,
so the first faulting instruction is the innermost node. It also catches a
non-finite *input* at its `Load`, and an unwritten local: the registers are
primed with NaN as a sentinel and a local the lowerer cannot prove assigned gets
an explicit `Check`, so reading one is a slot-allocation bug rather than a value.

Infinities are included deliberately, and `rewrite::monotone` is why: while
`ln(0)` was allowed to evaluate to `-inf`, zero satisfied *any* upper bound, and
the inversion pass had to carry a domain floor of `u >= 0` where the mathematics
asks for `u > 0`. Refusing the infinity is what lets the guards be the textbook
ones. `sqrt` keeps its inclusive floor, and the asymmetry is now principled:
`sqrt(0)` is a finite answer, `ln(0)` is not an answer.

**Eager per lane, per instruction.** This is the CPU evaluator's contract, not
the language's. The batched executor fuses the finite test into each
instruction's loop as an or-reduction, so the happy path pays nothing, and it
records each lane's first fault rather than stopping; the lowest faulted column
is reported at the end with the innermost span, exactly as the tree-walker it
replaced did. A future GPU sieve is the one place this is allowed to be
*coarse*: check the output buffer, re-run an offending column through this
evaluator for the span, and do not "fix" the kernel to match.

The policy is the `is_finite` test on every checked instruction in
`eval/tile.rs` and `eval/lane.rs`. If a real use for a saturating infinity turns
up, that is where it changes.

## The one type

`ast::Kind` is a single enum spanning every phase. That is deliberate: it keeps
every pass a composable `Program -> Program`, which is what makes the rewriter
pluggable. A separate post-rewrite type would make every pass change types, and
each new pass would need converting on both sides.

`Kind::Compare`, `Kind::NearEq` and `Kind::And` reach both backends intact, and
each lowers them its own way.

## The boolean convention belongs to `eval`

Babel has no boolean values at run time, so **the evaluator** turns a comparison
into arithmetic whose *sign* carries the truth value: `<= 0` is true. A violated
constraint then reports how badly it was violated rather than merely that it
was, which is the canonical `g(x) <= 0` form an optimizer wants.

Strictness rides on a nudge: `a < b` evaluates as `(a - b) + ε` with ε being
`f64::MIN_POSITIVE`, which vanishes into rounding at any real magnitude and
survives only when the difference is exactly zero — precisely where strict and
non-strict differ.

**This is one backend's convention, not the language's.** `cvg::smtlib` shares
none of it: a comparison is emitted as `(> x 5.0)`, an equality as two bounds
`and`-ed together. It used to receive `(< (- 5.0 x) 0.0)` and have to *detect* a
three-hundred-digit denormal to recover the strictness, and an equality arrived
as `(<= (expr_max …) 0.0)` — an `ite` where a conjunction was meant. Both are
gone with the pass that caused them.

`Kind::And` exists for the same reason. `invert_monotone` needs a conjunction
for its domain guard — `ln(x) < 2` means `x < e²` **and** `x > 0` — and used to
build `max(residual, residual) <= 0` by hand, which is the residual convention
leaking into the front end. A variant it can emit without knowing costs one arm
per backend.

## What the evaluator is held to

`tests/corpus.rs` (every construct at an ordinary input, exact unless it routes
through libm), `tests/runtime_errors.rs` (where a fault lands) and
`tests/special_values.rs` (signed zeros to the bit, operations that go
non-finite, every fault kind planted in a batch). Hand-written expectations
only. The tree-walking evaluator that preceded the tape was used once as a
differential oracle over a few thousand random rows and then deleted; a
recorded-output file was considered and rejected, because once the walker is
gone such a file is only the tape agreeing with itself.

## Who consumes the result

- [`eval/`](eval) — `compile(&str, &[names])`, then
  `CompiledExpression::eval(MatRef)`. **One column per sample, one row per schema
  variable**, which is the shape `cvg` produces, so a generated batch is directly
  an input matrix with no transpose. There is no scalar entry point in the public
  API: the crate-internal `eval_row` exists because the walker is sequential by
  nature, and it is the same tape through the per-lane executor, not a second
  implementation.
- [`cvg/smtlib.rs`](cvg/smtlib.rs) — renders constraints as SMT-LIB2 for a solver.
  What it cannot express it *reports* through `Document::untranslated` rather
  than dropping, which is most of why the two passes above exist: every
  constraint they rewrite is one the solver can then see.

## Files

| | |
|---|---|
| [`ast.rs`](ast.rs) | `Program`, `Block`, `Expr`, `Kind`, and the operator semantics in `UnaryOp::apply` / `BinaryOp::apply` |
| [`frontend/`](frontend) | text to `Ast`: `parse` and the `Ast` type in `mod.rs`, `parse.rs`, the `rewrite.rs` passes, the ANTLR output |
| [`eval/`](eval) | `compile`, the tape (`tape.rs`, `lower.rs`, `regalloc.rs`), its two CPU executors (`tile.rs`, `lane.rs`), and `wgsl.rs`, which turns a tape into the view that `templates/wgsl/` renders as a WGSL function for the GPU sieve |
| [`diagnostics.rs`](diagnostics.rs) | `ProblemKind`, spans, and rendering |
| [`generated.rs`](generated.rs) | ANTLR output, not hand-edited |
| [`../templates/wgsl/`](../templates/wgsl) | the WGSL, as askama templates: `operators.wgsl.jinja` (one macro arm per babel operator and its domain guard), `function.wgsl.jinja` (a tape as a function), `prelude.wgsl.jinja`, `harness.wgsl.jinja` (the sieve's entry points and bindings) |
| [`../templates/smt2/`](../templates/smt2) | the SMT-LIB2, as askama templates: `operators.smt2.jinja` (the operator table), `term.smt2.jinja` (one term, children already rendered), `condition.smt2.jinja` (divisor guards and root auxiliaries), `prelude.smt2.jinja` (the `babel_*` helpers), `document.smt2.jinja` (the whole document, owning every newline) |
| [`system.rs`](system.rs), [`solve.rs`](solve.rs), [`repair.rs`](repair.rs) | the generator's API: a validated set of constraints over a box, which answers whether a point is feasible and nothing harder; the solver builder and the solved region it returns, which hands out samples and repairs a point against its system; the repair algorithm |
| [`cvg/`](cvg) | the search engine, private: `progress.rs` (what the search has in hand, as a value), `sampling.rs` (probe, deliver, brute force), `walking.rs` (hit-and-run), `classify.rs`/`interval.rs`/`incidence.rs` (reading the constraints' structure), `smtlib.rs`/`smt.rs` (SMT-LIB2 and Z3), `sieve.rs` (the GPU sieve, behind the `gpu` feature); `mod.rs` holds the ladder and the worker |
