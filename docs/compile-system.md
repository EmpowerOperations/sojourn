# `compile_system` — named expressions as nodes with edges

*2026-09-21. Begun as a feature request from `opyl` against `33f5b10`, titled "`compile_system`
— compile a set of named expressions as one dependency graph". The request asked for a
`CompiledSystem` with a layered runtime; what was agreed and built is narrower, and this
document is the spec for that. The request's own text is kept where it still says the right
thing; what was dropped is listed at the end, with why. The tests are `tests/compile_system.rs`
(the contract) and `tests/dependency_diagram.rs` (a worked example of the intended use).*

## Who is asking, and for what

`opyl` is the parser for OASIS's problem-definition documents. A document declares input
variables with bounds, and outputs that are either computed by a babel expression or produced
by an external simulation. Expressions may reference inputs *and other outputs*, including ones
a simulation produces:

```yaml
inputs:
  x1: { lower: 0.0, upper: 10.0 }
  x2: { lower: 0.0, upper: 10.0 }
objectives:
  f1: x1 + 10             # babel, over inputs
  f2:
    invoke: ansys         # produced externally; a symbol with no expression
  f3: f1 * f2             # babel, over another babel output and an external one
constraints:
  c1: x1 + x2 > 0.5       # "cheap": decidable from inputs alone
  c2: f3 < 100            # "expensive": needs the simulation first
```

`opyl`'s contract is that a document is **valid front to back or it is rejected**, with every
problem reported at once. For the babel portion that means: every expression parses, every
name it uses is declared, nothing references a constraint (booleans are not values), and there
are no cycles. `compile(source, &variables)` gets most of the way per expression, but the
graph-level facts — cycles, self-reference, which outputs are cheap — are invisible to a
per-expression call, and `opyl` would have to reconstruct them from `references()` outside
sojourn. That is the wrong side of the boundary: sojourn is the one that holds the dependency
structure.

## What a per-expression compile cannot see

Probing thirteen expressions against one flat list `[x1, x2, x3, f1, f2, g1, g2, h]` (inputs
then outputs, the only way to make `f1` bindable):

| expression | `compile` says | what is true |
|---|---|---|
| `var[4]` | `references=[f1]` | a literal subscript walked past the inputs and bound to an output |
| `h: h + 1` | compiles | self-reference is just a name that exists |
| `g1: g2 + 1`, `g2: g1 + 1` | both compile | a cycle cannot be seen one expression at a time |
| `c3: c2 * 2 > x1` | `Unbound { c2 }` | correct only because `c2` was left off the list |
| `f2: f1 * 2` | `references=[f1]` | *cheap*, transitively — a per-expression fold says "depends on an output" |

All four are one flat list and one expression at a time. The fix is a call that sees the three
kinds of symbol a document has and every expression together.

## What was built

Sojourn does not schedule. It hands back **one node per expression, each carrying its edges**,
and the caller slots them into whatever graph it runs — OASIS's `GraphModel`, artemis's
whatever-it-decides, or the twenty-line walk in `tests/dependency_diagram.rs`. A node's read
list *is* its "waiting on" list; a topological order over the nodes comes free from the cycle
check and is the order the `Vec` comes back in, for a caller without a graph engine.

```rust
pub fn compile_system(
    inputs: &[&str],                 // the design vector, in order; the only thing var[i] indexes
    externals: &[&str],              // bindable by name, produced elsewhere
    expressions: &[(&str, &str)],    // (output name, source), declaration order
) -> Result<Vec<CompiledNode>, SystemCompilationFailure>;
// The Vec is in an evaluation order: every node after everything it reads,
// ties broken by declaration order — so it is declaration order whenever
// that already evaluates.

pub struct CompiledNode { /* name, reads, expression, cheap */ }

impl CompiledNode {
    /// The output name this expression defines.
    pub fn name(&self) -> &str;

    /// What this node reads — the edges into it, and the rows
    /// `expression().eval` takes, in this order.
    ///
    /// Canonical order: the inputs it reads (every input, if it has a
    /// computed subscript) in declaration order; then the externals it
    /// names; then the outputs it names; each in declaration order.
    pub fn reads(&self) -> &[Symbol];

    /// The tape: `is_constraint()`, `eval`, `gradient()` as from `compile`.
    pub fn expression(&self) -> &CompiledExpression;

    /// Whether everything this node depends on, transitively, is an input.
    pub fn is_cheap(&self) -> bool;
}

/// A name a node reads, tagged by which of the three lists it came from.
pub enum Symbol {
    Input(String),
    External(String),
    Output(String),
}

impl Symbol {
    pub fn name(&self) -> &str;
}

/// Every problem across every output, each with the name it was found under.
pub struct SystemCompilationFailure {
    pub problems: Vec<NamedProblem>,
}

/// `name` is the output the problem was found in — or, for a declaration
/// clash with no expression to point at (`x1` in both `inputs` and
/// `externals`), the name declared twice.
pub struct NamedProblem {
    pub name: String,
    pub problem: Problem,   // unchanged: carries its own source and span
}
```

From the other side:

```rust
let nodes = sojourn::compile_system(
    &["x1", "x2"],
    &["f2"],
    &[("f1", "x1 + 10"), ("f3", "f1 * f2"), ("c2", "f3 < 100")],
)?;

// Wiring into a graph: one edge per read, by name.
for node in &nodes {
    for read in node.reads() {
        graph.add_edge(read.name(), node.name());
    }
}

// Running one node over n points, once everything it reads is known.
let rows = Mat::from_fn(node.reads().len(), n, |r, c| values[node.reads()[r].name()][c]);
let result = node.expression().eval(rows.as_ref())?;
```

`reads()` is one list doing two jobs on purpose: a caller who wires edges from it gets the
evaluation layout for free, and there are not two lists that must agree.

### What `compile_system` enforces

- Names bind against `inputs ∪ externals ∪ outputs`. A name in none of them is
  `ProblemKind::Unbound`, at its first reference, as `compile` reports it.
- **`var[i]` indexes the inputs, and only the inputs.** A literal subscript past them is
  `DynamicIndexOutOfBounds` at the subscript, with `available` the input count. A computed
  subscript (`var[floor(x1)]`) compiles, reads every input (its node's `reads()` lists them
  all), and faults at run time if the row it names is not an input — it cannot read an
  external or an output, whatever the row layout holds after the inputs.
- A name declared twice — within a list, or across two of the three — is
  `ProblemKind::Duplicate { name }`, at the second definition.
- An output whose expression is a constraint (`is_constraint()`) may not be referenced by
  another expression: `ProblemKind::ConstraintAsValue { name }`, at the reference. A boolean
  is not a value; `opyl` used to get `Unbound` for this by leaving constraints off the list.
- A cycle, self-reference included, is `ProblemKind::Cycle { chain }`, reported once, in the
  expression that closes it, at the reference that closes it; `chain` lists the outputs round
  the cycle starting from that expression, so `g1: g2 + 1`, `g2: g1 + 1` reports under `g2`
  with chain `[g2, g1]` — read as `g2 → g1 → g2`.
- Every problem detectable in one pass is reported in that pass, each under the output (or the
  clashing name) it belongs to: parse failures, unbound names, subscripts past the inputs,
  duplicates, references to constraints, cycles. One thing is out of reach: a cycle that runs
  through an expression which did not parse cannot be seen until that expression does.
- Each node is **cheap** when its transitive closure contains no external — so `f2: f1 * 2`
  with `f1: x1 + 10` is cheap, and `c2: f3 < 100` with `f3: f1 * ansys` is not. The
  classification is the reason a caller can evaluate the cheap constraints before dispatching
  a simulation and skip an infeasible point; `tests/dependency_diagram.rs` does exactly that.

### The subscript rule, and why the evaluator changed for it

Before this, a `Gather` — the tape's instruction for a computed subscript — was bounded by
the width of the row it was handed, and the row was always exactly the inputs, so the two
numbers were the same by construction. A node's row is its `reads()`, which holds externals
and outputs after the inputs, so the bound and the width came apart and the bound had to be
stated. A row's shape is now a value, `RowLayout { inputs, intermediates }` — the two
regions, under those names everywhere — held by the schema and handed to the executors (the
lane, the tile, the WGSL guard) beside the batch: a gather is bounded by `inputs`, the row is
checked against `total()`, and neither reads the other off the data. Not a field on the tape:
the gradient tape, the interval tape and the shader are all derived from the same tape and
none of them would read it. For `compile` and for `ConstraintSystem` every name is an input
and nothing observable changed.

Deliberately not done, and recorded in `docs/todo.md` under *Standing*: proving statically
that a computed subscript can land in range at all, and teaching interval narrowing to follow
the indirection. A gather still reads as `ENTIRE` to `cvg::hc4`, which is sound and
concludes nothing. The feature is half a solution looking for its problem — conditional
variables, probably — and the work waits on knowing what that problem is.

## Acceptance table

The request's probe, with the verdicts `compile_system` gives. Inputs `x1, x2, x3`; external
`ext`; each row is one expression in a document that otherwise holds `f1: x1 + 10` and
`c2: x1 + x2 > 0.5`. Every row is a test in `tests/compile_system.rs`.

| output | expression | verdict |
|---|---|---|
| `f1` | `x1 + 10` | reads `[Input(x1)]`, cheap |
| `f2` | `f1 * 2` | reads `[Output(f1)]`, cheap — transitively |
| `f3` | `f1 * ext` | reads `[External(ext), Output(f1)]`, not cheap |
| `c1` | `f1 < x1` | reads `[Input(x1), Output(f1)]`, constraint, cheap |
| `c2` | `x1 + x2 > 0.5` | reads `[Input(x1), Input(x2)]`, constraint, cheap |
| `s` | `sum(1, 3, i -> var[i]^2)` | reads `[x1, x2, x3]` exactly; no subscript survives |
| `f4` | `var[4]` | `DynamicIndexOutOfBounds { 4, available: 3 }` at the subscript |
| `f5` | `var[floor(x1)] + f1` | reads `[x1, x2, x3, Output(f1)]`; at `x1 = 4` faults `{ 4, 3 }` rather than reading `f1`'s row |
| `h` | `h + 1` | `Cycle { chain: [h] }` |
| `g1`, `g2` | `g2 + 1`, `g1 + 1` | `Cycle { chain: [g2, g1] }` under `g2` |
| `e1` | `` | `EmptyExpression`, named `e1` |
| `e2` | `x9 + 1` | `Unbound { x9 }`, named `e2` |
| `c3` | `c2 * 2 > x1` | `ConstraintAsValue { c2 }` at the reference |
| `x1` | `x2` | `Duplicate { x1 }` — an input redefined |

## Dropped from the request, and why

- **`CompiledSystem`, `Step`/`Run`/`Results`, the layered schedule, `SymbolId`.** A scheduler
  is general-purpose machinery, OASIS already has one, and artemis has not said what it
  needs. Everything the schedule would have derived is derivable from `reads()` by the caller
  who owns the graph; what only sojourn can say — binding, cycles, cheapness — is what is
  handed over. The chained-tool question (an external that reads an output) evaporates with
  it: an external's dependencies are the caller's graph's business.
- **`Bounds` on inputs.** Compile needs count and order, nothing else. Bounds belong to
  whoever builds a `ConstraintSystem` later.
- **`Problem::output: Option<String>`.** `None` meaning "not from a system" is the optional
  field that makes a mistake representable, and a cycle names several outputs anyway.
  `Problem` already carries its own source; what a system failure adds is *which name*, and
  that is a pairing, not a hole in `Problem`.
- **Rejecting computed subscripts.** They stay; the bound moved instead (above).
- **Per-column fault masks and NaN placeholders.** With no runtime in sojourn there is
  nothing to mask; a node's `eval` reports a fault the way `compile`'s does.
- **"`Unbound` reports `0:0` although its message knows the span."** A misread: `x9` and
  `c2` both sit at column 0 in the probe. Positions are on the `Problem` as data; `opyl`
  reads them off and offsets from the document key.

## Out of scope, still

- Bridging the cheap constraints into `ConstraintSystem`, which has no notion of an
  intermediate; it needs a `rewrite` pass that inlines them. Natural follow-on.
- Gradients through a system (the chain rule across intermediates). When it lands,
  `compile_system` moves under `Compiler` beside `compile`, and the gradient's row order —
  today `gradient().symbols()`, first-reference order — gets aligned with `reads()`.
- Interval narrowing through a gather.
