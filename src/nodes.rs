//! Named expressions compiled together: one node per expression, each
//! carrying its edges, in an order that evaluates.
//!
//! A document declares inputs, outputs produced elsewhere, and expressions
//! over both — including over each other. One expression at a time,
//! [`compile`](crate::compile) cannot see what the set says: that `var[4]`
//! walked past the inputs onto an output, that `h: h + 1` is itself, that two
//! outputs define each other, that a constraint has been read as a number,
//! or that an output which names only other outputs is nonetheless decidable
//! from the inputs alone. [`compile_system`] sees all of it, reports every
//! problem it can find in one pass under the name it belongs to, and hands
//! back a [`CompiledNode`] per expression.
//!
//! What it does *not* hand back is a scheduler. Every consumer of this has a
//! graph of its own — a problem definition's model, an optimiser's pipeline
//! — and a node's [`reads`](CompiledNode::reads) is exactly the edge list
//! that graph wants, so the caller draws its edges from it and runs things
//! its own way; `tests/dependency_diagram.rs` is that, in a screen. The
//! `Vec` comes back topologically sorted because the cycle check produces
//! the order for free and a caller without a graph can walk it front to back.
//! `docs/compile-system.md` is the spec, and records what was asked for and
//! not built.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::diagnostics::{NamedProblem, Problem, ProblemKind, Span, SystemCompilationFailure};
use crate::eval::Gradient;
use crate::frontend::rewrite;
use crate::{Ast, CompiledExpression, Schema};

/// A name a node reads, tagged by which of [`compile_system`]'s three lists
/// declared it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Symbol {
    /// A coordinate of the design vector.
    Input(String),
    /// Produced outside babel; declared so that expressions may name it.
    External(String),
    /// Another expression's result.
    Output(String),
}

impl Symbol {
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Input(name) | Self::External(name) | Self::Output(name) => name,
        }
    }
}

/// One expression, compiled against the system it was declared in.
#[derive(Debug, Clone)]
pub struct CompiledNode {
    name: String,
    reads: Vec<Symbol>,
    expression: CompiledExpression,
    cheap: bool,
}

impl CompiledNode {
    /// The output name this expression defines.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// What this node reads — the edges into it, and the rows
    /// [`expression().eval`](CompiledExpression::eval) takes, in this order.
    ///
    /// One list doing both jobs on purpose: a caller who wires edges from it
    /// gets the evaluation layout for free, and there are not two lists that
    /// must agree. Canonical order: the inputs it reads — every input, if it
    /// has a computed subscript, since `var[floor(x1)]` may name any of them
    /// — in declaration order; then the externals it names; then the outputs
    /// it names; each in declaration order.
    #[must_use]
    pub fn reads(&self) -> &[Symbol] {
        &self.reads
    }

    /// The tape: `is_constraint()`, `eval`, `gradient()` as from
    /// [`compile`](crate::compile), over rows laid out as [`reads`](Self::reads).
    #[must_use]
    pub fn expression(&self) -> &CompiledExpression {
        &self.expression
    }

    /// Whether everything this node depends on, transitively, is an input —
    /// so it can be evaluated before anything external is asked for. Cheap
    /// constraints are how a caller skips a point without paying for a
    /// simulation.
    #[must_use]
    pub const fn is_cheap(&self) -> bool {
        self.cheap
    }
}

/// Compiles `expressions` — `(output name, source)` in declaration order —
/// against `inputs`, the design vector in order and the only thing `var[i]`
/// indexes, and `externals`, names produced elsewhere that an expression may
/// read.
///
/// The nodes come back in an evaluation order: every node after everything
/// it reads, ties broken by declaration order — so it is declaration order
/// whenever that already evaluates.
///
/// # Errors
/// [`SystemCompilationFailure`] with every problem found in one pass, each
/// under the name it belongs to: a name declared twice
/// ([`Duplicate`](ProblemKind::Duplicate)), an expression that does not
/// parse, a name nothing declares ([`Unbound`](ProblemKind::Unbound)), a
/// literal subscript past the inputs
/// ([`DynamicIndexOutOfBounds`](ProblemKind::DynamicIndexOutOfBounds)), a
/// constraint read as a value ([`ConstraintAsValue`](ProblemKind::ConstraintAsValue)),
/// and an output that depends on itself ([`Cycle`](ProblemKind::Cycle)).
/// A cycle through an expression that did not parse is the one thing out of
/// reach until it does.
pub fn compile_system(
    inputs: &[&str],
    externals: &[&str],
    expressions: &[(&str, &str)],
) -> Result<Vec<CompiledNode>, SystemCompilationFailure> {
    let mut problems: Vec<NamedProblem> = Vec::new();

    // Declarations: three lists, one namespace. The first declaration of a
    // name is the one references resolve to; a later one is the problem.
    let mut declared: HashSet<&str> = HashSet::new();
    for name in inputs.iter().chain(externals) {
        if !declared.insert(name) {
            problems.push(NamedProblem {
                name: (*name).to_owned(),
                problem: Problem::new(
                    ProblemKind::Duplicate {
                        name: (*name).to_owned(),
                    },
                    "",
                    Span::new(0, 0),
                ),
            });
        }
    }
    // Output name → declaration index, first definition wins.
    let mut outputs: HashMap<&str, usize> = HashMap::new();
    for (index, (name, source)) in expressions.iter().enumerate() {
        if !declared.insert(name) {
            problems.push(NamedProblem {
                name: (*name).to_owned(),
                problem: Problem::new(
                    ProblemKind::Duplicate {
                        name: (*name).to_owned(),
                    },
                    source,
                    whole(source),
                ),
            });
            continue;
        }
        outputs.insert(name, index);
    }

    // Parse, and resolve literal subscripts against the inputs alone —
    // exactly what `ConstraintSystem::new` does against its box. After this
    // a literal `var[2]` is the name `x2` in the symbol list, and the only
    // subscripts left are the ones a row decides.
    let input_schema = Schema::new(inputs.iter().copied());
    let parsed: Vec<Option<Ast>> = expressions
        .iter()
        .enumerate()
        .map(|(index, (name, source))| {
            if outputs.get(name) != Some(&index) {
                return None; // a duplicate: reported, not compiled
            }
            let ast = match crate::parse(source) {
                Ok(ast) => ast,
                Err(failure) => {
                    problems.extend(failure.problems.into_iter().map(|problem| NamedProblem {
                        name: (*name).to_owned(),
                        problem,
                    }));
                    return None;
                }
            };
            match rewrite::resolve_subscripts(ast, &input_schema) {
                Ok(ast) => Some(ast),
                Err(fault) => {
                    problems.push(NamedProblem {
                        name: (*name).to_owned(),
                        problem: Problem::new(fault.kind, source, fault.span),
                    });
                    None
                }
            }
        })
        .collect();

    // Bind by kind. A node's reads are built here in canonical order, and
    // its edges to other outputs are what the graph below is made of.
    let position = |list: &[&str], symbol: &str| list.iter().position(|name| *name == symbol);
    let mut reads: Vec<Vec<Symbol>> = vec![Vec::new(); expressions.len()];
    let mut edges: Vec<Vec<(usize, Span)>> = vec![Vec::new(); expressions.len()];
    for (index, ast) in parsed.iter().enumerate() {
        let Some(ast) = ast else { continue };
        let (name, source) = expressions[index];
        let mut input_reads: Vec<usize> = Vec::new();
        let mut external_reads: Vec<usize> = Vec::new();
        let mut output_reads: Vec<usize> = Vec::new();
        for (symbol, span) in ast.symbols().iter().zip(ast.reference_spans()) {
            let span = span.unwrap_or_else(|| whole(source));
            if let Some(at) = position(inputs, symbol) {
                input_reads.push(at);
            } else if let Some(at) = position(externals, symbol) {
                external_reads.push(at);
            } else if let Some(&at) = outputs.get(symbol.as_str()) {
                if parsed[at].as_ref().is_some_and(Ast::is_constraint) {
                    problems.push(NamedProblem {
                        name: name.to_owned(),
                        problem: Problem::new(
                            ProblemKind::ConstraintAsValue {
                                name: symbol.clone(),
                            },
                            source,
                            span,
                        ),
                    });
                } else {
                    output_reads.push(at);
                    edges[index].push((at, span));
                }
            } else {
                problems.push(NamedProblem {
                    name: name.to_owned(),
                    problem: Problem::new(
                        ProblemKind::Unbound {
                            name: symbol.clone(),
                        },
                        source,
                        span,
                    ),
                });
            }
        }
        // A subscript the row decides may name any input, so the node reads
        // every one of them, in order; nothing else changes.
        if ast.contains_dynamic_lookup() {
            input_reads = (0..inputs.len()).collect();
        } else {
            input_reads.sort_unstable();
        }
        external_reads.sort_unstable();
        output_reads.sort_unstable();
        reads[index].extend(
            input_reads
                .iter()
                .map(|&at| Symbol::Input(inputs[at].to_owned())),
        );
        reads[index].extend(
            external_reads
                .iter()
                .map(|&at| Symbol::External(externals[at].to_owned())),
        );
        reads[index].extend(
            output_reads
                .iter()
                .map(|&at| Symbol::Output(expressions[at].0.to_owned())),
        );
    }

    problems.extend(cycles(expressions, &edges));

    if !problems.is_empty() {
        return Err(SystemCompilationFailure { problems });
    }

    // Everything parsed, bound and acyclic: order, classify, lower. Kahn's
    // algorithm with the smallest declaration index always next, which is
    // deterministic and leaves a declaration order that already evaluates
    // untouched.
    let mut indegree: Vec<usize> = edges.iter().map(Vec::len).collect();
    let mut dependants: Vec<Vec<usize>> = vec![Vec::new(); expressions.len()];
    for (to, reads_from) in edges.iter().enumerate() {
        for (from, _) in reads_from {
            dependants[*from].push(to);
        }
    }
    let mut ready: BinaryHeap<Reverse<usize>> = (0..expressions.len())
        .filter(|&index| indegree[index] == 0)
        .map(Reverse)
        .collect();
    let mut order = Vec::with_capacity(expressions.len());
    while let Some(Reverse(next)) = ready.pop() {
        order.push(next);
        for &dependant in &dependants[next] {
            indegree[dependant] -= 1;
            if indegree[dependant] == 0 {
                ready.push(Reverse(dependant));
            }
        }
    }
    debug_assert_eq!(
        order.len(),
        expressions.len(),
        "acyclic, so every node is reached"
    );

    let mut cheap = vec![false; expressions.len()];
    let mut nodes = Vec::with_capacity(expressions.len());
    for index in order {
        let (name, _) = expressions[index];
        let ast = parsed[index]
            .as_ref()
            .expect("every expression parsed, or the problems above returned");
        cheap[index] = reads[index].iter().all(|read| match read {
            Symbol::Input(_) => true,
            Symbol::External(_) => false,
            Symbol::Output(output) => cheap[outputs[output.as_str()]],
        });

        let schema = Schema::with_intermediates(
            reads[index]
                .iter()
                .filter(|read| matches!(read, Symbol::Input(_)))
                .map(Symbol::name),
            reads[index]
                .iter()
                .filter(|read| !matches!(read, Symbol::Input(_)))
                .map(Symbol::name),
        );
        let expression = crate::eval::bind(ast, &schema, Gradient::BestEffort).expect(
            "every symbol is in the node's own schema, and every literal subscript is resolved",
        );
        nodes.push(CompiledNode {
            name: name.to_owned(),
            reads: std::mem::take(&mut reads[index]),
            expression,
            cheap: cheap[index],
        });
    }
    Ok(nodes)
}

/// The span of all of `source`: where a problem with no narrower place goes.
fn whole(source: &str) -> Span {
    Span::new(0, u32::try_from(source.chars().count()).unwrap_or(u32::MAX))
}

/// Every cycle in `edges` — `edges[to]` listing each `from` it reads, with
/// the span of the reference — as a problem under the expression whose
/// reference closes it, with the chain read from there.
///
/// Depth-first in declaration order with the usual three colours: a
/// reference to a node still on the stack is the back edge that closes a
/// cycle, and the stack from that node to the top is the cycle.
fn cycles(expressions: &[(&str, &str)], edges: &[Vec<(usize, Span)>]) -> Vec<NamedProblem> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Colour {
        Unvisited,
        OnStack,
        Done,
    }
    let mut colour = vec![Colour::Unvisited; expressions.len()];
    let mut stack: Vec<usize> = Vec::new();
    let mut found = Vec::new();

    fn visit(
        node: usize,
        expressions: &[(&str, &str)],
        edges: &[Vec<(usize, Span)>],
        colour: &mut [Colour],
        stack: &mut Vec<usize>,
        found: &mut Vec<NamedProblem>,
    ) {
        colour[node] = Colour::OnStack;
        stack.push(node);
        for &(read, span) in &edges[node] {
            match colour[read] {
                Colour::Unvisited => visit(read, expressions, edges, colour, stack, found),
                Colour::OnStack => {
                    let start = stack
                        .iter()
                        .position(|&on| on == read)
                        .expect("a node on the stack is in the stack");
                    let (name, source) = expressions[node];
                    let chain = std::iter::once(node)
                        .chain(stack[start..].iter().copied().filter(|&on| on != node))
                        .map(|on| expressions[on].0.to_owned())
                        .collect();
                    found.push(NamedProblem {
                        name: name.to_owned(),
                        problem: Problem::new(ProblemKind::Cycle { chain }, source, span),
                    });
                }
                Colour::Done => {}
            }
        }
        stack.pop();
        colour[node] = Colour::Done;
    }

    for node in 0..expressions.len() {
        if colour[node] == Colour::Unvisited {
            visit(
                node,
                expressions,
                edges,
                &mut colour,
                &mut stack,
                &mut found,
            );
        }
    }
    found
}
