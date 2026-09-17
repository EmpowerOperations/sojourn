//! Source text to [`Ast`].
//!
//! Everything here is meaning-preserving. [`parse`] lexes, parses and lowers to
//! [`crate::ast`], then [`rewrite::canonicalize`] normalises the tree
//! *without changing what it computes* — folding constants, inverting monotone
//! comparisons, unrolling aggregates over literal bounds, collecting a term
//! multiplied by itself into a power.
//!
//! That is the line this module draws. A pass that makes the tree easier to
//! analyse belongs here; a pass that lowers it toward one consumer's needs
//! belongs to that consumer. `src/README.md` has the table of where each pass
//! falls and why the order between them is forced.

use crate::ast;
use crate::diagnostics::{CompilationFailure, Fault, Problem, ProblemKind, Span};

pub(crate) mod generated;
pub(crate) mod parse;
pub(crate) mod rewrite;

pub(crate) use parse::{parses_as_variable, translate};

/// Compiles source text into an evaluable expression.
///
/// # Errors
/// Returns [`CompilationFailure`] with every problem found; compilation does
/// not stop at the first one.
pub(crate) fn parse(source: &str) -> Result<Ast, CompilationFailure> {
    if source.is_empty() {
        return Err(CompilationFailure {
            source: source.to_owned(),
            problems: vec![Problem::new(
                ProblemKind::EmptyExpression,
                source,
                Span::new(0, 0),
            )],
        });
    }

    let lowered = match translate(source) {
        Ok(lowered) => lowered,
        Err(problems) => {
            return Err(CompilationFailure {
                source: source.to_owned(),
                problems,
            });
        }
    };

    // Two of these passes report kind and span; rendering needs the source,
    // which lives here rather than in the rewriter.
    let render = |faults: Vec<Fault>| CompilationFailure {
        source: source.to_owned(),
        problems: faults
            .into_iter()
            .map(|fault| Problem::new(fault.kind, source, fault.span))
            .collect(),
    };

    // The canonical form: constants folded, monotone comparisons inverted,
    // aggregates over known bounds unrolled, repeated factors collected into
    // powers — in that order, for the reasons `rewrite::canonicalize` gives.
    let program = rewrite::canonicalize(lowered.program).map_err(render)?;

    Ok(Ast {
        source: source.to_owned(),
        program,
        symbols: lowered.symbols,
        contains_dynamic_lookup: lowered.contains_dynamic_lookup,
        is_constraint: lowered.is_constraint,
    })
}

/// A parsed expression, ready to be bound to a [`Schema`](crate::Schema).
///
/// Crate-private: a caller hands source text to `compile` or to
/// `ConstraintSystem::new` and never sees the tree between.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ast {
    pub(crate) source: String,
    pub(crate) program: ast::Program,
    /// Distinct statically-referenced names in first-reference order.
    /// `ast::GlobalId` indexes into *this*, not into the schema — the AST is
    /// built before any schema exists, so binding (`eval::bind`) is what maps
    /// these onto row positions.
    pub(crate) symbols: Vec<String>,
    pub(crate) contains_dynamic_lookup: bool,
    pub(crate) is_constraint: bool,
}

impl Ast {
    /// The source text this was compiled from.
    #[must_use]
    pub(crate) fn source(&self) -> &str {
        &self.source
    }

    /// Whether the expression uses `var[i]` dynamic lookup.
    ///
    /// A subscript is a one-based index into the whole [`Schema`](crate::Schema) in
    /// declaration order, so such an expression can read a variable it never
    /// names and its [`symbols`](Ast::symbols) are not the whole story.
    /// **A caller must not prune columns it believes are unreferenced while
    /// this is true.**
    ///
    #[must_use]
    pub(crate) const fn contains_dynamic_lookup(&self) -> bool {
        self.contains_dynamic_lookup
    }

    /// Whether the source was a boolean expression, and therefore whether the
    /// result should be read as a constraint residual rather than a value.
    #[must_use]
    pub(crate) const fn is_constraint(&self) -> bool {
        self.is_constraint
    }

    /// Statically-referenced names in first-reference order, indexed by
    /// `ast::GlobalId`.
    #[must_use]
    pub(crate) fn symbols(&self) -> &[String] {
        &self.symbols
    }

    /// Where each symbol is first referenced, indexed like
    /// [`symbols`](Self::symbols): the span to put a caret under when a
    /// symbol turns out to be unbound. Read off the tree — every `Global`
    /// node carries the span of the reference that made it — rather than
    /// recorded beside `symbols`, so it cannot drift from what the tree says.
    /// `None` for a symbol nothing references, which the translator never
    /// produces but the walk does not assume.
    pub(crate) fn reference_spans(&self) -> Vec<Option<Span>> {
        let mut spans = vec![None; self.symbols.len()];
        first_references(&self.program.body.result, &mut spans);
        for assignment in &self.program.body.assignments {
            first_references(&assignment.value, &mut spans);
        }
        spans
    }
}

/// Records the earliest-spanned reference of each global under `expr`.
fn first_references(expr: &ast::Expr, spans: &mut [Option<Span>]) {
    use ast::Kind;
    match &expr.kind {
        Kind::Global(id) => {
            let slot = &mut spans[id.index()];
            if slot.is_none_or(|held| expr.span < held) {
                *slot = Some(expr.span);
            }
        }
        Kind::Literal(_) | Kind::Local(_) => {}
        Kind::Unary { arg, .. } => first_references(arg, spans),
        Kind::Binary { lhs, rhs, .. } | Kind::Compare { lhs, rhs, .. } => {
            first_references(lhs, spans);
            first_references(rhs, spans);
        }
        Kind::NearEq { lhs, rhs, .. } => {
            first_references(lhs, spans);
            first_references(rhs, spans);
        }
        Kind::And { terms } | Kind::Fold { terms, .. } => {
            for term in terms {
                first_references(term, spans);
            }
        }
        Kind::DynamicIndex(index) => first_references(index, spans),
        Kind::Block(block) => {
            for assignment in &block.assignments {
                first_references(&assignment.value, spans);
            }
            first_references(&block.result, spans);
        }
        Kind::Aggregate {
            lower, upper, body, ..
        } => {
            first_references(lower, spans);
            first_references(upper, spans);
            for assignment in &body.assignments {
                first_references(&assignment.value, spans);
            }
            first_references(&body.result, spans);
        }
    }
}
