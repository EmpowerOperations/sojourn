//! Rendering constraints as an SMT-LIB2 document.
//!
//! A pure `&[Ast] -> String`, with no solver attached and none required
//! to test it. That separation is the whole reason for emitting text rather than
//! building a solver's AST: the JVM version transcoded straight into Z3 objects,
//! so the only way to see what it had asked was to ask Z3.
//!
//! # What gets asserted
//!
//! Every constraint asserts as a boolean term — `(assert (> |x| 4.0))` — named
//! `cN` so an unsat core points back at the constraint the author wrote. Side
//! conditions the term depends on, described below, are asserted alongside it.
//!
//! # A comparison is emitted as a comparison
//!
//! It did not used to be. `rewrite_booleans` ran in the shared pipeline and
//! flattened every constraint into a residual before this module saw it, so
//! `x > 4` arrived as `4 - x + EPSILON` — the strictness carried by a nudge that
//! only works *because of `f64` rounding*. Real arithmetic does not round, so
//! emitting it literally put a three-hundred-digit denormal in the document and
//! still meant the wrong thing; this module had to recognise the marker and undo
//! it.
//!
//! `Kind::Compare` and `Kind::NearEq` survive compilation now and each backend
//! lowers them itself. `x > 4` is `(> |x| 4.0)`, and `a == b +/- t` is two
//! bounds `and`-ed rather than `(<= (expr_max …) 0.0)` — an `ite` where a
//! conjunction was meant. Both are easier for a solver and for a reader.
//!
//! # Side conditions
//!
//! Some operations need more than a term. `(/ a b)` needs `b` pinned away from
//! zero, because SMT-LIB leaves division by zero *underspecified* — the solver
//! is free to choose whatever value for it satisfies the constraint, and it
//! does: the first document this emitter handed Z3 came back solved with every
//! variable zero, satisfying `x2 == x1 + x2/2 - x3/x4` through `0/0`. Babel
//! evaluates that point to NaN and rejects it, so the solver's answer would have
//! been binned and the search would have spun. With the guard, the same document
//! yields a real point.
//!
//! `sqrt` and `cbrt` need an auxiliary variable: there is no `sqrt` in QF_NRA,
//! but `y >= 0 and y*y = x` says the same thing, and a solver handles it
//! comfortably.
//!
//! Both need the translator to contribute *commands* alongside its term, which
//! is why the walk carries an accumulator rather than being a pure
//! `Expr -> Sexp`.
//!
//! # Syntax lives in the templates
//!
//! Nothing here writes a parenthesis, a keyword or an operator's SMT-LIB
//! name. The walk below decides *what* to say — which helper, which guard,
//! which auxiliary, what could not be said — and hands each node's already
//! rendered children to a [`Term`], whose askama template under
//! `templates/smt2/` spells it. A template arm closes what it opens, so a
//! term is balanced by construction; the document template owns every
//! newline, so a `;` comment cannot eat the command after it. Both matter
//! more than they sound: `Solver::from_string` reports a syntax error by
//! silently accepting nothing and then answering `sat`, and a comment missing
//! its newline loses one command quietly. `Z3Backend::solve` refuses a
//! document that produced no assertions, `smt::tests` put every document the
//! emitter can build through Z3's parser, and `every_document_is_balanced`
//! below counts the parentheses; those are the nets now that balance is a
//! property of five small templates rather than of a type.
//!
//! This replaced an `Sexp` value type with `sexp!`/`define_fun!` macros on
//! 2026-09-05. Crates exist that would do the s-expression layer — `smtlib`
//! and `aws-smt-ir` are permissively licensed, `smtlib-syntax` is closest in
//! intent but GPL-3.0, which rules it out for a library that gets linked and
//! shipped — and Z3's typed AST would skip text altogether at the price of
//! welding the emitter to one solver and giving up a document you can read,
//! diff and hand to something else. None of them touch the part that is
//! actually hard, which is the domain logic in this file.
//!
//! # What cannot be emitted, and why that is reported rather than dropped
//!
//! This targets `QF_NIRA`: the field operations, comparison, `ite`, `let`, and
//! `to_int`/`to_real`. Anything outside that — the transcendentals, `log`, a
//! power with a non-integer exponent — is *reported* through
//! [`Document::untranslated`].
//!
//! The JVM version dropped such constraints silently, and its own fixture pinned
//! the consequence: `sin of value offset by multiples of pi` asserted that the
//! returned points included ones which **did not satisfy the constraints**. A
//! solver answering a question you quietly did not ask is worse than a solver
//! that says it cannot help.
//!
//! The list is shorter than it was, because some of those gaps were dialect
//! rather than theory. `floor`, `ceil` and `%` used to be on it and are not: the
//! first two are `to_int` and the third is `a - b*trunc(a/b)`, and the only
//! price is the wider logic. What is left is genuinely missing rather than
//! merely unwritten — Z3 has no logarithm under any spelling, and its `sin`
//! parses and then answers `unknown`. dReal has all of them as primitives, at
//! the cost of a subprocess on a customer's machine.
//!
//! # Fidelity
//!
//! Literals are emitted as shortest round-trip decimals, which is *not* exact:
//! SMT-LIB reals are exact decimals, so `0.1` there means one tenth, while the
//! `f64` spelled `0.1` is 0.1000000000000000055511151231257827.
//!
//! Emitting exact dyadic rationals instead would close that particular gap and
//! is a one-function change — but it would not make the model exact, because the
//! far larger gap is that SMT reasons in *real arithmetic* while babel evaluates
//! in `f64`. Every operation rounds; `x^3` as `(* x x x)` need not equal
//! `x.powf(3.0)` in the last place. The model is a real-arithmetic idealisation
//! of the computation, deliberately — and a solver's points are filtered through
//! babel's own `evaluate` before anybody sees them. Exact literals would be false
//! precision about everything else.

use askama::Template;

use crate::ast::{AggregateKind, BinaryOp, Block, CompareOp, Expr, Kind, UnaryOp};
use crate::solve::SmtLogic;
use crate::{Ast, ast};
use crate::{InputVariable, Point};

/// The unary operators SMT-LIB can spell for `Real`.
///
/// A subset of [`UnaryOp`] on purpose: the operator template matches over
/// this exhaustively, so the transcendentals — which the walk refuses — and
/// the roots — which become auxiliaries — cannot reach it by accident.
#[derive(Debug, Clone, Copy)]
enum SmtUnary {
    Negate,
    Abs,
    Sqr,
    Cube,
    Sgn,
    Floor,
    Ceil,
}

/// The binary operators SMT-LIB can spell for `Real`. `Pow` is spelled as
/// multiplication for a whole exponent and refused otherwise; `LogB` is
/// refused. See [`SmtUnary`].
#[derive(Debug, Clone, Copy)]
enum SmtBinary {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Max,
    Min,
}

/// One SMT-LIB term, in the shape `templates/smt2/term.smt2.jinja` renders.
///
/// Children are already rendered: the recursion is in the walk, the syntax
/// in the template, and a node closes what it opens.
#[derive(Template)]
#[template(path = "smt2/term.smt2.jinja", escape = "none")]
enum Term<'a> {
    /// An expression identifier, quoted.
    Symbol(&'a str),
    /// A `let`-bound slot.
    Local(usize),
    /// A non-negative literal, digits from [`decimal`].
    Real(String),
    /// A negative literal: the operator `-` applied to the digits.
    NegativeReal(String),
    Unary(SmtUnary, String),
    Binary(SmtBinary, String, String),
    Relation(CompareOp, String, String),
    Fold(AggregateKind, Vec<String>),
    And(Vec<String>),
    /// `left == right +/- bound`, as two bounds.
    NearEq(String, String, String),
    /// `(let ((lN value)) body)`.
    Let(usize, String, String),
}

/// A command a term depends on, in the shape `templates/smt2/condition.smt2.jinja`
/// renders. Written out immediately before the assertion that uses the term.
#[derive(Template)]
#[template(path = "smt2/condition.smt2.jinja", escape = "none")]
enum Condition {
    DivisorNonZero(String),
    DeclareAuxiliary(String),
    NonNegative(String),
    /// `name * name = arg`.
    SquareOf(String, String),
    /// `name * name * name = arg`.
    CubeOf(String, String),
}

/// The `define-fun` helpers, `templates/smt2/prelude.smt2.jinja`.
#[derive(Template)]
#[template(path = "smt2/prelude.smt2.jinja", escape = "none")]
struct Prelude;

/// The whole document, in the shape `templates/smt2/document.smt2.jinja`
/// renders: one command per line, the template owning every newline.
#[derive(Template)]
#[template(path = "smt2/document.smt2.jinja", escape = "none")]
struct Script {
    logic: String,
    prelude: String,
    inputs: Vec<Declared>,
    constraints: Vec<Translated>,
    /// A term the answer must satisfy on top of the constraints, asked
    /// *unnamed* so it can never reach an unsat core. See [`keep_away_from`].
    exclusion: Option<String>,
}

struct Declared {
    /// The quoted symbol.
    symbol: String,
    /// The box, as rendered reals; `None` when a bound is not finite, which is
    /// the caller's problem to notice rather than something to assert.
    bounds: Option<Bounds>,
}

struct Bounds {
    low: String,
    high: String,
}

struct Translated {
    /// The `:named` tag, from [`core_name`], so an unsat core reads back
    /// through [`core_index`].
    name: String,
    /// The constraint as written, made safe for a `;` comment.
    source: String,
    /// The assertion, or the comment explaining why there is none.
    outcome: Result<Assertion, String>,
}

/// Text that stays on one comment line. An expression's source may legally
/// contain newlines, and one reaching the output verbatim would break out of
/// its comment and be read as commands, so interior whitespace — newlines
/// included — collapses to single spaces.
fn comment_safe(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Why a constraint could not be handed to a solver.
///
/// The pool's response to any of these is the same — fall back to sampling —
/// and that is the problem this type exists for. A caller who watches their
/// generator crawl on a narrow region deserves to know it crawled because
/// nothing could be asked of the solver, and which part of their expression was
/// responsible. Silence here was the JVM implementation's worst habit.
///
/// Reported through `tracing` at `INFO` rather than returned, because there is
/// nothing the caller can *do* differently — sampling is already the fallback —
/// and an error would imply otherwise.
///
/// **It names the first construct the walk could not render, not the only one.**
/// The walk descends before it refuses, so the reason is the innermost failure:
/// `x1 > sin(ln(cos(2.1^x1)))` is reported against its `2.1^x1`, which is true
/// and is not the whole story. Innermost is the better default — it names an
/// actual construct rather than whatever contained it — but a reader should not
/// read one reason as an exhaustive account of a deeply nested expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// A scalar expression, which has no `<= 0` reading to assert.
    NotABooleanExpression,
    /// `sin`, `cos`, `tan` and the rest. Z3 parses them and then answers
    /// `unknown`; `rewrite::invert_monotone` could not rewrite this one away
    /// because the function is not monotone, or because the variable appears on
    /// both sides of the comparison.
    Transcendental(&'static str),
    /// A logarithm `rewrite::invert_monotone` could not turn into a bound —
    /// a variable base, or one nested where there is no comparison to invert.
    Logarithm,
    /// An exponent that is neither a constant whole number nor invertible.
    RealExponent,
    /// `var[i]` with a computed subscript: a load from a row the solver has no
    /// model of.
    ComputedSubscript,
    /// A literal that is not a finite `Real`. Unreachable while
    /// `rewrite::fold_constants` refuses non-finite constants, and kept so the
    /// emitter does not depend on that from a distance.
    NonFiniteLiteral,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotABooleanExpression => {
                f.write_str("it is a scalar expression, not a constraint")
            }
            Self::Transcendental(function) => write!(
                f,
                "`{function}` is outside the solver's theory, and could not be \
                 rewritten away — it is not monotone, or its variable appears on \
                 both sides of the comparison"
            ),
            Self::Logarithm => f.write_str(
                "the solver has no logarithm, and this one could not be inverted \
                 into a bound — its base varies, or it is nested where there is \
                 no comparison to invert against",
            ),
            Self::RealExponent => {
                f.write_str("the exponent is neither a constant whole number nor invertible")
            }
            Self::ComputedSubscript => {
                f.write_str("`var[i]` with a computed subscript is a load the solver cannot model")
            }
            Self::NonFiniteLiteral => {
                f.write_str("it contains a literal that is not a finite real")
            }
        }
    }
}

/// An SMT-LIB2 document, and an honest account of what it left out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Document {
    pub(crate) text: String,
    /// Indices into the constraints given, for those that could not be
    /// expressed. A solver's verdict says nothing about these, so a caller that
    /// ignores this list will over-trust the answer — which is why it is not an
    /// `Option` or a log line.
    pub(crate) untranslated: Vec<usize>,
}

/// The prelude as text, for tests that put a helper's meaning to Z3.
#[cfg(test)]
fn prelude() -> String {
    Prelude.render().expect("the prelude template renders")
}

/// The document, plus — when `avoid` is non-empty — an assertion that the answer
/// lie outside the box those points span, widened by `reach` on every side.
///
/// An empty `avoid` asks the plain question, which is the point: "find a point"
/// and "find a *different* point" are one question asked with nothing and with
/// something to stay away from, not two code paths.
pub(crate) fn emit_away_from<'a>(
    inputs: &[InputVariable],
    constraints: impl IntoIterator<Item = &'a Ast>,
    logic: &SmtLogic,
    avoid: &[Point],
    reach: f64,
) -> Document {
    let declared = inputs
        .iter()
        .map(|input| Declared {
            symbol: symbol(&input.name),
            bounds: match (real(input.lower_bound), real(input.upper_bound)) {
                (Some(low), Some(high)) => Some(Bounds { low, high }),
                _ => None,
            },
        })
        .collect();

    let mut untranslated = Vec::new();
    let translated = constraints
        .into_iter()
        .enumerate()
        .map(|(index, constraint)| Translated {
            name: core_name(index),
            source: comment_safe(constraint.source()),
            outcome: match translate(constraint, index, inputs) {
                Ok(assertion) => Ok(assertion),
                Err(reason) => {
                    untranslated.push(index);
                    // The pool's answer to this is to fall back on sampling,
                    // and it does so without saying anything. On a narrow
                    // region that is the difference between a fast answer and
                    // a slow one, so the caller is told which constraint and
                    // why — see `Refusal`.
                    tracing::info!(
                        constraint = %constraint.source(),
                        %reason,
                        "no solver can be asked about this constraint; sampling must find it unaided"
                    );
                    Err(comment_safe(&format!("NOT TRANSLATED - {reason}")))
                }
            },
        })
        .collect();

    let text = Script {
        logic: logic.to_string(),
        prelude: Prelude.render().expect("the prelude template renders"),
        inputs: declared,
        constraints: translated,
        exclusion: keep_away_from(inputs, avoid, reach),
    }
    .render()
    .expect("the document template renders for every translated system");

    Document { text, untranslated }
}

/// `(or (< x lo) (> x hi) ...)` over the box `avoid` spans widened by `reach`,
/// or `None` when there is nothing to avoid or no way to say so.
///
/// # Why `reach` and not just the box
///
/// Excluding the points themselves does not work, and this is measured rather
/// than supposed. On `(x + 2) * (x - 1) == 0 +/- 1e-9`, excluding a witness at
/// `x = 1` makes the solver answer `0.999999999767` — it returns the *nearest*
/// satisfying point, and the component it is in is far thinner than the gap to
/// the next one, so the box crawls a fifth of a nanometre a round and never
/// escapes. The exclusion has to be on the scale of the **gap**, which nothing
/// knows in advance; `reach` is the caller's current guess at it, halved every
/// time the answer comes back `unsat`.
///
/// # The shape is a box, and that is a limitation worth naming
///
/// "Outside the box" means *at least one coordinate* differs by more than
/// `reach`. In one dimension that is exactly right. In two hundred it is
/// satisfiable by moving one coordinate and staying adjacent in the other one
/// hundred and ninety-nine, so it stops meaning "somewhere else" — and widening
/// does not repair that, because the flaw is the shape. A ball —
/// `sum of squares > reach^2` — forces real displacement and is expressible in
/// the logic already in use. Nothing in the corpus is a high-dimensional
/// *disjoint* region, so no test would presently catch the difference.
///
/// **Asserted unnamed** by the caller's template: an unsat core is what
/// [`Infeasibility::Proved`](crate::Infeasibility::Proved) blames to a
/// caller, and this clause is our invention rather than a constraint anybody
/// wrote.
fn keep_away_from(inputs: &[InputVariable], avoid: &[Point], reach: f64) -> Option<String> {
    if avoid.is_empty() {
        return None;
    }

    let mut literals = Vec::new();
    for (axis, input) in inputs.iter().enumerate() {
        let (low, high) = avoid
            .iter()
            .filter_map(|point| point.get(axis).copied())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
                (low.min(value), high.max(value))
            });
        // A non-finite bound cannot be written down, and asserting nothing beats
        // asserting something true of everything.
        let (Some(low), Some(high)) = (real(low - reach), real(high + reach)) else {
            return None;
        };
        let name = symbol(&input.name);
        literals.push(format!("(< {name} {low})"));
        literals.push(format!("(> {name} {high})"));
    }

    (!literals.is_empty()).then(|| format!("(or {})", literals.join(" ")))
}

/// The two name lists an expression resolves against, which are not the same
/// list.
///
/// `Global` holds an index into the expression's own statically-referenced
/// symbols, because the AST is built before any schema exists. `var[i]` holds a
/// one-based index into the *schema* — declaration order over every input,
/// including ones this expression never names. Using either for the other
/// resolves to the wrong variable silently, so both are carried explicitly.
struct Names<'a> {
    symbols: &'a [String],
    inputs: &'a [InputVariable],
    /// Which constraint is being translated. Only used to keep auxiliary names
    /// distinct across a document — two constraints each declaring `aux0` is a
    /// redeclaration, and the solver rejects the whole thing.
    constraint: usize,
    /// Complete SMT-LIB commands the term depends on: divisor guards, and the
    /// declarations and defining assertions of auxiliary variables. Written out
    /// immediately before the assertion that uses the term.
    conditions: Vec<String>,
    auxiliaries: usize,
    /// Why the walk gave up, set at whichever site returned `None` first.
    ///
    /// An accumulator field rather than a return type because the walk is
    /// already `Option`-shaped and threading a reason through every `?` would
    /// cost more than it explains. `refuse` below is the only writer.
    refused: Option<Refusal>,
}

/// The name an assertion is tagged with, so an unsat core can be read back as
/// constraint indices.
fn core_name(index: usize) -> String {
    format!("c{index}")
}

/// Parses [`core_name`] back. Lives next to it so the two cannot drift.
pub(crate) fn core_index(name: &str) -> Option<usize> {
    name.strip_prefix('c')?.parse().ok()
}

/// One constraint, ready to assert.
struct Assertion {
    conditions: Vec<String>,
    /// The constraint as a boolean term, ready to assert.
    ///
    /// It used to be a *residual* plus a relation to compare it against zero,
    /// because that is all the front end left behind. Now that `Kind::Compare`
    /// survives compilation this is `(> x 5.0)` rather than
    /// `(< (- 5.0 x) 0.0)` — the thing the author wrote.
    claim: String,
}

/// One constraint as a boolean term and its side conditions, or the reason it
/// could not be written down.
fn translate(
    constraint: &Ast,
    index: usize,
    inputs: &[InputVariable],
) -> Result<Assertion, Refusal> {
    // A scalar expression has no `<= 0` reading, so asserting one would invent a
    // constraint the user did not write.
    if !constraint.is_constraint {
        return Err(Refusal::NotABooleanExpression);
    }
    let mut names = Names {
        symbols: &constraint.symbols,
        inputs,
        constraint: index,
        conditions: Vec::new(),
        auxiliaries: 0,
        refused: None,
    };

    // The walk is `Option`-shaped; `Names::refused` carries the reason it
    // stopped. This pairs the two back up at the one place that needs both.
    macro_rules! rendered {
        ($walk:expr) => {
            match $walk {
                Some(rendered) => rendered,
                None => {
                    return Err(names
                        .refused
                        .expect("a walk that returned None recorded why"));
                }
            }
        };
    }

    let claim = rendered!(names.claim(&constraint.program.body));
    Ok(Assertion {
        conditions: names.conditions,
        claim,
    })
}

impl Names<'_> {
    /// Records why the walk is stopping and stops it.
    ///
    /// Mutating and returning `None` is the whole of its job, which is why it
    /// is called at the point of refusal rather than hidden inside something
    /// else. The first reason wins: it is the innermost, and therefore the one
    /// naming the actual construct rather than whatever contained it.
    fn refuse(&mut self, reason: Refusal) -> Option<String> {
        self.refused.get_or_insert(reason);
        None
    }

    /// A constraint's block as a boolean term.
    ///
    /// Mirrors [`block`](Self::block), which renders the scalar half. The two
    /// are separate because the grammar keeps them separate: a boolean is the
    /// root of a constraint and never an operand, so exactly one node in the
    /// tree needs this treatment and every node below it needs the other.
    fn claim(&mut self, body: &Block) -> Option<String> {
        let mut rendered = self.boolean(&body.result)?;
        for assignment in body.assignments.iter().rev() {
            let value = self.expression(&assignment.value)?;
            rendered = term(Term::Let(assignment.slot.index(), value, rendered));
        }
        Some(rendered)
    }

    /// One boolean node, rendered as the comparison the author wrote.
    ///
    /// Everything here used to arrive as arithmetic: `x > 5` as
    /// `(< (- 5.0 x) 0.0)` with a denormal standing in for strictness, and
    /// `a == b +/- t` as `(<= (expr_max …) 0.0)` — an `ite` where a
    /// conjunction was meant. A solver is markedly better at the direct form,
    /// and a reader is too.
    fn boolean(&mut self, expr: &Expr) -> Option<String> {
        match &expr.kind {
            Kind::Compare { op, lhs, rhs } => {
                let left = self.expression(lhs)?;
                let right = self.expression(rhs)?;
                Some(term(Term::Relation(*op, left, right)))
            }

            // `|a - b| <= t`, as two bounds rather than one `max`. SMT-LIB has
            // no absolute value, and a conjunction of linear bounds is the
            // easiest shape a solver can be given.
            Kind::NearEq {
                lhs,
                rhs,
                tolerance,
            } => {
                let left = self.expression(lhs)?;
                let right = self.expression(rhs)?;
                let bound = match real(*tolerance) {
                    Some(rendered) => rendered,
                    None => return self.refuse(Refusal::NonFiniteLiteral),
                };
                Some(term(Term::NearEq(left, right, bound)))
            }

            Kind::And { terms } => {
                let mut rendered = Vec::with_capacity(terms.len());
                for conjunct in terms {
                    rendered.push(self.boolean(conjunct)?);
                }
                Some(term(Term::And(rendered)))
            }

            // The grammar puts a boolean at the root of a constraint and
            // nowhere else, so anything else here is a scalar where a truth
            // value was wanted.
            _ => self.refuse(Refusal::NotABooleanExpression),
        }
    }

    fn block(&mut self, body: &Block) -> Option<String> {
        let mut rendered = self.expression(&body.result)?;
        // Innermost first, so that earlier assignments end up in outer `let`s
        // and stay visible to the later ones.
        for assignment in body.assignments.iter().rev() {
            let value = self.expression(&assignment.value)?;
            rendered = term(Term::Let(assignment.slot.index(), value, rendered));
        }
        Some(rendered)
    }

    fn expression(&mut self, expr: &Expr) -> Option<String> {
        match &expr.kind {
            Kind::Literal(value) => match real(*value) {
                Some(rendered) => Some(rendered),
                None => self.refuse(Refusal::NonFiniteLiteral),
            },
            Kind::Global(id) => Some(symbol(self.symbols.get(id.index())?)),
            Kind::Local(slot) => Some(term(Term::Local(slot.index()))),

            // A literal subscript names a variable, so it resolves here — and it
            // resolves against the schema, not against this expression's own
            // symbols. A computed one cannot: `var[i]` is a load from a row the
            // solver has no model of.
            Kind::DynamicIndex(index) => match index.kind {
                Kind::Literal(value) => {
                    let one_based = ast::to_index(value)?;
                    let position = usize::try_from(one_based.checked_sub(1)?).ok()?;
                    Some(symbol(&self.inputs.get(position)?.name))
                }
                _ => self.refuse(Refusal::ComputedSubscript),
            },

            Kind::Unary { op, arg } => {
                let rendered = self.expression(arg)?;
                self.unary(*op, &rendered)
            }
            Kind::Binary { op, lhs, rhs } => self.binary(*op, lhs, rhs),

            Kind::Fold { kind, terms } => {
                let mut rendered = Vec::with_capacity(terms.len());
                for operand in terms {
                    rendered.push(self.expression(operand)?);
                }
                Some(match rendered.len() {
                    // SMT-LIB's `+` and `*` want at least two arguments.
                    0 => real(kind.identity())?,
                    1 => rendered.into_iter().next()?,
                    _ => term(Term::Fold(*kind, rendered)),
                })
            }

            Kind::Block(inner) => self.block(inner),

            // The front end unrolls every aggregate or refuses the expression;
            // one reaching a backend is a bug in the passes, not a formulation.
            Kind::Aggregate { .. } => {
                unreachable!("`unroll_aggregates` leaves no aggregate for a backend to see")
            }

            // A boolean is the root of a constraint and never an operand, so
            // the scalar walk cannot meet one. `lambdaExpr` takes a
            // `scalarBlock`, which makes that a fact about the grammar rather
            // than a hope about what users write.
            Kind::Compare { .. } | Kind::NearEq { .. } | Kind::And { .. } => {
                unreachable!("the grammar keeps booleans out of scalar position")
            }
        }
    }

    fn binary(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr) -> Option<String> {
        let left = self.expression(lhs)?;
        let right = self.expression(rhs)?;
        let spelled = match op {
            BinaryOp::Add => SmtBinary::Add,
            BinaryOp::Sub => SmtBinary::Sub,
            BinaryOp::Mul => SmtBinary::Mul,
            BinaryOp::Max => SmtBinary::Max,
            BinaryOp::Min => SmtBinary::Min,
            BinaryOp::Div => {
                // Without this the solver may satisfy the constraint *through*
                // the division, because SMT-LIB does not say what `x/0` is.
                self.conditions
                    .push(condition(Condition::DivisorNonZero(right.clone())));
                SmtBinary::Div
            }

            // SMT-LIB's own `mod` is integer-only, so this goes through
            // `expr_rem` — `a - b*trunc(a/b)`, which keeps Java's sign rule.
            // The guard is the one division needs and for the same reason:
            // `a % 0` is NaN in babel and the pool bins NaN residuals, but
            // SMT-LIB leaves `/0` underspecified, so without it a solver may
            // satisfy the constraint *through* a zero divisor and hand back a
            // point that is then thrown away.
            BinaryOp::Rem => {
                self.conditions
                    .push(condition(Condition::DivisorNonZero(right.clone())));
                SmtBinary::Rem
            }

            // `rewrite::invert_monotone` turns `log(a, u) op c` for a constant
            // base into a bound on `u`, so a `log` still spelled as one has a
            // variable base, or sits inside another function where there was no
            // comparison to invert against. What is left needs a logarithm, and
            // Z3 has none under any spelling. Neither does cvc5.
            BinaryOp::LogB => return self.refuse(Refusal::Logarithm),

            // A whole exponent is a polynomial, spelled as the multiplication
            // every logic accepts: Z3's own `^` parses only with no logic set
            // or under `ALL`, and `(^ x -1)` is satisfiable at zero because
            // its division is total. So the reciprocal goes through the same
            // guard every division gets. Anything else — a real exponent or a
            // variable one — is `exp(n * ln x)`, and there is no `exp`. Nor
            // would passing it through help: Z3 runs minutes past its rlimit
            // on `(^ x 1.234)`, which is the hole `smt::Z3Backend` leashes.
            BinaryOp::Pow => {
                let Some(n) = rhs.whole_exponent() else {
                    return self.refuse(Refusal::RealExponent);
                };
                let count = usize::try_from(n.unsigned_abs()).expect("bounded by POWER_LIMIT");
                let product = match count {
                    0 => return real(1.0),
                    1 => left,
                    _ => term(Term::Fold(AggregateKind::Prod, vec![left; count])),
                };
                if n > 0 {
                    return Some(product);
                }
                self.conditions
                    .push(condition(Condition::DivisorNonZero(product.clone())));
                return Some(term(Term::Binary(
                    SmtBinary::Div,
                    "1.0".to_owned(),
                    product,
                )));
            }
        };
        Some(term(Term::Binary(spelled, left, right)))
    }

    /// A fresh auxiliary variable, declared and returned by name.
    fn fresh_auxiliary(&mut self) -> String {
        let name = format!("aux_{}_{}", self.constraint, self.auxiliaries);
        self.auxiliaries += 1;
        self.conditions
            .push(condition(Condition::DeclareAuxiliary(name.clone())));
        name
    }

    fn unary(&mut self, op: UnaryOp, arg: &str) -> Option<String> {
        let spelled = match op {
            UnaryOp::Negate => SmtUnary::Negate,
            UnaryOp::Abs => SmtUnary::Abs,
            UnaryOp::Sqr => SmtUnary::Sqr,
            UnaryOp::Cube => SmtUnary::Cube,
            UnaryOp::Sgn => SmtUnary::Sgn,
            UnaryOp::Floor => SmtUnary::Floor,
            UnaryOp::Ceil => SmtUnary::Ceil,

            // QF_NRA has no `sqrt`, but `y >= 0 and y*y = x` says the same. It
            // also gets the domain right for free: for a negative `x` there is
            // no such `y`, which is exactly babel's NaN.
            UnaryOp::Sqrt => {
                let name = self.fresh_auxiliary();
                self.conditions
                    .push(condition(Condition::NonNegative(name.clone())));
                self.conditions
                    .push(condition(Condition::SquareOf(name.clone(), arg.to_owned())));
                return Some(name);
            }
            // No sign constraint: the cubic is one-to-one over the reals, so it
            // pins a negative root as readily as a positive one.
            UnaryOp::Cbrt => {
                let name = self.fresh_auxiliary();
                self.conditions
                    .push(condition(Condition::CubeOf(name.clone(), arg.to_owned())));
                return Some(name);
            }

            // The transcendentals are dReal primitives and Z3 non-starters:
            // `sin` and friends parse and then answer `unknown` on anything
            // narrow enough to be worth asking, and `ln` is not a symbol at
            // all. Reported rather than dropped — see `Document::untranslated`.
            UnaryOp::Ln
            | UnaryOp::Log10
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Tan
            | UnaryOp::Asin
            | UnaryOp::Acos
            | UnaryOp::Atan
            | UnaryOp::Sinh
            | UnaryOp::Cosh
            | UnaryOp::Tanh
            | UnaryOp::Cot => return self.refuse(Refusal::Transcendental(name_of(op))),
        };
        Some(term(Term::Unary(spelled, arg.to_owned())))
    }
}

/// A transcendental's name as the author wrote it, for the refusal message.
///
/// Only the ones that can be refused: everything else is rendered rather than
/// named, so a name would be dead weight.
fn name_of(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Ln => "ln",
        UnaryOp::Log10 => "log",
        UnaryOp::Sin => "sin",
        UnaryOp::Cos => "cos",
        UnaryOp::Tan => "tan",
        UnaryOp::Asin => "asin",
        UnaryOp::Acos => "acos",
        UnaryOp::Atan => "atan",
        UnaryOp::Sinh => "sinh",
        UnaryOp::Cosh => "cosh",
        UnaryOp::Tanh => "tanh",
        UnaryOp::Cot => "cot",
        _ => "an unsupported function",
    }
}

/// A [`Term`], rendered.
fn term(term: Term<'_>) -> String {
    term.render()
        .expect("the term template renders every variant")
}

/// A [`Condition`], rendered.
fn condition(condition: Condition) -> String {
    condition
        .render()
        .expect("the condition template renders every variant")
}

/// An expression identifier as a quoted SMT-LIB symbol.
fn symbol(name: &str) -> String {
    term(Term::Symbol(name))
}

/// An `f64` as an SMT-LIB `Real` literal.
///
/// Three things this has to get right that `format!("{value}")` does not: a
/// `Real` literal must carry a decimal point (`1` is an `Int`, and SMT-LIB will
/// not mix the sorts); a negative number is the *operator* `-` applied to a
/// literal, which is why the sign is a [`Term`] variant and not a character;
/// and exponent notation is not a `Real` literal at all, so `1e300` has to be
/// written out.
///
/// `None` for infinities and NaN, which have no `Real` to be.
fn real(value: f64) -> Option<String> {
    if !value.is_finite() {
        return None;
    }
    let digits = decimal(value.abs())?;
    Some(term(if value.is_sign_negative() {
        Term::NegativeReal(digits)
    } else {
        Term::Real(digits)
    }))
}

fn decimal(magnitude: f64) -> Option<String> {
    // `{:?}` gives the shortest representation that round-trips and always
    // includes a decimal point — but it uses exponent notation at the extremes.
    let shortest = format!("{magnitude:?}");
    if !shortest.contains(['e', 'E']) {
        return Some(shortest);
    }

    // Widen a fixed-point rendering until it round-trips. Bounded by the widest
    // an `f64` can need, which is the smallest subnormal at about 1e-324.
    let padded = [20usize, 40, 80, 160, 340, 700, 1_100]
        .into_iter()
        .map(|places| format!("{magnitude:.places$}"))
        .find(|rendered| rendered.parse::<f64>() == Ok(magnitude))?;

    // Drop the zeros the fixed width padded on, so `1e-5` reads as `0.00001`
    // rather than `0.00001000000000000000`. The decimal point stops the trim
    // from eating an integer's own zeros, and one digit is put back so the
    // literal stays a `Real` rather than becoming an `Int`.
    let trimmed = padded.trim_end_matches('0');
    let candidate = if trimmed.ends_with('.') {
        format!("{trimmed}0")
    } else {
        trimmed.to_owned()
    };
    Some(if candidate.parse::<f64>() == Ok(magnitude) {
        candidate
    } else {
        padded
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `real` renders a term; these tests are about how.
    fn render(value: f64) -> Option<String> {
        real(value)
    }

    fn document(variables: &[(&str, f64, f64)], sources: &[&str]) -> Document {
        let inputs: Vec<InputVariable> = variables
            .iter()
            .map(|(name, low, high)| InputVariable::new(*name, *low, *high))
            .collect();
        let constraints: Vec<Ast> = sources
            .iter()
            .map(|source| crate::parse(source).expect("test constraint should compile"))
            .collect();
        emit_away_from(&inputs, &constraints, &SmtLogic::default(), &[], 0.0)
    }

    /// An aggregate's subscript is arithmetic, and arithmetic that folds.
    ///
    /// `var[i-1]` unrolls to `var[2 - 1]`, and this refused it as a computed
    /// subscript until `rewrite::substitute` learned to fold — `fold_constants`
    /// runs *before* unrolling, so nothing else was going to. Every aggregate
    /// subscript in the corpus is arithmetic (Rosenbrock's `var[i-1]`,
    /// `var[2*i-1]`), so this was most of them, refused to the solver for want
    /// of one reduction.
    #[test]
    fn a_folded_subscript_translates() {
        let inputs = vec![
            InputVariable::new("x1", 0.0, 10.0),
            InputVariable::new("x2", 0.0, 10.0),
        ];
        let expression = crate::parse("sum(2, 2, i -> var[i-1]) > 0").expect("should compile");
        let document = emit_away_from(
            &inputs,
            std::slice::from_ref(&expression),
            &SmtLogic::default(),
            &[],
            0.0,
        );

        assert!(
            document.untranslated.is_empty(),
            "a constant subscript resolves; document was {}",
            document.text
        );
        assert!(document.text.contains("|x1|"), "{}", document.text);
    }

    /// The prelude is a hand-written table and nothing else pins what is *in*
    /// it. Swap `expr_min`'s `<=` for `>=` and the document still parses, Z3
    /// still solves it, and every other test in the crate still passes — because
    /// no case in the corpus exercises `min` through a solver. These are the
    /// assertions that would not survive it.
    ///
    /// Each claim is closed (no free variables), so Z3 answering `sat` means the
    /// claim is true and `unsat` means it is false.
    #[test]
    fn the_prelude_helpers_mean_what_they_say() {
        use crate::cvg::smt::{Outcome, SmtBackend, Z3Backend};

        let claims = [
            ("(= (expr_abs (- 3.0)) 3.0)", true),
            ("(= (expr_abs 3.0) 3.0)", true),
            ("(= (expr_abs 0.0) 0.0)", true),
            ("(= (expr_abs (- 3.0)) (- 3.0))", false),
            ("(= (expr_sqr (- 3.0)) 9.0)", true),
            ("(= (expr_cube (- 2.0)) (- 8.0))", true),
            ("(= (expr_cube 2.0) 8.0)", true),
            // The pair most worth pinning: they differ only in one character,
            // and swapping them is invisible everywhere else.
            ("(= (expr_max 2.0 5.0) 5.0)", true),
            ("(= (expr_max 5.0 2.0) 5.0)", true),
            ("(= (expr_min 2.0 5.0) 2.0)", true),
            ("(= (expr_min 5.0 2.0) 2.0)", true),
            ("(= (expr_max 2.0 5.0) 2.0)", false),
            ("(= (expr_min 2.0 5.0) 5.0)", false),
            // Babel's `sgn` follows Java: zero maps to zero. Rust's
            // `f64::signum` returns 1.0 there, so this is a real divergence and
            // the emitter has to encode babel's version, not the host's.
            ("(= (expr_sgn (- 4.0)) (- 1.0))", true),
            ("(= (expr_sgn 4.0) 1.0)", true),
            ("(= (expr_sgn 0.0) 0.0)", true),
            ("(= (expr_sgn 0.0) 1.0)", false),
            // `to_int` is floor, so the negative cases are where a plausible
            // wrong encoding shows up. Truncation would give -2.0 here.
            ("(= (expr_floor 2.7) 2.0)", true),
            ("(= (expr_floor (- 2.7)) (- 3.0))", true),
            ("(= (expr_floor (- 2.7)) (- 2.0))", false),
            ("(= (expr_floor 3.0) 3.0)", true),
            ("(= (expr_ceil 2.3) 3.0)", true),
            ("(= (expr_ceil (- 2.3)) (- 2.0))", true),
            ("(= (expr_ceil (- 2.3)) (- 3.0))", false),
            ("(= (expr_ceil 3.0) 3.0)", true),
            // The other half of the same risk. Babel's `%` is Java's, so the
            // sign follows the dividend: floored modulo would answer 2.0 to
            // the third of these and 0.5 to the fifth.
            ("(= (expr_rem 10.0 4.5) 1.0)", true),
            ("(= (expr_rem 7.0 3.0) 1.0)", true),
            ("(= (expr_rem (- 7.0) 3.0) (- 1.0))", true),
            ("(= (expr_rem (- 7.0) 3.0) 2.0)", false),
            ("(= (expr_rem 7.0 (- 3.0)) 1.0)", true),
            ("(= (expr_rem 7.5 2.5) 0.0)", true),
        ];

        let preamble = format!("{}\n", prelude());
        for (claim, should_hold) in claims {
            let document = format!("(set-logic QF_NIRA)\n{preamble}(assert {claim})\n");
            let reply = Z3Backend
                .solve(&document, 0, &crate::cvg::Cancellation::never())
                .unwrap_or_else(|e| panic!("Z3 rejected {claim}: {e}"));

            let held = matches!(reply.outcome, Outcome::Sat(_));
            assert_eq!(
                held, should_hold,
                "{claim} should have been {should_hold}, Z3 said {held}"
            );
        }
    }

    #[test]
    fn a_whole_document() {
        let rendered = document(&[("x", 0.0, 10.0)], &["x > 4"]);
        // Not newline-terminated, so the golden below keeps its own.
        let preamble = prelude();

        assert_eq!(rendered.untranslated, Vec::<usize>::new());
        assert_eq!(
            rendered.text,
            format!(
                "(set-option :produce-unsat-cores true)\n\
                 (set-logic QF_NIRA)\n{preamble}\n\
                 (declare-const |x| Real)\n\
                 (assert (and (>= |x| 0.0) (<= |x| 10.0)))\n\
                 ; x > 4\n\
                 (assert (! (> |x| 4.0) :named c0))\n\
                 (check-sat)\n(get-model)\n"
            )
        );
    }

    /// Balance used to be a property of the `Sexp` type; now it is a property
    /// of five template arms each closing what it opens, and this is the test
    /// that says so over every document the tests above build.
    #[test]
    fn every_document_is_balanced() {
        let documents = [
            document(&[("x", 0.0, 10.0)], &["x > 4"]),
            document(
                &[("a", 0.0, 1.0), ("b", 0.0, 1.0), ("c", 0.0, 1.0)],
                &["sum(1, 3, i -> var[i]) > 2", "a / b + a / c > 2"],
            ),
            document(
                &[("x", 0.0, 10.0), ("y", -10.0, 10.0)],
                &[
                    "var a = x * 2; var b = a + 1; b > 3",
                    "y > cbrt(x)",
                    "y > sqrt(x)",
                    "x == pi +/- 0.001",
                ],
            ),
            document(&[("x", 0.1, 10.0)], &["sin(x) <= 0", "3 > log(x, 2)"]),
        ];
        for rendered in documents {
            let mut depth = 0i32;
            for character in rendered.text.chars() {
                match character {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    _ => {}
                }
                assert!(depth >= 0, "closed too early:\n{}", rendered.text);
            }
            assert_eq!(depth, 0, "left open:\n{}", rendered.text);
        }
    }

    /// A quoted symbol accepts anything but `|` and `\`, neither of which
    /// babel's lexer admits, so every identifier is quoted rather than judged.
    #[test]
    fn a_symbol_is_quoted_whatever_it_contains() {
        assert_eq!(symbol("λ"), "|λ|");
        assert_eq!(symbol("x1"), "|x1|");
        assert_eq!(symbol("变量"), "|变量|");
    }

    /// An expression's source may contain newlines. A comment runs to the end of its
    /// line, so a newline reaching the output would turn the rest of the
    /// source into commands; the whole source stays on the one comment line
    /// and the assertion follows it intact.
    #[test]
    fn a_multi_line_source_is_one_comment_line() {
        let rendered = document(
            &[("x", 0.0, 10.0)],
            &["var a = x * 2;\nvar b = a + 1;\n  b > 3"],
        );
        let comment_lines: Vec<&str> = rendered
            .text
            .lines()
            .filter(|line| line.starts_with("; "))
            .collect();
        assert_eq!(
            comment_lines,
            vec!["; var a = x * 2; var b = a + 1; b > 3"],
            "{}",
            rendered.text
        );
        assert!(rendered.text.contains(":named c0)"), "{}", rendered.text);
    }

    #[test]
    fn a_comparison_is_emitted_as_a_comparison() {
        // It used to arrive as the residual `(< (- 4.0 |x|) 0.0)`, because the
        // front end had already flattened it on the evaluator's behalf. The
        // solver now gets what the author wrote.
        let rendered = document(&[("x", 0.0, 10.0)], &["x > 4"]);
        assert!(
            rendered.text.contains("(> |x| 4.0)"),
            "expected a plain comparison:
{}",
            rendered.text
        );
    }

    #[test]
    fn strictness_is_native_and_no_denormal_exists_to_leak() {
        // `eval` marks `<` by adding `f64::MIN_POSITIVE`, relying on rounding
        // that real arithmetic does not do. This used to be *the emitter's*
        // problem: recognise that marker and undo it, or put ~310 digits in the
        // document and still mean the wrong thing. The epsilon never reaches
        // here now, because strictness is the relation.
        let strict = document(&[("x", 0.0, 10.0)], &["x > 4"]);
        let loose = document(&[("x", 0.0, 10.0)], &["x >= 4"]);

        assert!(strict.text.contains("(assert (! (> |x| 4.0) :named c0))"));
        assert!(loose.text.contains("(assert (! (>= |x| 4.0) :named c0))"));
        assert!(
            !strict.text.contains("0.0000000000"),
            "a denormal reached the document: {}",
            strict.text
        );
    }

    #[test]
    fn literals_carry_a_point_and_negatives_are_applications() {
        assert_eq!(render(1.0).as_deref(), Some("1.0"));
        assert_eq!(render(0.0001).as_deref(), Some("0.0001"));
        // Not `-1.5`: in SMT-LIB the minus is an operator, not part of a literal.
        assert_eq!(render(-1.5).as_deref(), Some("(- 1.5)"));
        assert_eq!(real(f64::NAN), None);
        assert_eq!(real(f64::INFINITY), None);
    }

    #[test]
    fn extreme_magnitudes_avoid_exponent_notation() {
        // `1e300` is not an SMT-LIB `Real` literal, so it has to be written out.
        for value in [1e300, 1e-300, f64::MIN_POSITIVE, f64::MAX] {
            let rendered = render(value).expect("finite values render");
            assert!(
                !rendered.contains('e') && !rendered.contains('E'),
                "{value} rendered with an exponent: {rendered}"
            );
            assert_eq!(
                rendered.parse::<f64>(),
                Ok(value),
                "{value} did not round-trip"
            );
        }
    }

    #[test]
    fn an_unrolled_aggregate_becomes_one_n_ary_application() {
        // Why `Kind::Fold` is n-ary: it maps onto SMT-LIB's `(+ a b c)` with no
        // flattening pass in between. Also the case that pins `var[i]` resolving
        // against the schema — this expression names no symbols statically, so
        // resolving the subscript against its own symbol list would find nothing.
        let rendered = document(
            &[("a", 0.0, 1.0), ("b", 0.0, 1.0), ("c", 0.0, 1.0)],
            &["sum(1, 3, i -> var[i]) > 2"],
        );
        assert_eq!(rendered.untranslated, Vec::<usize>::new());
        assert!(
            rendered.text.contains("(+ |a| |b| |c|)"),
            "expected one n-ary sum, got:\n{}",
            rendered.text
        );
    }

    #[test]
    fn assignments_become_nested_lets() {
        let rendered = document(
            &[("x", 0.0, 10.0)],
            &["var a = x * 2; var b = a + 1; b > 3"],
        );
        assert_eq!(rendered.untranslated, Vec::<usize>::new());
        // Earlier bindings outermost, so the later ones can see them.
        let text = &rendered.text;
        let outer = text.find("(let ((l0").expect("first binding");
        let inner = text.find("(let ((l1").expect("second binding");
        assert!(outer < inner, "bindings nested the wrong way:\n{text}");
    }

    #[test]
    fn a_whole_power_is_emitted_as_multiplication() {
        // Z3's `^` would parse this only under `ALL`; under the logics the
        // document declares it is a silent parse failure. Multiplication is
        // what every logic accepts and what Z3 rewrites `^` into anyway.
        let rendered = document(&[("x", 0.0, 10.0)], &["x^3 > 2"]);
        assert_eq!(rendered.untranslated, Vec::<usize>::new());
        assert!(
            rendered.text.contains("(* |x| |x| |x|)"),
            "expected repeated multiplication, got:{}",
            rendered.text
        );
    }

    #[test]
    fn a_negative_power_pins_its_base_away_from_zero() {
        // The bug that went out with the old `smtlib::power`. It rendered
        // `x^-2` as `(/ 1.0 (* x x))` with no guard, so a solver could satisfy
        // the constraint through `x = 0` — Z3's division is total, and
        // `(^ x -1) = 0` really is sat there. The reciprocal takes the same
        // path every division does and picks up the guard with it.
        let rendered = document(&[("x", 0.0, 10.0)], &["x^-2 < 1"]);
        assert_eq!(rendered.untranslated, Vec::<usize>::new());
        assert!(
            rendered.text.contains("(assert (not (= (* |x| |x|) 0.0)))"),
            "no divisor guard on a negative power:{}",
            rendered.text
        );
    }

    #[test]
    fn what_cannot_be_expressed_is_reported_not_dropped() {
        // The JVM version dropped these and returned points that did not satisfy
        // them. Each must appear in `untranslated`, and none may produce an
        // assertion.
        for source in [
            // A logarithm whose *base* varies. `log(a, u)` inverts when the
            // base is constant, which is not this.
            "3 > log(x, 2)",
            "sin(x) <= 0",
            "x^x > 2",
            // The variable appears inside a transcendental and outside it.
            // Nothing short of causalization touches this.
            "x > sin(ln(cos(2.1^x)))",
        ] {
            let rendered = document(&[("x", 0.1, 10.0)], &[source]);
            assert_eq!(
                rendered.untranslated,
                vec![0],
                "{source:?} should have been reported as untranslated"
            );
            // Only a translated constraint is named, so this catches a stray
            // assertion without tripping over the side conditions.
            assert!(
                !rendered.text.contains(":named"),
                "{source:?} produced an assertion anyway:\n{}",
                rendered.text
            );
        }
    }

    #[test]
    fn translatable_and_untranslatable_constraints_coexist() {
        // A document is still worth emitting when only part of it can be
        // expressed. It just has to say which part.
        let rendered = document(&[("x", 0.1, 10.0)], &["x > 4", "sin(x) <= 0", "x < 9"]);
        assert_eq!(rendered.untranslated, vec![1]);
        // Named `c0` and `c2`; the gap at `c1` is what keeps an unsat core
        // pointing at the right constraint.
        assert!(rendered.text.contains(":named c0)"));
        assert!(!rendered.text.contains(":named c1)"));
        assert!(rendered.text.contains(":named c2)"));
    }

    #[test]
    fn a_division_pins_its_divisor_away_from_zero() {
        // The bug this exists for. Without the guard, Z3 satisfied
        // `x2 == x1 + x2/2 - x3/x4` by setting every variable to zero and
        // reading `0/0` as whatever it liked; babel evaluates that to NaN and
        // throws the point away.
        let rendered = document(&[("a", 0.0, 1.0), ("b", 0.0, 1.0)], &["a / b > 2"]);
        assert_eq!(rendered.untranslated, Vec::<usize>::new());
        assert!(
            rendered.text.contains("(assert (not (= |b| 0.0)))"),
            "no divisor guard:\n{}",
            rendered.text
        );
    }

    #[test]
    fn a_modulo_pins_its_divisor_away_from_zero_too() {
        // Same hazard, same guard. `expr_rem` divides internally, so a
        // symbolic divisor is one a solver could otherwise drive to zero and
        // satisfy the constraint through — and `cvg_pools` has precisely that
        // case, in `modulo_with_a_symbolic_divisor`.
        let rendered = document(&[("a", 0.0, 10.0), ("b", 0.0, 10.0)], &["3 > a % b"]);
        assert_eq!(rendered.untranslated, Vec::<usize>::new());
        assert!(
            rendered.text.contains("(assert (not (= |b| 0.0)))"),
            "no divisor guard on `%`:\n{}",
            rendered.text
        );
        assert!(
            rendered.text.contains("expr_rem"),
            "`%` did not reach the helper:\n{}",
            rendered.text
        );
    }

    #[test]
    fn every_division_gets_its_own_guard() {
        let rendered = document(
            &[("a", 0.0, 1.0), ("b", 0.0, 1.0), ("c", 0.0, 1.0)],
            &["a / b + a / c > 2"],
        );
        assert!(rendered.text.contains("(assert (not (= |b| 0.0)))"));
        assert!(rendered.text.contains("(assert (not (= |c| 0.0)))"));
    }

    #[test]
    fn a_root_becomes_an_auxiliary_variable() {
        // QF_NRA has no `sqrt`, so `y >= 0 and y*y = x` stands in. The sign
        // constraint is also what makes `sqrt` of a negative unsatisfiable,
        // which is the right answer — babel gives NaN there.
        let rendered = document(&[("x", 0.0, 10.0), ("y", 0.0, 10.0)], &["y > sqrt(x)"]);
        assert_eq!(rendered.untranslated, Vec::<usize>::new());
        assert!(rendered.text.contains("(declare-const aux_0_0 Real)"));
        assert!(rendered.text.contains("(assert (>= aux_0_0 0.0))"));
        assert!(
            rendered
                .text
                .contains("(assert (= (* aux_0_0 aux_0_0) |x|))")
        );

        // A cube root is one-to-one over the reals, so it gets no sign
        // constraint — pinning it positive would lose the negative branch.
        let cubed = document(&[("x", -10.0, 10.0), ("y", -10.0, 10.0)], &["y > cbrt(x)"]);
        assert!(
            cubed
                .text
                .contains("(assert (= (* aux_0_0 aux_0_0 aux_0_0) |x|))")
        );
        assert!(!cubed.text.contains("(assert (>= aux_0_0 0.0))"));
    }

    #[test]
    fn auxiliary_names_do_not_collide_across_constraints() {
        // Two constraints each declaring `aux0` is a redeclaration, and the
        // solver rejects the entire document rather than just that line.
        let rendered = document(
            &[("x", 0.0, 10.0), ("y", 0.0, 10.0)],
            &["y > sqrt(x)", "x > sqrt(y)"],
        );
        assert_eq!(rendered.untranslated, Vec::<usize>::new());
        assert!(rendered.text.contains("(declare-const aux_0_0 Real)"));
        assert!(rendered.text.contains("(declare-const aux_1_0 Real)"));
        assert_eq!(rendered.text.matches("declare-const aux").count(), 2);
    }

    #[test]
    fn side_conditions_precede_the_assertion_that_needs_them() {
        // A declaration after its use is a parse error, and `Solver::from_string`
        // reports one by silently solving nothing.
        let rendered = document(&[("x", 0.0, 10.0), ("y", 0.0, 10.0)], &["y > sqrt(x)"]);
        let declaration = rendered
            .text
            .find("declare-const aux_0_0")
            .expect("declared");
        let usage = rendered.text.find(":named c0").expect("asserted");
        assert!(declaration < usage, "declared too late:\n{}", rendered.text);
    }

    #[test]
    fn the_logic_is_the_callers_to_choose() {
        let rendered = document(&[("x", 0.0, 10.0)], &["x > 4"]);
        assert!(
            rendered.text.contains("(set-logic QF_NIRA)"),
            "the default logic is not what it claims:
{}",
            rendered.text
        );

        let inputs = [InputVariable {
            name: "x".to_owned(),
            lower_bound: 0.0,
            upper_bound: 10.0,
        }];
        let constraints = [crate::parse("x > 4").expect("compiles")];
        let overridden =
            emit_away_from(&inputs, &constraints, &SmtLogic::named("QF_NRA"), &[], 0.0);
        assert!(overridden.text.contains("(set-logic QF_NRA)"));
        assert!(!overridden.text.contains("QF_NIRA"));
    }

    #[test]
    fn a_comment_cannot_swallow_the_command_after_it() {
        // The one case where a missing newline is a bug rather than a
        // formatting quibble: `;` runs to end of line. Two ways it could go
        // wrong — the renderer forgetting the separator, and a source string
        // that carries its own newlines out of the comment and into the
        // command stream. Babel statements are newline-legal, so the second is
        // reachable from user input.
        let source = "var a = 4;
  return x
 > a";
        let inputs = [InputVariable {
            name: "x".to_owned(),
            lower_bound: 0.0,
            upper_bound: 10.0,
        }];
        let constraints = [crate::parse(source).expect("compiles")];
        let rendered = emit_away_from(&inputs, &constraints, &SmtLogic::default(), &[], 0.0);

        for line in rendered.text.lines() {
            assert!(
                !(line.starts_with(';') && line.contains('(')),
                "a command ended up inside a comment:
{line}"
            );
        }
        assert_eq!(
            rendered.text.lines().filter(|l| l.starts_with(';')).count(),
            1,
            "the source broke across more than one comment line:
{}",
            rendered.text
        );
        assert!(
            rendered.text.contains(":named"),
            "the assertion after the comment did not survive:
{}",
            rendered.text
        );
    }

    /// Each way of being untranslatable reports itself as the right one.
    ///
    /// The reason reaches a caller only through a `tracing` line, which nothing
    /// asserts on, so without this the messages could drift into nonsense and
    /// every test would stay green.
    #[test]
    fn a_refusal_says_which_construct_defeated_it() {
        let inputs = [
            InputVariable {
                name: "x".to_owned(),
                lower_bound: 0.1,
                upper_bound: 10.0,
            },
            InputVariable {
                name: "n".to_owned(),
                lower_bound: 1.0,
                upper_bound: 4.0,
            },
        ];

        for (source, expected) in [
            ("sin(x) <= 0", Refusal::Transcendental("sin")),
            ("cos(x) <= 0", Refusal::Transcendental("cos")),
            ("tanh(x) < 4", Refusal::Transcendental("tanh")),
            ("3 > log(x, 2)", Refusal::Logarithm),
            ("x ^ x > 2", Refusal::RealExponent),
            ("x > 2 ^ n", Refusal::RealExponent),
            ("x ^ 2.5 > 2", Refusal::RealExponent),
            // Past the cap a chain of multiplications is the wrong shape for a
            // solver too, so the cap is the emitter's as much as the tape's.
            ("x ^ 65 > 2", Refusal::RealExponent),
            // Still computed, and rightly refused: `n` is a variable, so which
            // one this reads depends on the point.
            ("var[n] > 2", Refusal::ComputedSubscript),
            ("x + 1", Refusal::NotABooleanExpression),
        ] {
            let expression = crate::parse(source).expect("should compile");
            let Err(reason) = translate(&expression, 0, &inputs) else {
                panic!("{source:?} was expected to be untranslatable");
            };
            assert_eq!(reason, expected, "{source:?}");
        }
    }

    /// What the emitter still cannot express, and which operators are implicated.
    ///
    /// Run it for the report: `cargo test --lib residue -- --nocapture`.
    ///
    /// This is the measurement wave 3 opens with, kept as a test so it cannot go
    /// stale. The assertion at the bottom is a **ratchet**: the untranslatable
    /// set is pinned, so teaching the emitter something new fails here and makes
    /// somebody update the record deliberately.
    ///
    /// The *why* is derived rather than declared. Nothing here knows which
    /// operators `translate` refuses — it collects the operators in every
    /// constraint, then subtracts the ones appearing in something translatable.
    /// What is left cannot be expressible, and no list needs maintaining twice.
    #[test]
    fn residue_what_the_emitter_still_cannot_express() {
        // Every distinct constraint shape in the CVG corpus, from `cvg_pools.rs`
        // and `cvg_benchmarks.rs`. The thirty P118 rows are one shape and appear
        // once; placeholders are given concrete values.
        const CORPUS: &[&str] = &[
            // -- plain arithmetic and comparison
            "x > 8",
            "x < 2",
            "x1 < x2",
            "x1 + x2 > x3",
            "0 > -x4+x1-7",
            "x > 10.5",
            "x > 0.0",
            // -- equalities with a tolerance
            "x1 == x2 +/- 0.1",
            "x2 == x1 + 1/2*x2 - x3 / x4 +/- 0.00001",
            "1.5 == var[1] + var[2] +/- 0.001",
            "(x + 2) * (x - 1) == 0 +/- 1.0",
            // -- constants
            "x1 == pi +/- 0.001",
            "x2 == e +/- 0.001",
            // -- powers
            "x1 == x2^3 +/- 0.0001",
            "20 > 2^x5",
            // -- roots, absolute value, sign
            "x1 == sqrt(x2) +/- 0.0001",
            "x3 == cbrt(x4) +/- 0.0001",
            "abs(x1) == 1 +/- 0.001",
            "x2 == sgn(x1) +/- 0.001",
            // -- integer-ish
            "x1 > floor(x2)",
            "x3 > ceil(x4) + floor(x4)",
            "x1 % 3.0 >= 2",
            "3 > 10 % x1",
            "x3 == x4 % 4.5 +/- 0.0001",
            // -- logarithms
            "2 < ln(x1)",
            "3 > log(2, x)",
            // -- trigonometry
            "sin(x1) <= 0",
            "y == sin(x) +/- 0.000001",
            "y > sin(theta)",
            "y < sin(x*pi)",
            "y > 1.1*sin(x*pi-0.5)",
            "x1 > sin(ln(cos(2.1^x1)))",
        ];

        let names: Vec<String> = ["x", "y", "theta"]
            .iter()
            .map(|s| (*s).to_owned())
            .chain((1..=15).map(|i| format!("x{i}")))
            .collect();
        let inputs: Vec<InputVariable> = names
            .iter()
            .map(|name| InputVariable {
                name: name.clone(),
                lower_bound: 0.1,
                upper_bound: 10.0,
            })
            .collect();

        let mut translated: Vec<(&str, Vec<String>)> = Vec::new();
        let mut refused: Vec<(&str, Vec<String>, Refusal)> = Vec::new();

        for source in CORPUS {
            let expression = crate::parse(source)
                .unwrap_or_else(|e| panic!("{source:?} did not compile: {:#?}", e.problems));
            let rendered = emit_away_from(
                &inputs,
                std::slice::from_ref(&expression),
                &SmtLogic::default(),
                &[],
                0.0,
            );
            let mut operators = Vec::new();
            collect_operators(&expression.program.body, &mut operators);
            operators.sort_unstable();
            operators.dedup();

            if rendered.untranslated.is_empty() {
                translated.push((source, operators));
            } else {
                // Asked again for the reason, which `Document` reports through
                // `tracing` rather than carrying. Two independent accounts of
                // the same refusal: the emitter's own, and the operator set
                // derived below. They should agree.
                let Err(reason) = translate(&expression, 0, &inputs) else {
                    panic!("{source:?} is in `untranslated` yet translates");
                };
                refused.push((source, operators, reason));
            }
        }

        // Anything appearing in something translatable is not the reason
        // anything else was refused.
        let expressible: std::collections::BTreeSet<&str> = translated
            .iter()
            .flat_map(|(_, ops)| ops.iter().map(String::as_str))
            .collect();

        println!();
        println!(
            "what the emitter cannot express, out of {} corpus shapes",
            CORPUS.len()
        );
        println!("{:-<78}", "");
        for (source, operators, reason) in &refused {
            let blamed: Vec<&str> = operators
                .iter()
                .map(String::as_str)
                .filter(|op| !expressible.contains(op))
                .collect();
            // `Debug` here rather than `Display`: the prose belongs in a log
            // line, where there is one of them, not in a column.
            println!("{source:<32} {:<14} {reason:?}", blamed.join(" "));
        }
        println!("{:-<78}", "");
        println!(
            "{} of {} translate; {} do not.",
            translated.len(),
            CORPUS.len(),
            refused.len()
        );

        let mut culprits: Vec<&str> = refused
            .iter()
            .flat_map(|(_, ops, _)| ops.iter().map(String::as_str))
            .filter(|op| !expressible.contains(op))
            .collect();
        culprits.sort_unstable();
        culprits.dedup();
        println!("operators implicated: {}", culprits.join(" "));
        println!();

        let outstanding: Vec<&str> = refused.iter().map(|(source, ..)| *source).collect();
        assert_eq!(
            outstanding,
            vec![
                "sin(x1) <= 0",
                "y == sin(x) +/- 0.000001",
                "y > sin(theta)",
                "y < sin(x*pi)",
                "y > 1.1*sin(x*pi-0.5)",
                "x1 > sin(ln(cos(2.1^x1)))",
            ],
            "the untranslatable set moved — update this list and `docs/todo.md` together"
        );
    }

    /// Every operator appearing anywhere in a block, by name.
    fn collect_operators(block: &Block, into: &mut Vec<String>) {
        for assignment in &block.assignments {
            collect_from(&assignment.value, into);
        }
        collect_from(&block.result, into);
    }

    fn collect_from(node: &Expr, into: &mut Vec<String>) {
        match &node.kind {
            Kind::Unary { op, arg } => {
                into.push(format!("{op:?}"));
                collect_from(arg, into);
            }
            Kind::Binary { op, lhs, rhs } => {
                into.push(format!("{op:?}"));
                collect_from(lhs, into);
                collect_from(rhs, into);
            }
            Kind::DynamicIndex(index) => {
                into.push("DynamicIndex".to_owned());
                collect_from(index, into);
            }
            Kind::And { terms } => {
                into.push("And".to_owned());
                for term in terms {
                    collect_from(term, into);
                }
            }
            Kind::Fold { terms, .. } => {
                into.push("Fold".to_owned());
                for term in terms {
                    collect_from(term, into);
                }
            }
            Kind::Block(block) => collect_operators(block, into),
            Kind::Aggregate { .. } => {
                unreachable!("`unroll_aggregates` leaves no aggregate for a backend to see")
            }
            Kind::Compare { op, lhs, rhs } => {
                into.push(format!("{op:?}"));
                collect_from(lhs, into);
                collect_from(rhs, into);
            }
            Kind::NearEq { lhs, rhs, .. } => {
                into.push("NearEq".to_owned());
                collect_from(lhs, into);
                collect_from(rhs, into);
            }
            Kind::Literal(_) | Kind::Global(_) | Kind::Local(_) => {}
        }
    }

    #[test]
    fn core_names_round_trip() {
        for index in [0usize, 1, 7, 29] {
            assert_eq!(core_index(&core_name(index)), Some(index));
        }
        assert_eq!(core_index("not-a-core-name"), None);
    }

    #[test]
    fn unicode_names_survive_as_quoted_symbols() {
        let rendered = document(&[("λ", 0.0, 1.0)], &["λ > 0.5"]);
        assert!(rendered.text.contains("(declare-const |λ| Real)"));
    }

    #[test]
    fn the_reds_this_is_meant_to_unlock() {
        // The `cvg_pools` cases that no amount of sampling or walking will
        // reach. All of them are expressible now: `%`, `floor` and `ceil` were
        // the last holdouts and they are encodings rather than theory.
        for source in [
            // Inverted rather than encoded: `ln(x1) > 2` becomes `x1 > e^2`,
            // `2^x5 < 20` becomes `x5 < log2(20)`, and `log(2, x1) < 3` becomes
            // `x1 < 8`. Z3 is never asked about a logarithm — as well, since it
            // has none.
            "2 < ln(x1)",
            "20 > 2^x5",
            "3 > log(2, x1)",
            "x1 % 3.0 >= 2",
            "x3 == x4 % 4.5 +/- 0.0001",
            "x1 > floor(x2)",
            "x3 > ceil(x4) + floor(x4)",
            "x2 == x1 + 1/2*x2 - x3 / x4 +/- 0.00001",
            "x1 == x2^3 +/- 0.0001",
            "abs(x1) == 1 +/- 0.001",
            "x1 == pi +/- 0.001",
            "1.5 == var[1] + var[2] +/- 0.001",
            "x1 == sqrt(x2) +/- 0.0001",
            "x3 == cbrt(x4) +/- 0.0001",
        ] {
            let rendered = document(
                &[
                    ("x1", 0.0, 10.0),
                    ("x2", 0.0, 10.0),
                    ("x3", 0.0, 10.0),
                    ("x4", 0.0, 10.0),
                ],
                &[source],
            );
            assert_eq!(
                rendered.untranslated,
                Vec::<usize>::new(),
                "{source:?} should be expressible in QF_NIRA, but:\n{}",
                rendered.text
            );
        }
    }
}
