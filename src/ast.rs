//! Babel's abstract syntax tree.
//!
//! ANTLR 4 produces a *parse* tree and offers no way to rewrite it — the tree
//! rewriting that ANTLR 3 supported was deliberately removed, and Parr's
//! recommendation for v4 is to build your own model with a visitor. This module
//! is that model. Everything downstream of [`crate::frontend::parse`] works here, not on
//! ANTLR contexts.
//!
//! Two invariants shape the design:
//!
//! 1. **Symbols are resolved during lowering**, not at evaluation time. Names
//!    become [`GlobalId`] or [`LocalSlot`], so shadowing (`sum(1,3,x1 -> x1) + x1`)
//!    is settled structurally and evaluation needs no scope chain.
//! 2. **Every node carries a [`Span`]**, because diagnostics are reported against
//!    arbitrary sub-expressions at both compile time and run time.

use crate::diagnostics::Span;

/// Position of a variable in the bound [`Schema`](crate::Schema)'s declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GlobalId(u32);

impl GlobalId {
    /// Callers treat this as an opaque handle. That it happens to be a dense
    /// index into a `Vec` is [`Schema`](crate::Schema)'s business — the schema
    /// presents as a fast map and is free to change how it stores things.
    pub(crate) const fn from_index(index: u32) -> Self {
        Self(index)
    }

    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Position of a value in the current evaluation frame — a `var x = …` binding
/// or a lambda parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LocalSlot(u32);

impl LocalSlot {
    /// Slots are handed out monotonically during translation and never reused,
    /// so one flat frame serves the whole tree. Opaque for the same reason
    /// [`GlobalId`] is: that it indexes a `Vec` is the evaluator's business.
    pub(crate) const fn from_index(index: u32) -> Self {
        Self(index)
    }

    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A complete compiled expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub body: Block,
    /// Number of local slots the deepest frame requires. Lets evaluation
    /// allocate one flat frame instead of growing a scope chain.
    pub frame_size: u32,
}

/// `(statement ';')* returnStatement ';'?`
///
/// The grammar guarantees a trailing result expression, so this is a struct
/// with a required `result` rather than a list of statements that might not
/// produce a value.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub assignments: Vec<Assignment>,
    pub result: Expr,
}

/// `var <name> = <expr>`
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub slot: LocalSlot,
    pub value: Expr,
    pub span: Span,
}

/// An expression node: a [`Kind`] plus its source location.
///
/// The split follows `rustc_ast::Expr { kind, span, .. }` — it keeps `match`
/// arms free of span noise while guaranteeing every node has one.
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: Kind,
    pub span: Span,
}

impl Expr {
    #[must_use]
    pub fn new(kind: Kind, span: Span) -> Self {
        Self { kind, span }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    Literal(f64),

    /// A statically named variable, resolved against the schema.
    Global(GlobalId),
    /// A `var x = …` binding or a lambda parameter.
    Local(LocalSlot),
    /// `var[expr]` — a one-based index into schema declaration order, computed
    /// at run time. This is what makes [`crate::Schema`] ordered.
    DynamicIndex(Box<Expr>),

    Unary {
        op: UnaryOp,
        arg: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },

    /// `sum(lower, upper, param -> body)` / `prod(…)`.
    Aggregate {
        kind: AggregateKind,
        lower: Box<Expr>,
        upper: Box<Expr>,
        param: LocalSlot,
        body: Box<Block>,
    },

    /// An aggregate whose bounds were known at compile time, unrolled into its
    /// terms by [`crate::frontend::rewrite::unroll_aggregates`].
    ///
    /// N-ary rather than a chain of [`Kind::Binary`]: a thousand-term
    /// aggregate is one node deep rather than a thousand, which keeps the
    /// recursive passes — lowering, interval narrowing — off the stack limit,
    /// and a fold is what the runtime loop was anyway.
    ///
    /// Evaluated left-to-right from [`AggregateKind::identity`], which is what
    /// the runtime loop does — so unrolling cannot change a result. Rebalancing
    /// would, since `f64` addition is not associative.
    Fold {
        kind: AggregateKind,
        terms: Vec<Expr>,
    },

    /// A multi-statement lambda body used in expression position.
    Block(Box<Block>),

    // ---- the boolean variants ----
    //
    // These survive compilation. They used to be flattened into arithmetic by a
    // `rewrite_booleans` pass in the shared pipeline, which meant the `<= 0`
    // residual convention — *the evaluator's* convention — destroyed the
    // structure `cvg` needs before `cvg` could read it. Each backend lowers
    // them its own way now: `eval` computes a residual inline, interval
    // narrowing reads a comparison as a target interval.
    //
    // The grammar keeps them at the root of an expression and nowhere else:
    // `lambdaExpr` takes a `scalarBlock`, so a boolean cannot appear inside
    // arithmetic.
    /// `a < b`, `a >= b`, …
    Compare {
        op: CompareOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// `a == b +/- tolerance`.
    NearEq {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        tolerance: f64,
    },
    /// Every term holding at once.
    ///
    /// Not something the grammar produces — babel has no `and`. It exists
    /// because [`crate::frontend::rewrite::invert_monotone`] needs a conjunction
    /// for its domain guard: `ln(x) < 2` means `x < e^2` **and** `x > 0`.
    ///
    /// It used to build `max(residual_a, residual_b) <= 0` by hand, which meant
    /// the front end knew the residual convention. A variant it can emit
    /// without knowing costs a match arm in each backend and buys the
    /// separation outright.
    And {
        terms: Vec<Expr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Cos,
    Sin,
    Tan,
    Acos,
    Asin,
    Atan,
    Cosh,
    Sinh,
    Tanh,
    Cot,
    Ln,    // Natural logarithm (log base e) -- Babel's `ln`
    Log10, // Base-10 logarithm -- Babel's unary `log`
    Abs,
    Sqrt,
    Cbrt,
    Sqr,
    Cube,
    Ceil,
    Floor,
    Sgn,
}

impl UnaryOp {
    /// Maps a `unaryFunction` keyword to its operator.
    ///
    /// Lives here rather than in the front end because it constructs this type
    /// and has to stay in step with its variants.
    #[must_use]
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        Some(match keyword {
            "cos" => UnaryOp::Cos,
            "sin" => UnaryOp::Sin,
            "tan" => UnaryOp::Tan,
            "acos" => UnaryOp::Acos,
            "asin" => UnaryOp::Asin,
            "atan" => UnaryOp::Atan,
            "cosh" => UnaryOp::Cosh,
            "sinh" => UnaryOp::Sinh,
            "tanh" => UnaryOp::Tanh,
            "cot" => UnaryOp::Cot,
            // Babel renames Java's log/log10 to ln/log respectively.
            "ln" => UnaryOp::Ln,
            "log" => UnaryOp::Log10,
            "abs" => UnaryOp::Abs,
            "sqrt" => UnaryOp::Sqrt,
            "cbrt" => UnaryOp::Cbrt,
            "sqr" => UnaryOp::Sqr,
            "cube" => UnaryOp::Cube,
            "ceil" => UnaryOp::Ceil,
            "floor" => UnaryOp::Floor,
            "sgn" => UnaryOp::Sgn,
            _ => return None,
        })
    }
}

impl UnaryOp {
    /// Applies the operator. Lives here rather than in the evaluator because
    /// the meaning of an operator belongs with the operator, and constant
    /// folding needs it too.
    #[must_use]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Self::Negate => -x,
            Self::Cos => x.cos(),
            Self::Sin => x.sin(),
            Self::Tan => x.tan(),
            Self::Acos => x.acos(),
            Self::Asin => x.asin(),
            Self::Atan => x.atan(),
            Self::Cosh => x.cosh(),
            Self::Sinh => x.sinh(),
            Self::Tanh => x.tanh(),
            Self::Cot => 1.0 / x.tan(),
            Self::Ln => x.ln(),
            Self::Log10 => x.log10(),
            Self::Abs => x.abs(),
            Self::Sqrt => x.sqrt(),
            Self::Cbrt => x.cbrt(),
            Self::Sqr => x * x,
            Self::Cube => x * x * x,
            Self::Ceil => x.ceil(),
            Self::Floor => x.floor(),
            // Java's Math.signum returns +/-0.0 for +/-0.0 and NaN for NaN;
            // Rust's f64::signum returns 1.0 for +0.0 and -1.0 for -0.0 and NaN.
            // Babel's semantics are Java's, so preserve them.
            Self::Sgn => {
                if x == 0.0 || x.is_nan() {
                    x
                } else {
                    x.signum()
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    /// `a % b` — a *remainder*, not a modulo, and the distinction is not
    /// pedantry. The sign follows the dividend, so `-7 % 3` is -1; a modulo
    /// takes the sign of the divisor and would answer 2. Babel follows Java
    /// here, as `apply` does. The grammar's token is still `MOD` and the
    /// operator is still spelled `%`, because both are the JVM's and neither
    /// is ours to rename.
    Rem,
    Pow,
    Max,
    Min,
    /// `log(base, x)` — Babel's binary `log`.
    LogB,
}

impl BinaryOp {
    /// Maps a `binaryFunction` keyword to its operator. Only the three
    /// call-syntax functions; the infix operators come from the grammar's
    /// operator rules instead.
    #[must_use]
    pub fn from_function_keyword(keyword: &str) -> Option<Self> {
        Some(match keyword {
            "max" => BinaryOp::Max,
            "min" => BinaryOp::Min,
            "log" => BinaryOp::LogB,
            _ => return None,
        })
    }
}

impl BinaryOp {
    /// Applies the operator.
    #[must_use]
    pub fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
            Self::Div => a / b,
            // Rust's `%` on f64 is the truncated remainder, which is
            // exactly Java's. No adjustment needed.
            Self::Rem => a % b,
            Self::Pow => a.powf(b),
            // Java's Math.max/min propagate NaN and order the signed zeros;
            // Rust's f64::max/min discard NaN and, on this toolchain, answer
            // `max(-0.0, 0.0)` one way when constant-folded and the other way
            // at run time. Babel's semantics are Java's, spelled out.
            Self::Max => nan_or(a, b, java_max),
            Self::Min => nan_or(a, b, java_min),
            // log(base, x) == ln(x) / ln(base)
            Self::LogB => b.ln() / a.ln(),
        }
    }
}

fn nan_or(a: f64, b: f64, f: impl Fn(f64, f64) -> f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        f(a, b)
    }
}

/// `Math.max` for two non-NaN doubles: the larger, and on equal values the
/// one with the positive sign — which only differs from "either" for `-0.0`
/// against `0.0`, where Java answers `0.0`. Two equal values that are not
/// zeros are bitwise identical, so the sign rule cannot pick wrong there.
fn java_max(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else if b > a || a.is_sign_negative() {
        b
    } else {
        a
    }
}

/// `Math.min`: the mirror of [`java_max`], answering `-0.0` for the zeros.
fn java_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else if a < b || a.is_sign_negative() {
        a
    } else {
        b
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Lt,
    Lte,
    Gt,
    Gte,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateKind {
    Sum,
    Prod,
}

impl AggregateKind {
    /// The fold's identity: `0` for `sum`, `1` for `prod`.
    #[must_use]
    pub const fn identity(self) -> f64 {
        match self {
            Self::Sum => 0.0,
            Self::Prod => 1.0,
        }
    }

    /// Accumulates one term.
    #[must_use]
    pub fn combine(self, accumulated: f64, term: f64) -> f64 {
        match self {
            Self::Sum => accumulated + term,
            Self::Prod => accumulated * term,
        }
    }
}

/// The single place where an `f64` becomes an integer index.
///
/// Babel conflates integer and floating-point maths: `sum`/`prod` bounds and
/// `var[i]` subscripts are all `f64`. Rather than give indices their own type,
/// every conversion routes through here, so there is exactly one place that
/// decides what counts as an index.
///
/// Returns `None` for NaN, infinities, anything with a fractional part, and
/// anything beyond ±2^53 — past which `f64` cannot represent consecutive
/// integers, so "integral" stops meaning anything. Callers attach the
/// diagnostic, since only they know whether a failure is a bad bound or a bad
/// subscript.
///
/// Deliberately strict. The JVM implementation rounded, so `sum(1, 20/3, ...)`
/// silently became `sum(1, 7, ...)` and `var[1.7]` became `var[2]`.
#[must_use]
pub fn to_index(value: f64) -> Option<i64> {
    /// `2^53` — the largest magnitude at which every integer is representable.
    const LIMIT: f64 = 9_007_199_254_740_992.0;

    (value.is_finite() && value.fract() == 0.0 && value.abs() <= LIMIT).then_some(value as i64)
}

/// The largest whole exponent a backend lowers into repeated multiplication.
///
/// Past this a chain of multiplications is the wrong shape for a solver as
/// much as for the evaluator, and `powf` takes over.
pub(crate) const POWER_LIMIT: i64 = 64;

impl Expr {
    /// The whole exponent this node spells, if it is a literal whole number
    /// within [`POWER_LIMIT`]; `None` for anything a backend hands to `powf`.
    ///
    /// One rule shared by the tape, the WGSL kernel and interval narrowing,
    /// so the three agree on which powers are polynomials without a pass
    /// enforcing it. A whole power is sign-safe everywhere and has a root to
    /// narrow through; a real one is `exp(n * ln x)`, undefined for a
    /// negative base and without an inverse anything here will use.
    #[must_use]
    pub(crate) fn whole_exponent(&self) -> Option<i64> {
        match self.kind {
            Kind::Literal(value) => to_index(value).filter(|n| n.abs() <= POWER_LIMIT),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Expr, GlobalId, Kind};
    use crate::diagnostics::Span;

    #[test]
    fn a_whole_exponent_is_a_literal_integer_within_the_cap() {
        let exponent =
            |value: f64| Expr::new(Kind::Literal(value), Span::new(0, 1)).whole_exponent();
        assert_eq!(exponent(2.0), Some(2));
        assert_eq!(exponent(0.0), Some(0));
        assert_eq!(exponent(-3.0), Some(-3));
        assert_eq!(exponent(64.0), Some(64), "the cap is inclusive");
        assert_eq!(exponent(65.0), None);
        assert_eq!(exponent(2.5), None);
        assert_eq!(exponent(f64::NAN), None);

        let variable = Expr::new(Kind::Global(GlobalId::from_index(0)), Span::new(0, 1));
        assert_eq!(variable.whole_exponent(), None);
    }
}
