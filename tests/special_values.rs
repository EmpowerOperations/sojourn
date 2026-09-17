//! Edge values, to the bit, and faults in batches.
//!
//! The corpus checks every operator at one ordinary input. This file checks
//! what happens at the edges: signed zeros through the operators that treat
//! them specially, the operations that produce NaN or an infinity from finite
//! inputs and must fault at their own span, and a faulting column planted in a
//! wide batch, which must be reported by its own kind, span and column index
//! for every kind of fault there is.
//!
//! Bits are compared where a value comparison would lie: `-0.0 == 0.0` is true
//! in `f64`, so a test that wants `-0.0` has to say so in bits.

mod common;

use faer::Mat;
use sojourn::diagnostics::{EvaluationFailure, ProblemKind, RuntimeProblem, Span};

fn value(source: &str, inputs: &[(&str, f64)]) -> f64 {
    common::eval_one(source, inputs).unwrap_or_else(|e| panic!("{source:?} at {inputs:?}: {e:?}"))
}

fn assert_bits(source: &str, inputs: &[(&str, f64)], expected: f64) {
    let got = value(source, inputs);
    assert_eq!(
        got.to_bits(),
        expected.to_bits(),
        "{source:?} at {inputs:?}: got {got:?}, expected {expected:?}"
    );
}

fn runtime_error(source: &str, inputs: &[(&str, f64)]) -> Box<RuntimeProblem> {
    match common::eval_one(source, inputs) {
        Err(EvaluationFailure::Runtime(problem)) => problem,
        other => panic!("{source:?} at {inputs:?}: expected a runtime problem, got {other:?}"),
    }
}

fn batch_error(source: &str, names: &[&str], columns: &[&[f64]]) -> Box<RuntimeProblem> {
    let compiled = sojourn::compile(source, names).expect("binds");
    let batch = Mat::from_fn(names.len(), columns.len(), |r, c| columns[c][r]);
    match compiled.eval(batch.as_ref()) {
        Err(EvaluationFailure::Runtime(problem)) => problem,
        other => panic!("{source:?}: expected a runtime problem, got {other:?}"),
    }
}

// ------------------------------------------------------------- signed zeros

/// Java's `Math.signum` returns the zero it was given, sign and all.
#[test]
fn signum_returns_the_zero_it_was_given() {
    assert_bits("sgn(x1)", &[("x1", 0.0)], 0.0);
    assert_bits("sgn(x1)", &[("x1", -0.0)], -0.0);
    assert_bits("sgn(x1)", &[("x1", -2.5)], -1.0);
    assert_bits("sgn(x1)", &[("x1", 1e-300)], 1.0);
}

#[test]
fn negation_and_absolute_value_flip_and_clear_the_sign_of_zero() {
    assert_bits("-x1", &[("x1", 0.0)], -0.0);
    assert_bits("-x1", &[("x1", -0.0)], 0.0);
    assert_bits("abs(x1)", &[("x1", -0.0)], 0.0);
    assert_bits("abs(x1)", &[("x1", 0.0)], 0.0);
}

/// `%` is a remainder, and a remainder takes the dividend's sign — including
/// a negative zero's.
#[test]
fn remainder_keeps_the_sign_of_the_dividend() {
    assert_bits("x1 % x2", &[("x1", -0.0), ("x2", 3.0)], -0.0);
    assert_bits("x1 % x2", &[("x1", -7.0), ("x2", 3.0)], -1.0);
    assert_bits("x1 % x2", &[("x1", 7.0), ("x2", -3.0)], 1.0);
}

/// Java orders the zeros: `Math.max(-0.0, 0.0)` is `0.0` and `Math.min` is
/// `-0.0`, whichever side each is on. Rust's `f64::max` leaves this
/// unspecified — and on the current toolchain answers differently when
/// constant-folded than at run time — which is why babel spells it out.
#[test]
fn max_and_min_order_the_signed_zeros_like_java() {
    for (a, b) in [(-0.0, 0.0), (0.0, -0.0)] {
        assert_bits("max(x1, x2)", &[("x1", a), ("x2", b)], 0.0);
        assert_bits("min(x1, x2)", &[("x1", a), ("x2", b)], -0.0);
    }
    assert_bits("max(x1, x2)", &[("x1", -0.0), ("x2", -0.0)], -0.0);
    assert_bits("min(x1, x2)", &[("x1", 0.0), ("x2", 0.0)], 0.0);
    assert_bits("max(x1, x2)", &[("x1", 2.0), ("x2", -3.0)], 2.0);
    assert_bits("min(x1, x2)", &[("x1", 2.0), ("x2", -3.0)], -3.0);
}

#[test]
fn rounding_and_roots_preserve_a_negative_zero() {
    assert_bits("ceil(x1)", &[("x1", -0.5)], -0.0);
    assert_bits("floor(x1)", &[("x1", -0.0)], -0.0);
    assert_bits("sqrt(x1)", &[("x1", -0.0)], -0.0);
    assert_bits("x1 * x2", &[("x1", -1.0), ("x2", 0.0)], -0.0);
}

// ------------------------------------ non-finite results from finite inputs

/// A source, the inputs that make it go non-finite, and the span of the
/// operation that does.
type NonFiniteCase = (&'static str, &'static [(&'static str, f64)], Span);

/// Each of these produces NaN or an infinity from ordinary inputs, and must
/// fault at its own span rather than let the value travel.
#[test]
fn an_operation_that_goes_non_finite_faults_at_its_own_span() {
    let cases: &[NonFiniteCase] = &[
        ("x1 % x2", &[("x1", 5.0), ("x2", 0.0)], Span::new(0, 7)),
        ("x1 / x2", &[("x1", 0.0), ("x2", 0.0)], Span::new(0, 7)),
        ("x1 / x2", &[("x1", 1.0), ("x2", 0.0)], Span::new(0, 7)),
        ("x1 ^ x2", &[("x1", -8.0), ("x2", 0.5)], Span::new(0, 7)),
        ("cot(x1)", &[("x1", 0.0)], Span::new(0, 7)),
        ("sqrt(x1)", &[("x1", -1.0)], Span::new(0, 8)),
        ("acos(x1)", &[("x1", 2.0)], Span::new(0, 8)),
        (
            "log(x1, x2)",
            &[("x1", 1.0), ("x2", 10.0)],
            Span::new(0, 11),
        ),
        ("1 + ln(x1) * 2", &[("x1", 0.0)], Span::new(4, 10)),
    ];
    for (source, inputs, span) in cases {
        let problem = runtime_error(source, inputs);
        assert!(
            matches!(problem.problem.kind, ProblemKind::NonFiniteValue { .. }),
            "{source:?}: {:?}",
            problem.problem.kind
        );
        assert_eq!(problem.problem.span, *span, "{source:?}");
    }
}

/// An unrolled aggregate is checked term by term: the division inside the
/// second term is the fault, not the sum.
#[test]
fn a_fault_inside_an_unrolled_term_names_the_term() {
    let problem = runtime_error("sum(1, 3, i -> x1 / (i - 2))", &[("x1", 4.0)]);
    assert_eq!(
        problem.problem.kind,
        ProblemKind::NonFiniteValue {
            value: f64::INFINITY
        }
    );
    assert_eq!(problem.problem.column_idx, 15);
}

// ----------------------------------------------------------- faults in batches

/// The planted column is the one reported, with its own value and span.
#[test]
fn a_non_finite_value_in_a_batch_names_its_column() {
    let problem = batch_error("ln(x1)", &["x1"], &[&[1.0], &[2.0], &[0.0], &[3.0], &[4.0]]);
    assert_eq!(problem.sample, Some(2));
    assert_eq!(
        problem.problem.kind,
        ProblemKind::NonFiniteValue {
            value: f64::NEG_INFINITY
        }
    );
    assert_eq!(problem.problem.span, Span::new(0, 6));
    assert_eq!(problem.parameters, vec![("x1".to_owned(), 0.0)]);
}

#[test]
fn an_out_of_range_subscript_in_a_batch_names_its_column() {
    let problem = batch_error(
        "var[floor(x1)] + x2",
        &["x1", "x2"],
        &[&[1.0, 0.0], &[2.0, 0.0], &[3.0, 0.0]],
    );
    assert_eq!(problem.sample, Some(2));
    assert_eq!(
        problem.problem.kind,
        ProblemKind::DynamicIndexOutOfBounds {
            requested_1index: 3,
            available: 2,
        }
    );
    // The subscript, not the whole `var[...]`.
    assert_eq!(problem.problem.span, Span::new(4, 13));
}

/// Two faulting columns: the lower index wins, whatever order the tile found
/// them in.
#[test]
fn the_lowest_faulting_column_is_the_one_reported() {
    let problem = batch_error(
        "sqrt(x1)",
        &["x1"],
        &[&[4.0], &[-1.0], &[9.0], &[-4.0], &[16.0]],
    );
    assert_eq!(problem.sample, Some(1));
    assert_eq!(problem.parameters, vec![("x1".to_owned(), -1.0)]);
}

/// A batch with no faults gives every column its own answer, so a batch of
/// many agrees with the same points evaluated one at a time.
#[test]
fn a_batch_agrees_with_one_at_a_time() {
    let source = "max(x1, x2) - min(x1, x2) + x1 % x2";
    let compiled = sojourn::compile(source, &["x1", "x2"]).unwrap();
    let points: Vec<[f64; 2]> = (0..300)
        .map(|i| [f64::from(i % 17) - 8.0, f64::from(i % 5) + 1.0])
        .collect();
    let batch = Mat::from_fn(2, points.len(), |r, c| points[c][r]);
    let residuals = compiled.eval(batch.as_ref()).unwrap();
    for (c, [x1, x2]) in points.iter().enumerate() {
        let single = common::eval_one(source, &[("x1", *x1), ("x2", *x2)]).unwrap();
        assert_eq!(residuals[c].to_bits(), single.to_bits(), "column {c}");
    }
}
