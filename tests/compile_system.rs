//! `compile_system`: named expressions compiled together, as nodes with edges.
//!
//! The spec is `docs/compile-system.md`; its acceptance table is this file, one
//! test per row. Everything here is what a per-expression `compile` cannot see:
//! a subscript that walked past the inputs, a self-reference, a cycle, a
//! constraint used as a value, and which outputs are cheap *transitively*.
//! `tests/dependency_diagram.rs` is the worked example of putting the nodes to
//! use; this file is the contract.

use faer::Mat;
use sojourn::diagnostics::{EvaluationFailure, ProblemKind, Span, SystemCompilationFailure};
use sojourn::{CompiledNode, Symbol, compile_system};

const INPUTS: [&str; 3] = ["x1", "x2", "x3"];
const EXTERNALS: [&str; 1] = ["ext"];

fn input(name: &str) -> Symbol {
    Symbol::Input(name.to_owned())
}

fn external(name: &str) -> Symbol {
    Symbol::External(name.to_owned())
}

fn output(name: &str) -> Symbol {
    Symbol::Output(name.to_owned())
}

/// Compiles under the fixture's inputs and external, expecting success.
fn nodes(expressions: &[(&str, &str)]) -> Vec<CompiledNode> {
    compile_system(&INPUTS, &EXTERNALS, expressions)
        .unwrap_or_else(|failure| panic!("expected {expressions:?} to compile:\n{failure}"))
}

/// Compiles under the fixture's inputs and external, expecting failure.
fn failure(expressions: &[(&str, &str)]) -> SystemCompilationFailure {
    match compile_system(&INPUTS, &EXTERNALS, expressions) {
        Ok(nodes) => panic!(
            "expected {expressions:?} to fail, got nodes {:?}",
            nodes.iter().map(CompiledNode::name).collect::<Vec<_>>()
        ),
        Err(failure) => failure,
    }
}

fn node<'a>(nodes: &'a [CompiledNode], name: &str) -> &'a CompiledNode {
    nodes
        .iter()
        .find(|node| node.name() == name)
        .unwrap_or_else(|| panic!("no node named {name}"))
}

// ---- binding, by kind and in canonical order ----

#[test]
fn an_expression_over_inputs_reads_them_in_declaration_order() {
    let nodes = nodes(&[("f1", "x2 + x1")]);
    assert_eq!(node(&nodes, "f1").reads(), [input("x1"), input("x2")]);
}

#[test]
fn an_output_and_an_external_are_read_by_kind_externals_first() {
    let nodes = nodes(&[("f1", "x1 + 10"), ("f3", "f1 * ext")]);
    assert_eq!(node(&nodes, "f3").reads(), [external("ext"), output("f1")]);
}

#[test]
fn a_constraint_over_an_output_reads_inputs_then_outputs() {
    let nodes = nodes(&[("f1", "x1 + 10"), ("c1", "f1 < x1")]);
    let c1 = node(&nodes, "c1");
    assert_eq!(c1.reads(), [input("x1"), output("f1")]);
    assert!(c1.expression().is_constraint());
}

#[test]
fn an_unrolled_aggregate_reads_exactly_the_inputs_it_subscripts() {
    let nodes = nodes(&[("s", "sum(1, 3, i -> var[i]^2)")]);
    let s = node(&nodes, "s");
    assert_eq!(s.reads(), [input("x1"), input("x2"), input("x3")]);
    assert!(!s.expression().uses_dynamic_lookup());
}

#[test]
fn reads_is_the_row_layout_eval_takes() {
    let nodes = nodes(&[("f1", "x1 + 10"), ("f3", "f1 * ext")]);
    let f3 = node(&nodes, "f3");
    // Rows in `reads()` order: ext, then f1.
    let rows = Mat::from_fn(f3.reads().len(), 2, |row, column| match (row, column) {
        (0, c) => 2.0 + c as f64, // ext
        (1, _) => 11.0,           // f1
        _ => unreachable!(),
    });
    let values = f3.expression().eval(rows.as_ref()).expect("evaluates");
    assert_eq!(values[0], 22.0);
    assert_eq!(values[1], 33.0);

    let too_wide = Mat::zeros(f3.reads().len() + 1, 1);
    assert!(matches!(
        f3.expression().eval(too_wide.as_ref()),
        Err(EvaluationFailure::RowWidthMismatch {
            expected: 2,
            actual: 3
        })
    ));
}

// ---- subscripts index the inputs, and only the inputs ----

#[test]
fn a_literal_subscript_past_the_inputs_is_refused_at_the_subscript() {
    let failure = failure(&[("f1", "x1 + 10"), ("f4", "var[4]")]);
    let [named] = failure.problems.as_slice() else {
        panic!("expected one problem, got {:#?}", failure.problems);
    };
    assert_eq!(named.name, "f4");
    assert_eq!(
        named.problem.kind,
        ProblemKind::DynamicIndexOutOfBounds {
            requested_1index: 4,
            available: 3,
        }
    );
    assert_eq!(named.problem.span, Span::new(4, 5));
}

#[test]
fn a_computed_subscript_reads_every_input_and_cannot_reach_past_them() {
    let nodes = nodes(&[("f1", "x1 + 10"), ("f5", "var[floor(x1)] + f1")]);
    let f5 = node(&nodes, "f5");
    assert_eq!(
        f5.reads(),
        [input("x1"), input("x2"), input("x3"), output("f1")]
    );
    assert!(f5.expression().uses_dynamic_lookup());

    // x1 = 2 names x2: fine.
    let row = Mat::from_fn(4, 1, |row, _| [2.0, 5.0, 7.0, 12.0][row]);
    let values = f5.expression().eval(row.as_ref()).expect("in range");
    assert_eq!(values[0], 17.0);

    // x1 = 4 names the fourth row, which holds f1 — an output, not an input.
    // The gather is bounded by the inputs, so this is a fault, not f1 + f1.
    let row = Mat::from_fn(4, 1, |row, _| [4.0, 5.0, 7.0, 12.0][row]);
    match f5.expression().eval(row.as_ref()) {
        Err(EvaluationFailure::Runtime(problem)) => {
            assert_eq!(
                problem.problem.kind,
                ProblemKind::DynamicIndexOutOfBounds {
                    requested_1index: 4,
                    available: 3,
                }
            );
            assert_eq!(problem.problem.span, Span::new(4, 13));
        }
        other => panic!("expected an out-of-bounds fault, got {other:?}"),
    }
}

// ---- the graph ----

#[test]
fn self_reference_is_a_cycle_of_one() {
    let failure = failure(&[("h", "h + 1")]);
    let [named] = failure.problems.as_slice() else {
        panic!("expected one problem, got {:#?}", failure.problems);
    };
    assert_eq!(named.name, "h");
    assert_eq!(
        named.problem.kind,
        ProblemKind::Cycle {
            chain: vec!["h".to_owned()],
        }
    );
    assert_eq!(named.problem.span, Span::new(0, 1));
}

#[test]
fn a_cycle_is_reported_once_with_its_chain_from_the_closing_expression() {
    let failure = failure(&[("g1", "g2 + 1"), ("g2", "g1 + 1")]);
    let [named] = failure.problems.as_slice() else {
        panic!("expected one problem, got {:#?}", failure.problems);
    };
    // Walked in declaration order, g1 -> g2 -> g1: g2 is where it closes.
    assert_eq!(named.name, "g2");
    assert_eq!(
        named.problem.kind,
        ProblemKind::Cycle {
            chain: vec!["g2".to_owned(), "g1".to_owned()],
        }
    );
    assert_eq!(named.problem.source, "g1 + 1");
    assert_eq!(named.problem.span, Span::new(0, 2));
}

#[test]
fn a_constraint_is_not_a_value() {
    let failure = failure(&[("c2", "x1 + x2 > 0.5"), ("c3", "c2 * 2 > x1")]);
    let [named] = failure.problems.as_slice() else {
        panic!("expected one problem, got {:#?}", failure.problems);
    };
    assert_eq!(named.name, "c3");
    assert_eq!(
        named.problem.kind,
        ProblemKind::ConstraintAsValue {
            name: "c2".to_owned(),
        }
    );
    assert_eq!(named.problem.span, Span::new(0, 2));
}

#[test]
fn an_input_redefined_as_an_output_is_a_duplicate() {
    let failure = failure(&[("x1", "x2")]);
    let [named] = failure.problems.as_slice() else {
        panic!("expected one problem, got {:#?}", failure.problems);
    };
    assert_eq!(named.name, "x1");
    assert_eq!(
        named.problem.kind,
        ProblemKind::Duplicate {
            name: "x1".to_owned(),
        }
    );
    assert_eq!(named.problem.source, "x2");
}

#[test]
fn an_output_defined_twice_is_a_duplicate_at_the_second() {
    let failure = failure(&[("f", "x1"), ("f", "x2")]);
    let [named] = failure.problems.as_slice() else {
        panic!("expected one problem, got {:#?}", failure.problems);
    };
    assert_eq!(named.name, "f");
    assert_eq!(
        named.problem.kind,
        ProblemKind::Duplicate {
            name: "f".to_owned(),
        }
    );
    assert_eq!(named.problem.source, "x2");
    assert_eq!(named.problem.span, Span::new(0, 2));
}

#[test]
fn a_clash_between_inputs_and_externals_is_named_without_a_source() {
    let failure = match compile_system(&["x1"], &["x1"], &[("f", "x1")]) {
        Ok(_) => panic!("x1 declared twice compiled"),
        Err(failure) => failure,
    };
    let [named] = failure.problems.as_slice() else {
        panic!("expected one problem, got {:#?}", failure.problems);
    };
    assert_eq!(named.name, "x1");
    assert_eq!(
        named.problem.kind,
        ProblemKind::Duplicate {
            name: "x1".to_owned(),
        }
    );
    assert_eq!(named.problem.source, "");
}

// ---- cheap is transitive ----

#[test]
fn cheap_means_the_closure_holds_no_external() {
    let nodes = nodes(&[
        ("f1", "x1 + 10"),
        ("f2", "f1 * 2"),
        ("f3", "f1 * ext"),
        ("c1", "x1 + x2 > 0.5"),
        ("c2", "f3 < 100"),
    ]);
    assert!(node(&nodes, "f1").is_cheap());
    assert!(
        node(&nodes, "f2").is_cheap(),
        "f2 reads only f1, which reads only x1"
    );
    assert!(!node(&nodes, "f3").is_cheap());
    assert!(node(&nodes, "c1").is_cheap());
    assert!(
        !node(&nodes, "c2").is_cheap(),
        "c2 reads f3, which reads ext"
    );
}

// ---- order ----

#[test]
fn nodes_come_back_in_an_evaluation_order() {
    let nodes = nodes(&[("f2", "f1 * 2"), ("f1", "x1")]);
    let names = nodes.iter().map(CompiledNode::name).collect::<Vec<_>>();
    assert_eq!(names, ["f1", "f2"]);
}

#[test]
fn a_declaration_order_that_already_evaluates_is_kept() {
    let nodes = nodes(&[
        ("c1", "x1 + x2 > 0.5"),
        ("f1", "x1 + 10"),
        ("f3", "f1 * ext"),
        ("c2", "f3 < 100"),
        ("f2", "f1 * 2"),
    ]);
    let names = nodes.iter().map(CompiledNode::name).collect::<Vec<_>>();
    assert_eq!(names, ["c1", "f1", "f3", "c2", "f2"]);
}

// ---- everything at once, each under its name ----

#[test]
fn an_empty_expression_is_named_by_its_output() {
    let failure = failure(&[("e1", "")]);
    let [named] = failure.problems.as_slice() else {
        panic!("expected one problem, got {:#?}", failure.problems);
    };
    assert_eq!(named.name, "e1");
    assert_eq!(named.problem.kind, ProblemKind::EmptyExpression);
}

#[test]
fn an_unbound_name_is_named_by_its_output() {
    let failure = failure(&[("e2", "x9 + 1")]);
    let [named] = failure.problems.as_slice() else {
        panic!("expected one problem, got {:#?}", failure.problems);
    };
    assert_eq!(named.name, "e2");
    assert_eq!(
        named.problem.kind,
        ProblemKind::Unbound {
            name: "x9".to_owned(),
        }
    );
    assert_eq!(named.problem.span, Span::new(0, 2));
}

#[test]
fn every_problem_is_reported_in_one_pass() {
    let failure = failure(&[
        ("e1", "x1 +"),
        ("f1", "x1 + 10"),
        ("e2", "x9 + 1"),
        ("g1", "g2 + 1"),
        ("g2", "g1 + 1"),
    ]);
    // A parse failure may be several diagnostics (ANTLR's recovery is not a
    // contract), so count names, not problems.
    let mut names = failure
        .problems
        .iter()
        .map(|named| named.name.as_str())
        .collect::<Vec<_>>();
    names.dedup();
    assert_eq!(names, ["e1", "e2", "g2"]);
    assert!(
        failure
            .problems
            .iter()
            .any(|named| matches!(named.problem.kind, ProblemKind::Syntax { .. }))
    );
    assert!(
        failure
            .problems
            .iter()
            .any(|named| matches!(named.problem.kind, ProblemKind::Cycle { .. }))
    );
}

#[test]
fn a_failure_renders_one_line_per_problem_under_its_name() {
    let failure = failure(&[("e2", "x9 + 1"), ("h", "h + 1")]);
    let rendered = failure.to_string();
    let lines = rendered.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 2, "{rendered}");
    assert!(lines[0].starts_with("e2: "), "{rendered}");
    assert!(lines[0].contains("x9"), "{rendered}");
    assert!(lines[1].starts_with("h: "), "{rendered}");
    assert!(lines[1].contains("h -> h"), "{rendered}");

    // The block form carries the caret under each.
    let block = format!("{failure:#}");
    assert!(block.contains("~~"), "{block}");
}
