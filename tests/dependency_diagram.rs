//! A dependency diagram built from `compile_system`'s nodes: the worked
//! example for a caller that owns its own graph.
//!
//! Sojourn hands back one node per expression with its edges (`reads()`) and
//! nothing that runs them. This file is what the other side looks like: a
//! `Diagram` with three kinds of node — an input, a compiled expression, and
//! a *simulation* that sojourn knows only as an external name — wired by
//! name, walked in dependency order, and run over many design points from
//! one compilation. Readability over speed throughout; the point is that a
//! reader can see the whole mechanism in one screen.
//!
//! The one policy worth copying is *cheap first*: every expression whose
//! reads are known is evaluated before any simulation is dispatched, and a
//! cheap constraint that fails stops the walk with the simulation never run.
//! That is what `CompiledNode::is_cheap` exists for.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt;

use faer::Mat;
use sojourn::{CompiledNode, compile_system};

/// What a simulation does with the values it reads: the diagram's business,
/// not sojourn's.
type Simulate<'a> = Box<dyn Fn(&[f64]) -> f64 + 'a>;

/// One box on the diagram.
enum Node<'a> {
    /// A coordinate of the design vector; known before the walk starts.
    Input(&'a str),
    /// A babel expression, compiled once by `compile_system`.
    Expression(&'a CompiledNode),
    /// Something sojourn cannot evaluate — a solver, a mesh, a call to
    /// another process. Declared to `compile_system` as an external, so
    /// expressions may read it; what it reads is the diagram's business.
    Simulation {
        name: &'a str,
        reads: Vec<&'a str>,
        run: Simulate<'a>,
    },
}

impl Node<'_> {
    fn name(&self) -> &str {
        match self {
            Node::Input(name) | Node::Simulation { name, .. } => name,
            Node::Expression(node) => node.name(),
        }
    }

    /// The names this node needs before it can be evaluated.
    fn reads(&self) -> Vec<&str> {
        match self {
            Node::Input(_) => Vec::new(),
            Node::Expression(node) => node.reads().iter().map(|read| read.name()).collect(),
            Node::Simulation { reads, .. } => reads.clone(),
        }
    }
}

/// Nodes and the edges between them, drawn from what each node reads.
struct Diagram<'a> {
    nodes: Vec<Node<'a>>,
    /// `(from, to)` by position in `nodes`: `to` reads `from`.
    edges: Vec<(usize, usize)>,
}

/// How a walk over the diagram ended.
#[derive(Debug, PartialEq)]
enum Outcome {
    /// Every node evaluated; the value of each, by name.
    Complete(BTreeMap<String, f64>),
    /// A constraint decidable without a simulation failed, so none was run.
    /// Carries the constraint and what had been computed by then.
    Infeasible {
        constraint: String,
        known: BTreeMap<String, f64>,
    },
}

impl<'a> Diagram<'a> {
    fn new(inputs: &[&'a str], compiled: &'a [CompiledNode], simulations: Vec<Node<'a>>) -> Self {
        let mut nodes = Vec::new();
        nodes.extend(inputs.iter().map(|name| Node::Input(name)));
        nodes.extend(compiled.iter().map(Node::Expression));
        nodes.extend(simulations);

        let position = |name: &str| {
            nodes
                .iter()
                .position(|node| node.name() == name)
                .unwrap_or_else(|| panic!("{name} is read but nothing on the diagram produces it"))
        };
        let mut edges = Vec::new();
        for (to, node) in nodes.iter().enumerate() {
            for read in node.reads() {
                edges.push((position(read), to));
            }
        }
        Self { nodes, edges }
    }

    /// Walks the diagram for one design point: whatever is ready goes next,
    /// expressions before simulations, until everything has a value or a
    /// cheap constraint has said no.
    fn run(&self, design: &[f64]) -> Outcome {
        let mut known = BTreeMap::<String, f64>::new();
        for (node, value) in self.nodes.iter().zip(design) {
            let Node::Input(name) = node else {
                panic!("the design vector is longer than the inputs");
            };
            known.insert((*name).to_owned(), *value);
        }

        loop {
            let ready = |(index, node): (usize, &Node<'a>)| {
                !known.contains_key(node.name())
                    && self
                        .edges
                        .iter()
                        .filter(|(_, to)| *to == index)
                        .all(|(from, _)| known.contains_key(self.nodes[*from].name()))
            };
            let expression = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| matches!(node, Node::Expression(_)))
                .find(|candidate| ready(*candidate));
            let simulation = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| matches!(node, Node::Simulation { .. }))
                .find(|candidate| ready(*candidate));

            let (_, next) = match (expression, simulation) {
                (Some(expression), _) => expression,
                (None, Some(simulation)) => {
                    // Nothing cheap is left. Before paying for the
                    // simulation, ask whether a constraint already said no.
                    if let Some(constraint) = self.failed_constraint(&known) {
                        return Outcome::Infeasible { constraint, known };
                    }
                    simulation
                }
                (None, None) => return Outcome::Complete(known),
            };

            let arguments = next
                .reads()
                .iter()
                .map(|read| known[*read])
                .collect::<Vec<_>>();
            let value = match next {
                Node::Input(_) => unreachable!("inputs are known before the walk"),
                Node::Expression(node) => {
                    let column = Mat::from_fn(arguments.len(), 1, |row, _| arguments[row]);
                    node.expression().eval(column.as_ref()).expect("evaluates")[0]
                }
                Node::Simulation { run, .. } => run(&arguments),
            };
            known.insert(next.name().to_owned(), value);
        }
    }

    /// The first evaluated constraint whose residual says it does not hold.
    fn failed_constraint(&self, known: &BTreeMap<String, f64>) -> Option<String> {
        self.nodes.iter().find_map(|node| match node {
            Node::Expression(compiled) if compiled.expression().is_constraint() => known
                .get(compiled.name())
                .filter(|residual| **residual > 0.0)
                .map(|_| compiled.name().to_owned()),
            Node::Input(_) | Node::Expression(_) | Node::Simulation { .. } => None,
        })
    }
}

/// One line per non-input node: what it reads, an arrow, its name.
impl fmt::Display for Diagram<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (to, node) in self.nodes.iter().enumerate() {
            if matches!(node, Node::Input(_)) {
                continue;
            }
            let reads = self
                .edges
                .iter()
                .filter(|(_, edge_to)| *edge_to == to)
                .map(|(from, _)| self.nodes[*from].name())
                .collect::<Vec<_>>()
                .join(", ");
            let kind = match node {
                Node::Input(_) => unreachable!(),
                Node::Expression(compiled) if compiled.expression().is_constraint() => "constraint",
                Node::Expression(compiled) if compiled.is_cheap() => "cheap",
                Node::Expression(_) => "expensive",
                Node::Simulation { .. } => "simulation",
            };
            writeln!(f, "{reads} -> {} [{kind}]", node.name())?;
        }
        Ok(())
    }
}

/// The document from `docs/compile-system.md`, with `ansys` standing in for
/// a simulation of `x1 * x2`.
const INPUTS: [&str; 2] = ["x1", "x2"];
const EXTERNALS: [&str; 1] = ["ansys"];
const EXPRESSIONS: [(&str, &str); 4] = [
    ("f1", "x1 + 10"),
    ("f3", "f1 * ansys"),
    ("c1", "x1 + x2 > 0.5"),
    ("c2", "f3 < 100"),
];

#[test]
fn compile_once_and_walk_the_diagram_many_times() {
    // Compiled once; every walk below reuses these nodes.
    let compiled =
        compile_system(&INPUTS, &EXTERNALS, &EXPRESSIONS).expect("the document compiles");

    let simulations_run = Cell::new(0);
    let ansys = Node::Simulation {
        name: "ansys",
        reads: vec!["x1", "x2"],
        run: Box::new(|args: &[f64]| {
            simulations_run.set(simulations_run.get() + 1);
            args[0] * args[1]
        }),
    };
    let diagram = Diagram::new(&INPUTS, &compiled, vec![ansys]);

    let designs = [[1.0, 2.0], [3.0, 4.0], [0.5, 0.25], [2.0, 0.5]];
    for [x1, x2] in designs {
        let Outcome::Complete(values) = diagram.run(&[x1, x2]) else {
            panic!("({x1}, {x2}) satisfies c1, so the walk completes");
        };
        let f1 = x1 + 10.0;
        let ansys = x1 * x2;
        assert_eq!(values["f1"], f1);
        assert_eq!(values["ansys"], ansys);
        assert_eq!(values["f3"], f1 * ansys);
        // Constraints come back as residuals: `<= 0` holds.
        assert!(values["c1"] <= 0.0);
        assert_eq!(values["c2"] <= 0.0, f1 * ansys < 100.0, "({x1}, {x2})");
    }
    assert_eq!(simulations_run.get(), designs.len());
}

#[test]
fn a_cheap_constraint_that_fails_spares_the_simulation() {
    let compiled =
        compile_system(&INPUTS, &EXTERNALS, &EXPRESSIONS).expect("the document compiles");

    let simulations_run = Cell::new(0);
    let ansys = Node::Simulation {
        name: "ansys",
        reads: vec!["x1", "x2"],
        run: Box::new(|args: &[f64]| {
            simulations_run.set(simulations_run.get() + 1);
            args[0] * args[1]
        }),
    };
    let diagram = Diagram::new(&INPUTS, &compiled, vec![ansys]);

    // x1 + x2 = 0.3, so c1 fails before anything expensive is asked for.
    let outcome = diagram.run(&[0.1, 0.2]);
    let Outcome::Infeasible { constraint, known } = outcome else {
        panic!("expected c1 to stop the walk, got {outcome:?}");
    };
    assert_eq!(constraint, "c1");
    assert_eq!(simulations_run.get(), 0);
    // Everything cheap was still computed — a caller can report it.
    assert_eq!(known["f1"], 10.1);
    assert!(!known.contains_key("f3"));
}

#[test]
fn the_diagram_reads_as_the_document() {
    let compiled =
        compile_system(&INPUTS, &EXTERNALS, &EXPRESSIONS).expect("the document compiles");
    let ansys = Node::Simulation {
        name: "ansys",
        reads: vec!["x1", "x2"],
        run: Box::new(|args: &[f64]| args[0] * args[1]),
    };
    let diagram = Diagram::new(&INPUTS, &compiled, vec![ansys]);

    assert_eq!(
        diagram.to_string(),
        "x1 -> f1 [cheap]\n\
         ansys, f1 -> f3 [expensive]\n\
         x1, x2 -> c1 [constraint]\n\
         f3 -> c2 [constraint]\n\
         x1, x2 -> ansys [simulation]\n"
    );
}

/// A caller with no graph at all: the `Vec` is already an evaluation order,
/// so a straight walk works once the externals are supplied.
#[test]
fn the_returned_order_is_enough_for_a_caller_without_a_graph() {
    let compiled =
        compile_system(&INPUTS, &EXTERNALS, &EXPRESSIONS).expect("the document compiles");

    let mut known = BTreeMap::from([
        ("x1".to_owned(), 3.0),
        ("x2".to_owned(), 4.0),
        ("ansys".to_owned(), 12.0),
    ]);
    for node in &compiled {
        let column = Mat::from_fn(node.reads().len(), 1, |row, _| {
            known[node.reads()[row].name()]
        });
        let value = node.expression().eval(column.as_ref()).expect("evaluates")[0];
        known.insert(node.name().to_owned(), value);
    }
    assert_eq!(known["f1"], 13.0);
    assert_eq!(known["f3"], 156.0);
    assert!(known["c1"] <= 0.0);
    assert!(known["c2"] > 0.0, "156 < 100 does not hold");
}
