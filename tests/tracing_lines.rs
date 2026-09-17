//! A run that should have finished and has not is located by its last log
//! line. That only works if the lines are there, so this pins them: every
//! COBYLA run announces itself *before* it runs, and every evaluation it
//! makes is a numbered `trace` event.
//!
//! The subscriber is scoped to the test (`with_default`), so nothing here
//! leaks into another test's output.

use std::io::Write;
use std::sync::{Arc, Mutex};

use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;
use sojourn::{ConstraintSolver, ConstraintSystem, InputVariable, Strategy};
use tracing_subscriber::fmt::MakeWriter;

/// Everything the subscriber wrote, readable after the run.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .expect("no panic held the log")
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl MakeWriter<'_> for Captured {
    type Writer = Self;

    fn make_writer(&self) -> Self {
        self.clone()
    }
}

impl Captured {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().expect("no panic held the log").clone())
            .expect("the subscriber writes UTF-8")
    }
}

#[test]
fn a_cobyla_run_is_announced_before_it_runs_and_traced_as_it_goes() -> anyhow::Result<()> {
    let log = Captured::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(log.clone())
        .finish();

    let system = ConstraintSystem::new(
        vec![
            InputVariable::new("x", -1.0, 1.0),
            InputVariable::new("y", -1.0, 1.0),
        ],
        ["x^2 + y^2 < 0.25"],
    )?;
    tracing::subscriber::with_default(subscriber, || {
        ConstraintSolver::new()
            .with_strategies(vec![Strategy::LocalSolve])
            .solve(&system, &mut Xoshiro256PlusPlus::seed_from_u64(1))
    })?;

    let text = log.text();
    let starts = text
        .lines()
        .find(|line| line.contains("cobyla starts"))
        .unwrap_or_else(|| panic!("no announcement before the run:\n{text}"));
    assert!(
        starts.contains("dimensions=2") && starts.contains("budget="),
        "the announcement says what it is about to do: {starts}"
    );
    let evaluations: Vec<&str> = text
        .lines()
        .filter(|line| line.contains("cobyla evaluation"))
        .collect::<Vec<_>>();
    assert!(!evaluations.is_empty(), "no evaluation was traced:\n{text}");
    assert!(
        evaluations[0].contains("evaluation=1 ") && evaluations[0].contains("cost="),
        "evaluations are numbered from one and carry the cost: {}",
        evaluations[0]
    );
    // The announcement comes first, so a hang inside the run still leaves it.
    let announced_at = text.find("cobyla starts").expect("found above");
    let first_evaluation_at = text.find("cobyla evaluation").expect("found above");
    assert!(announced_at < first_evaluation_at);
    Ok(())
}
