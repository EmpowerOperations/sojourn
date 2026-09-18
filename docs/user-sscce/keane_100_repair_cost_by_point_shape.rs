//! SSCCE: sojourn's `repair` on Keane's bump at n=100 costs ~5 s per wolf proposal.
//!
//! Observed on the 0.13.5 c11 `c=2` probe (2026-09-18, `docs/2026-09-18-sojourn-repair-on-
//! keane-100.md`). Once the run leaves its initial design, ~5 proposals per iteration go:
//! Newton declines after one iteration ("a singular solve") -> the sampling box misses (20
//! rounds, 5120 draws) -> COBYLA from the nearest point runs ~1900 evaluations and lands
//! infeasible -> COBYLA from the reference runs ~1900 more and lands feasible, 1e-6 to 1e-3
//! from where the point started. Two 100-D COBYLA runs, ~5.3 s, to move a point a millionth;
//! ~16 s per iteration, all of it here.
//!
//! The trigger (captured with `ARTEMIS_TRACE=artemis::repair=trace`): the wolves' proposals
//! have **6-11 coordinates clamped exactly to the lower bound 0** and several at the upper
//! bound 10. The product is exactly 0, so every partial derivative of `prod` is exactly 0 --
//! the KKT system is singular by construction, not by conditioning. The fix is tiny (lift the
//! zeros to ~1e-7: the other ~90 coordinates multiply to something astronomical), which is
//! why the landing is a millionth away, and why two COBYLA runs at 100-D is the wrong tool
//! for it.
//!
//! This test uses nothing from Artemis but the census: the problem as the benchmark declares
//! it, a synthetic family with `k` coordinates at 0, and one captured proposal verbatim.
//! `repair_point` on each, timed, with sojourn's trace printed so the chain is visible:
//!
//! ```text
//! cargo test --release -p empowerops-artemis-core --test sscce_keane_repair -- --nocapture --ignored
//! ```
//!
//! `#[ignore]` because it is a measurement, not an invariant: it prints, it does not assert
//! timings.
//!
//! Kept verbatim as received. What it became in sojourn:
//! `tests/regression_fixture.rs::repair_strands_on_a_product_constraint::
//! the_100_variable_proposal_with_eleven_zeros_is_projected_not_searched`, and the
//! `2026-09-18` note in `docs/todo.md`.

mod common;

use std::time::Instant;

use faer::Col;
use empowerops_artemis_core::feasible::{FeasibleRegion, Residual};
use empowerops_artemis_core::sojourn_region::{census, SojournProblem};
use tracing_subscriber::EnvFilter;

const N: usize = 100;

fn keane_bump_problem(n: usize) -> SojournProblem {
    SojournProblem {
        variables: common::vars(n, 0.0, 10.0),
        constraints: vec![
            format!("0.75 - prod(1, {n}, i -> var[i]) < 0"),
            format!("sum(1, {n}, i -> var[i]) - {} < 0", 7.5 * n as f64),
        ],
    }
}

fn col(v: &[f64]) -> Col<f64> { Col::from_fn(v.len(), |i| v[i]) }

fn value(r: Residual) -> f64 {
    match r { Residual::Value(v) => v, Residual::Indeterminate => f64::NAN }
}

fn prod(x: &[f64]) -> f64 { x.iter().product() }

/// A Keane-like point: the bump's optimum has a few coordinates near 3 and the rest small,
/// decreasing. `head` coordinates at ~3.1, the rest at a value that makes the product land at
/// `target` (feasible needs `>= 0.75`).
fn keane_like(head: usize, target: f64) -> Vec<f64> {
    let mut x = vec![3.1; N];
    let head_prod = 3.1f64.powi(head as i32);
    let tail = (target / head_prod).powf(1.0 / (N - head) as f64);
    for v in x.iter_mut().skip(head) { *v = tail; }
    x
}

/// A wolf proposal from the c11 `c=2` probe, iteration ~40, verbatim: 11 coordinates at 0,
/// 4 at 10, residual 0.75 (the product is 0).
const CAPTURED: [f64; 100] = [
    0.0, 2.22189998407938, 8.764678459934343, 3.094397616664273, 9.322036592108637, 5.96944660860104, 4.787862356901575, 10.0, 4.156313322093435, 5.492732599826343,
    7.96266545811535, 2.695502308509957, 8.644014077791333, 4.555671713744944, 9.902076329536735, 2.445656114201819, 10.0, 0.0, 4.994131875757679, 0.0,
    0.0, 5.777610047250547, 0.5699988022423685, 0.6962735961306832, 1.09929179352666, 5.040518425671441, 0.0, 9.088581274463243, 3.564946687550064, 6.621452820028271,
    6.885080448669081, 6.550165621124641, 8.359535695141281, 6.544844407984651, 9.232063752528061, 1.354670283660715, 5.077923294883641, 7.054988665863998, 1.916639097571489, 9.137756928142721,
    8.399863586147342, 5.377688101404801, 6.822260240396487, 10.0, 7.591575155697313, 0.0, 3.142269463722272, 2.695085463732622, 4.769566559574289, 6.738536825083114,
    4.355807716420325, 9.969060742946116, 6.774070428523348, 8.850062252944381, 3.717416120592227, 3.200680293761951, 5.168521481504353, 2.502798634450153, 3.510901723504484, 4.01320392064421,
    1.929152151919125, 6.20551029507453, 4.762900002487842, 0.0, 6.13932973699611, 0.0, 0.0, 3.731335636501735, 4.024881880957532, 0.0,
    0.7918236521353679, 9.993988809039658, 2.173470090981415, 7.483616708693935, 8.652037630722493, 3.27764366356877, 10.0, 8.340596752550143, 6.050906151569723, 6.96277271298732,
    9.023894227528649, 2.858096910293614, 4.950271479100283, 1.010931231541861, 1.3102671886915, 0.0, 0.3860269067771993, 5.270677597923203, 9.766103646212891, 9.422419853216706,
    6.098498540653791, 6.731197910902245, 3.812265661718834, 2.432386701248863, 9.931687289243193, 8.808060980025456, 6.283511779263659, 2.858858937003123, 8.222027796986254, 6.147271873609433,
];

#[test]
#[ignore]
fn keane_100_repair_cost_by_point_shape() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("sojourn=debug"))
        .with_target(true)
        .with_ansi(false)
        .without_time()
        .try_init();

    let p = keane_bump_problem(N);
    let mut c = census(&p, 0).expect("census");

    // The wolves' shape: a Keane-like point with `k` coordinates clamped to the lower bound.
    // With `k = 1` the gradient of `prod` is nonzero in that one coordinate and Newton works;
    // from `k = 2` it is identically zero and Newton is singular by construction.
    let with_zeros = |k: usize| {
        let mut x = keane_like(10, 0.80);
        for v in x.iter_mut().rev().take(k) { *v = 0.0; }
        x
    };
    let mut wall_short = keane_like(10, 0.75);
    for v in wall_short.iter_mut() { *v *= 1.0 - 1e-4; }  // a hair below the wall, no zeros

    let mut cases: Vec<(String, Vec<f64>)> = vec![("just below the wall, no zeros".into(), wall_short)];
    for k in [1, 2, 4, 6, 8, 11] {
        cases.push((format!("{k} coordinate(s) at 0"), with_zeros(k)));
    }
    cases.push(("captured wolf proposal (11 zeros)".into(), CAPTURED.to_vec()));

    println!("\n{:34} {:>12} {:>12} {:>9} {:>12}", "point", "prod before", "residual", "seconds", "moved (L2)");
    for (name, x) in cases {
        let name = name.as_str();
        let before = value(c.region.residual(col(&x).as_ref()).unwrap());
        eprintln!("\n=== {name}: prod {:.3e}, residual {before:.3e}", prod(&x));
        let t = Instant::now();
        let y = c.region.repair_point(col(&x).as_ref()).expect("repair");
        let secs = t.elapsed().as_secs_f64();
        let after = value(c.region.residual(y.as_ref()).unwrap());
        let moved = (&y - &col(&x)).norm_l2();
        let y_vec: Vec<f64> = (0..N).map(|i| y[i]).collect();
        println!("{name:34} {:>12.3e} {before:>12.3e} {secs:>9.2} {moved:>12.3e}   -> residual {after:.2e}, prod {:.3}", prod(&x), prod(&y_vec));
        assert!(after <= 0.0, "{name}: repair landed infeasible ({after})");
    }
}
