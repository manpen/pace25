use std::time::Duration;

use ::log::LevelFilter;
use itertools::Itertools;
use rayon::prelude::*;
use tww::{exact::two_stage_sat_solver::TwoStageSatSolver, prelude::*, *};

use structopt::*;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long, default_value = "1000")]
    repeats: u32,

    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: usize,
}

fn main() {
    let opt = Opt::from_args();
    log::build_pace_logger_for_verbosity(LevelFilter::Warn, opt.verbose);

    let nodes = [10, 15];
    let avg_deg = [1, 2, 5, 8, 10];

    let params: Vec<_> = nodes
        .into_iter()
        .cartesian_product(avg_deg)
        .filter_map(|(n, d)| (d + 1 < n).then_some((n, d as f64 / (n - 1) as f64)))
        .collect();

    let atomic_errors = std::sync::atomic::AtomicU32::new(0);
    let atomic_instances = std::sync::atomic::AtomicU32::new(0);

    (0..opt.repeats)
        .into_par_iter()
        .flat_map(|_| params.par_iter())
        .for_each_init(rand::thread_rng, |rng, &(n, p)| {
            let graph = AdjArray::random_black_gnp(rng, n, p);
            let (tww_sat, _) = TwoStageSatSolver::new(&graph, Duration::from_millis(100)).solve();

            let (tww_heu, _) = heuristic::sweep_solver::heuristic_solve(&graph);
            atomic_instances.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if tww_heu != tww_sat {
                atomic_errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });
    let total = atomic_instances.load(std::sync::atomic::Ordering::Relaxed);
    let errors = atomic_errors.load(std::sync::atomic::Ordering::Relaxed);

    println!("Success heuristic {}/{}", total - errors, total);
}
