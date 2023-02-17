use std::{fs::File, io::BufWriter, time::Duration};

use ::log::LevelFilter;
use itertools::Itertools;
use rayon::prelude::*;
use tww::{exact::two_stage_sat_solver::TwoStageSatSolver, prelude::*, *};

use structopt::*;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long, default_value = "100")]
    repeats: u32,

    #[structopt(short, long)]
    write: bool,

    #[structopt(short = "e", long)]
    write_buggy_only: bool,

    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: usize,
}

fn main() {
    let opt = Opt::from_args();
    log::build_pace_logger_for_verbosity(LevelFilter::Warn, opt.verbose);

    let nodes = [5, 6, 7, 8, 9, 10, 15];
    let avg_deg = [1, 2, 5, 8, 10];

    let params: Vec<_> = nodes
        .into_iter()
        .cartesian_product(avg_deg)
        .filter_map(|(n, d)| (d + 1 < n).then_some((n, d as f64 / (n - 1) as f64)))
        .collect();

    (0..opt.repeats)
        .into_par_iter()
        .flat_map(|_| params.par_iter())
        .for_each_init(rand::thread_rng, |rng, &(n, p)| {
            let graph = AdjArray::random_black_gnp(rng, n, p);
            let (tww_naive, sol_naive) = naive::naive_solver(&graph);
            let (tww_sat, sol_sat) =
                TwoStageSatSolver::new(&graph, Duration::from_millis(100)).solve();

            #[allow(unused_variables)]
            let filename = format!(
                "instances/small-random/n{n:>03}_m{:>04}_tww{tww_naive:>03}_{}.gr",
                graph.number_of_edges(),
                graph.digest_sha256()
            );

            if opt.write && (tww_naive != tww_sat || !opt.write_buggy_only) {
                graph
                    .try_write_pace_file(filename.clone())
                    .expect("Failed to write file");

                sol_naive
                    .pace_writer(BufWriter::new(
                        File::create(format!("{filename}.naive.solution"))
                            .expect("Failed to open solution file"),
                    ))
                    .expect("Cannot write solution");

                sol_sat
                    .pace_writer(BufWriter::new(
                        File::create(format!("{filename}.sat.solution"))
                            .expect("Failed to open solution file"),
                    ))
                    .expect("Cannot write solution");
            }

            assert_eq!(tww_naive, tww_sat, "{filename}");
        });
}
