use std::{
    fs::File,
    io::BufWriter,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use ::log::LevelFilter;
use itertools::Itertools;
use rand::{seq::IteratorRandom, Rng};
use rayon::prelude::*;
use tww::{prelude::*, *};

use structopt::*;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long, default_value = "500000")]
    repeats: u32,

    #[structopt(short = "w", long)]
    write: bool,

    #[structopt(short = "e", long)]
    write_buggy_only: bool,

    #[structopt(short = "b", long)]
    bipartite_only: bool,

    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: usize,
}

fn main() {
    let opt = Opt::from_args();
    log::build_pace_logger_for_verbosity(LevelFilter::Warn, opt.verbose);

    let nodes = [15, 18, 20];
    let avg_deg = [6, 8, 10];

    let params: Vec<_> = nodes
        .into_iter()
        .cartesian_product(avg_deg)
        .filter_map(|(n, d)| (d + 1 < n).then_some((n, d as f64 / (n - 1) as f64)))
        .collect();

    let completed = AtomicU64::new(0);
    let time_in_algo1 = AtomicU64::new(0);
    let time_in_algo2 = AtomicU64::new(0);

    let total_instances = (nodes.len() as u64) * (avg_deg.len() as u64) * (opt.repeats as u64);

    (0..opt.repeats)
        .into_par_iter()
        .flat_map(|_| params.par_iter())
        .for_each_init(rand::thread_rng, |rng, &(n, p)| {
            let graph = AdjArray::random_colored_gnp(rng, n, p, 0.7);

            process_graph(&opt, graph, rng, n, &time_in_algo1, &time_in_algo2);

            let counter = completed.fetch_add(1, Ordering::Relaxed);
            if counter % 100000 == 0 && counter > 0 {
                let t1 = time_in_algo1.load(Ordering::Relaxed);
                let t2 = time_in_algo2.load(Ordering::Relaxed);

                println!(
                    "Completed {:>7} of {:>7} | Time1: {t1:>8}ms Time2: {t2:>8}ms {}",
                    counter,
                    total_instances,
                    (t1.max(t2) as f64) / (t1.min(t2) as f64)
                );
            }
        });
}

fn process_graph(
    opt: &Opt,
    mut graph: AdjArray,
    rng: &mut rand::rngs::ThreadRng,
    n: u32,
    time1: &AtomicU64,
    time2: &AtomicU64,
) {
    if opt.bipartite_only && !graph.is_bipartite() {
        let class_size = rng.gen_range((n / 4)..(n / 2)) as usize;
        let part = BitSet::new_with_bits_set(n, (0..n).choose_multiple(rng, class_size));
        graph.remove_edges_within_bipartition_class(&part);
        assert!(graph.is_bipartite());
    }

    if graph.number_of_edges() < 4 {
        return;
    }

    let graph = graph.remove_disconnected_verts().0;

    let n = graph.number_of_nodes();

    let time = Instant::now();

    let config = branch_and_bound::FeatureConfiguration::pessimitic();

    //config.atmost_distance_three = false;
    let (tww_bb, sol_bb) = {
        let mut algo = branch_and_bound::BranchAndBound::new(graph.clone());
        algo.configure_features(config);
        algo.solve().unwrap()
    };

    if tww_bb < 5 {
        return;
    }

    let ela1 = time.elapsed().as_millis() as u64;

    let time = Instant::now();

    //config.atmost_distance_three = true;
    let (tww_dist3, sol_dist3) = {
        let mut algo = branch_and_bound::BranchAndBound::new(graph.clone());
        algo.configure_features(config);
        algo.solve().unwrap()
    };

    time1.fetch_add(ela1, Ordering::Relaxed);
    time2.fetch_add(time.elapsed().as_millis() as u64, Ordering::Relaxed);

    let tww = tww_dist3.min(tww_bb);
    let mismatched = tww_dist3 != tww_bb;
    if mismatched {
        println!(
            "Mismatched solutions for n = {n}, m = {}, tww_naive = -, tww_dist3 = {tww_dist3}, tww_bb = {tww_bb} | digest = {}",
            graph.number_of_edges(),
            graph.digest_sha256()
        );
    }

    let is_bipartite = graph.is_bipartite();
    #[allow(unused_variables)]
    let filename = format!(
        "instances/small-random/n{n:>03}_m{:>04}_tww{tww:>03}{}_{}.gr",
        graph.number_of_edges(),
        if is_bipartite { "_bip" } else { "" },
        &graph.digest_sha256()[..16]
    );

    if opt.write && (mismatched || !opt.write_buggy_only) {
        graph
            .try_write_pace_file(filename.clone())
            .expect("Failed to write file");

        graph
            .try_write_edgelist_file(format!("{filename}.edges"))
            .expect("Failed to write file");

        sol_bb
            .pace_writer(BufWriter::new(
                File::create(format!("{filename}.bb.solution"))
                    .expect("Failed to open solution file"),
            ))
            .expect("Cannot write solution");

        sol_dist3
            .pace_writer(BufWriter::new(
                File::create(format!("{filename}.dist3.solution"))
                    .expect("Failed to open solution file"),
            ))
            .expect("Cannot write solution");

        println!("./analysis/visualize.py {filename}.bb.solution");
    }
}
