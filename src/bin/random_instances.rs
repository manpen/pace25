use itertools::Itertools;
use rayon::prelude::*;
use tww::{exact::*, graph::*, io::*};

fn main() {
    let nodes = [5, 6, 7, 8, 9, 10, 15];
    let avg_deg = [1, 2, 5, 8, 10];
    let num_instances = 100;

    let params: Vec<_> = nodes
        .into_iter()
        .cartesian_product(avg_deg)
        .filter_map(|(n, d)| (d + 1 < n).then_some((n, d as f64 / (n - 1) as f64)))
        .collect();

    (0..num_instances)
        .into_par_iter()
        .flat_map(|_| params.par_iter())
        .for_each_init(rand::thread_rng, |rng, &(n, p)| {
            let graph = AdjArray::random_black_gnp(rng, n, p);
            let (tww, _) = naive::naive_solver(&graph);

            let filename = format!(
                "instances/small-random/n{:>03}_m{:>04}_tww{:>03}_{}.gr",
                n,
                graph.number_of_edges(),
                tww,
                graph.digest_sha256()
            );

            graph
                .try_write_pace_file(filename)
                .expect("Failed to write file");
        });
}
