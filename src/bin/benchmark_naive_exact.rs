use std::time::Instant;

use glob::glob;
use itertools::Itertools;
use tww::{exact::*, graph::*, io::*};

#[allow(unused_imports)]
use rayon::prelude::*;

fn main() {
    let files = glob("instances/exact-public/*.gr")
        .expect("Failed to glob")
        .map(|r| r.expect("Failed to access globbed path"))
        .collect_vec();

    files.par_iter().for_each(|file| {
        let filename = String::from(file.as_os_str().to_str().unwrap());
        let graph = AdjArray::try_read_pace_file(file).expect("Cannot open PACE file");

        let start = Instant::now();
        let (sol_size, _sol) = naive::naive_solver(&graph);
        let duration = start.elapsed();

        println!(
            "{filename:<50} | {:>6} | {:>8} | {sol_size:>6} | {:>6} ms",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            duration.as_millis()
        );
    });
}
