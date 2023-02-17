use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};

use glob::glob;
use itertools::Itertools;
use rand::seq::SliceRandom;
use tww::{exact::*, graph::*, heuristic::lower_bound_lb1::lower_bound, io::*};

#[allow(unused_imports)]
use rayon::prelude::*;

fn load_best_known() -> std::io::Result<HashMap<String, NumNodes>> {
    let reader = File::open("instances/best_known_solutions.csv")?;
    let buf_reader = BufReader::new(reader);

    let mut dict = HashMap::new();

    for line in buf_reader.lines() {
        let line = line?;
        let parts = line.split(',').filter(|t| !t.is_empty()).collect_vec();

        if parts.is_empty() {
            continue;
        }

        if parts.len() != 2 {
            eprintln!("Invalid best-known line: {line} -> {parts:?}");
            continue;
        }

        let file = parts[0].trim();
        let tww = parts[1].parse().unwrap();

        if let Some(old) = dict.insert(file.to_owned(), tww) {
            assert_eq!(old, tww, "Mismatch for file {file}");
            eprintln!("Warning: duplicate of best-known for {file}");
        }
    }

    Ok(dict)
}

fn main() {
    let mut files = ["tiny", "exact-public"]
        .into_iter()
        .flat_map(|p| {
            glob(format!("instances/{p}/*.gr").as_str())
                .expect("Failed to glob")
                .map(|r| r.expect("Failed to access globbed path"))
        })
        .collect_vec();

    files.shuffle(&mut rand::thread_rng());

    let best_known = load_best_known().unwrap_or_default();
    println!("Found {} best known values", best_known.len());

    files.par_iter().for_each(|file| {
        let filename = String::from(file.as_os_str().to_str().unwrap());
        let graph = AdjArray::try_read_pace_file(file).expect("Cannot open PACE file");

        let start = Instant::now();
        let (sol_size, sol) = naive::naive_solver(&graph);
        let duration = start.elapsed();

        let best_known = best_known.get(&filename);
        let lb = lower_bound(&graph, 0);

        println!(
            "{filename:<50} | {:>6} | {:>8} | {sol_size:>4} ({:>4}) lb: {lb:>4} | {:>6} ms",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            best_known.map_or_else(|| String::from("?"), |b| format!("{b}")),
            duration.as_millis()
        );

        if sol_size > *best_known.unwrap_or(&Node::MAX) {
            println!("VIOLATION FOR {filename} <===============================================================");
            return;
        }

        {
            let mut solution_writer = BufWriter::new(
                File::create(format!("{filename}.solution")).expect("Unable to create file"),
            );
            writeln!(solution_writer, "c time: {}ms", duration.as_millis())
                .expect("Could not header");
            sol.pace_writer(solution_writer)
                .expect("Could not write solution");
        }
    });
}
