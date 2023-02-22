use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};

use glob::glob;
use itertools::Itertools;
use tww::{
    graph::{AdjArray, GraphEdgeOrder, GraphNodeOrder, NumNodes},
    io::GraphPaceReader,
    prelude::{lower_bound_lb1::lower_bound, sweep_solver::heuristic_solve},
};

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
    let files = ["exact-public"]
        .into_iter()
        .flat_map(|p| {
            glob(format!("instances/{p}/*.gr").as_str())
                .expect("Failed to glob")
                .map(|r| r.expect("Failed to access globbed path"))
        })
        .collect_vec();

    let best_known = load_best_known().unwrap_or_default();
    println!("Found {} best known values", best_known.len());

    let cumulative_score = std::sync::Arc::new(std::sync::Mutex::new(0));
    let mut cumulative_best_known = 0;
    let mut sols_compared = 0;
    files.iter().take(34).for_each(|file| {
        let filename = String::from(file.as_os_str().to_str().unwrap());
        let graph = AdjArray::try_read_pace_file(file).expect("Cannot open PACE file");

        let start = Instant::now();
        let result = heuristic_solve(&graph);

        let lb = lower_bound(&graph, 0);

        let sol_size = result.0;

        let duration = start.elapsed();

        let best_known = best_known.get(&filename);

        println!(
            "{filename:<50} | {:>6} | {:>8} | {sol_size:>6} ({:>6}) | {:>6} ms | {lb} lb",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            best_known.map_or_else(|| String::from("?"), |b| format!("{b}")),
            duration.as_millis()
        );
        if let Some(best_known) = best_known.as_ref() {
            *cumulative_score.lock().unwrap() += sol_size;
            cumulative_best_known += *best_known;
            sols_compared += 1;
        }
    });
    println!("Cumulative score {}", cumulative_score.lock().unwrap());
    println!("Best known score {cumulative_best_known}");
    println!("Solutions {sols_compared}");
}
