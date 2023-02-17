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
    heuristic::monte_carlo_search_tree::{
        timeout_monte_carlo_search_tree_solver_with_descend, MonteCarloSearchTree,
        MonteCarloSearchTreeGame,
    },
    io::GraphPaceReader,
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

    //let files = vec![std::path::PathBuf::from_str("instances/exact-public/exact_004.gr").unwrap()];

    let best_known = load_best_known().unwrap_or_default();
    println!("Found {} best known values", best_known.len());

    let timeout = Some(std::time::Duration::from_secs(10));
    //let timeout = None;

    let cumulative_score = std::sync::Arc::new(std::sync::Mutex::new(0));
    files.iter().take(10).for_each(|file| {
        let filename = String::from(file.as_os_str().to_str().unwrap());
        let graph = AdjArray::try_read_pace_file(file).expect("Cannot open PACE file");

        let start = Instant::now();

        let result = if let Some(timeout) = timeout {
            //timeout_monte_carlo_search_tree_solver(&graph, timeout)
            timeout_monte_carlo_search_tree_solver_with_descend(
                &graph,
                timeout,
                std::time::Duration::from_millis(200),
                50,
            )
        } else {
            let mut full_tree = MonteCarloSearchTree::new(&graph, true);

            for _ in 0..100000 {
                let mut tree = full_tree.new_game();
                tree.make_random_choice(MonteCarloSearchTreeGame::random_choice, &mut full_tree);

                full_tree.add_game(&tree);
            }
            full_tree.permanently_collapse_one_move();
            (
                full_tree.best_score(),
                full_tree.into_best_contraction_seq(),
                100000,
            )
        };
        let sol_size = result.0;

        let duration = start.elapsed();

        let best_known = best_known.get(&filename);

        println!(
            "{filename:<50} | {:>6} | {:>8} | {sol_size:>6} ({:>6}) | {:>6} ms | {} games",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            best_known.map_or_else(|| String::from("?"), |b| format!("{b}")),
            duration.as_millis(),
            result.2
        );
        *cumulative_score.lock().unwrap() += sol_size;
    });
    println!("Cumulative score {}", cumulative_score.lock().unwrap());
}
