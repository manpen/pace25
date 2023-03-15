use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

use glob::glob;
use itertools::Itertools;
use tww::prelude::*;

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
    //build_pace_logger_for_level(LevelFilter::Trace);

    let args = std::env::args();
    let files = if args.len() > 1 {
        args.skip(1).map(PathBuf::from).collect_vec()
    } else {
        ["tiny", "exact-public"]
            .into_iter()
            .flat_map(|p| {
                glob(format!("instances/{p}/*.gr").as_str())
                    .expect("Failed to glob")
                    .map(|r| r.expect("Failed to access globbed path"))
            })
            .collect_vec()
    };

    let best_known = load_best_known().unwrap_or_default();

    files
        .par_iter()
        .for_each(|file| process_graph(file, &best_known));
}

fn process_graph(file: &PathBuf, best_known: &HashMap<String, u32>) {
    let filename = String::from(file.as_os_str().to_str().unwrap());
    let org_graph = AdjArray::try_read_pace_file(file).expect("Cannot open PACE file");

    //trace!("LB1: {}", lower_bound_lb1(&org_graph, 0));

    /*trace!(
        "LB Subgraph: {}",
        lower_bound_subgraph(&org_graph, 0, org_graph.number_of_nodes())
    );

    return;*/

    let start = Instant::now();
    let (sol_size, sol) = naive::naive_solver(&org_graph);
    let duration = start.elapsed();

    let best_known = best_known.get(&filename);

    /*{
        let mut graph = org_graph.clone();

        let mut cs = ContractionSequence::new(graph.number_of_nodes());
        initial_pruning(&mut graph, &mut cs);

        if graph.number_of_edges() * 4
            > (graph.number_of_nodes() as NumEdges) * ((graph.number_of_nodes() as NumEdges) - 1)
        {
            graph = graph.trigraph_complement();
            println!(
                "{file:?} Complementing! {} -> {}",
                org_graph.number_of_edges(),
                graph.number_of_edges()
            );
            initial_pruning(&mut graph, &mut cs);
        }

        let (graph, _) = graph.remove_disconnected_verts();

        let mut largest_mod = 0;
        if let Some(part) = graph.compute_modules() {
            for x in 0..part.number_of_classes() {
                largest_mod = largest_mod.max(part.number_in_class(x));
            }
        }

        let num_art = graph.compute_articulation_points().cardinality();
        let num_bridges = graph.compute_bridges().len();

        /*println!(
            "{filename:<50} | n={:>6} | m={:>8} | art: {num_art:>6} brd: {num_bridges:>6} mod: {largest_mod:?}",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        );*/

        //return;

        let ub = heuristic_solve(&graph).0;

        let lb1 = 0; //lower_bound(&graph, 0);
        let lb2 = 0; // lower_bounds_subgraph::lower_bounds_subgraph(&graph, 0, ub);
        let lb3 = 0; //lower_bounds_subgraph::lower_bounds_subgraph_art(&graph, 0, ub);

        println!(
            "{filename:<50} | n={:>6} | m={:>8} | lb1={lb1:>4} lb2={lb2:>4} lb3={lb3:>4} ub={ub:>4} best={:>4} | {:>6} ms {}",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            best_known.map_or_else(|| String::from("?"), |b| format!("{b}")),
            start.elapsed().as_millis(),
            ub - lb1.max(lb2.max(lb3)),
        );
    }

    return; */

    let lb = lower_bound_lb1(&org_graph, 0);

    println!(
        "{filename:<50} | {:>6} | {:>8} | {sol_size:>4} ({:>4}) lb: {lb:>4} | {:>6} ms",
        org_graph.number_of_nodes(),
        org_graph.number_of_edges(),
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
        writeln!(solution_writer, "c time: {}ms", duration.as_millis()).expect("Could not header");
        sol.pace_writer(solution_writer)
            .expect("Could not write solution");
    }
}
