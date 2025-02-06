use std::{io::Write, time::Instant};

use dss::{
    graph::CsrGraph,
    heuristic::{greedy_approximation, reverse_greedy_search::GreedyReverseSearch},
    io::GraphPaceReader,
    prelude::IterativeAlgorithm,
    utils::signal_handling,
};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

fn main() -> anyhow::Result<()> {
    signal_handling::initialize();

    let mut timer = Instant::now();

    let mut graph = CsrGraph::try_read_pace(std::io::stdin().lock()).unwrap();

    let read_time = timer.elapsed().as_millis();
    timer = Instant::now();

    let mut rng = Pcg64Mcg::seed_from_u64(123u64);
    let domset = greedy_approximation(&graph);

    let greedy_time = timer.elapsed().as_millis();
    timer = Instant::now();

    let mut search = GreedyReverseSearch::<_, _, 10, 10>::new(&mut graph, domset, &mut rng);

    let init_time = timer.elapsed().as_millis();
    timer = Instant::now();

    let mut counter = 1usize;
    while !signal_handling::received_ctrl_c() && !search.is_completed() {
        counter += 1;
        search.step();
    }

    let search_time = timer.elapsed().as_millis();

    let size = search.best_known_solution().unwrap().len();

    writeln!(
        std::io::stdout(),
        "{size},{counter},{read_time},{greedy_time},{init_time},{search_time}"
    )?;

    Ok(())
}
