use dss::{
    graph::CsrGraph,
    heuristic::{greedy_approximation, reverse_greedy_search::GreedyReverseSearch},
    io::GraphPaceReader,
    prelude::TerminatingIterativeAlgorithm,
    utils::signal_handling,
};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

fn main() -> anyhow::Result<()> {
    signal_handling::initialize();

    let mut graph = CsrGraph::try_read_pace(std::io::stdin().lock()).unwrap();

    let mut rng = Pcg64Mcg::seed_from_u64(123u64);
    let domset = greedy_approximation(&graph);

    let mut search = GreedyReverseSearch::<_, _, 8, 10>::new(&mut graph, domset, &mut rng);

    let solution = search.run_to_completion().unwrap();
    solution.write(std::io::stdout())?;

    Ok(())
}
