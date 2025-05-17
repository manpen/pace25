use std::{io::Write, time::Instant};

use dss::{
    graph::{
        AdjArray, AdjacencyList, BitSet, CsrGraph, Edge, Getter, GraphFromReader, GraphNodeOrder,
        relabel::cuthill_mckee,
    },
    heuristic::{greedy_approximation, reverse_greedy_search::GreedyReverseSearch},
    io::GraphPaceReader,
    kernelization::{KernelizationRule, SubsetRule, rule1::Rule1},
    prelude::IterativeAlgorithm,
    utils::{DominatingSet, signal_handling},
};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

fn main() -> anyhow::Result<()> {
    signal_handling::initialize();

    let mut timer = Instant::now();

    let mut graph = AdjArray::try_read_pace(std::io::stdin().lock()).unwrap();

    let read_time = timer.elapsed().as_millis();
    timer = Instant::now();

    // PreProcessing
    // (1) Run Rule1 (x times?)
    // (2) TBD: Run Path-Rule
    // (3) Run SubsetRule
    // (4) Collect set of fixed nodes, deletable nodes, deletable edges
    // (5) Mark non-deletable nodes with fixed neighbor in is_perm_covered
    // (6) Compact graph (without relabeling)
    // (7) Relabel graph
    // (8) Convert to CsrGraph
    // (9) Run heuristics
    let mut domset = DominatingSet::new(graph.number_of_nodes());

    // (1)
    let _rule1_deletable = Rule1::apply_rule(&graph, &mut domset);

    // (2) TBD

    // (3)
    let redundant = SubsetRule::apply_rule(&mut graph, &mut domset);

    // (4-6)

    // (7-8)
    let orig_graph = graph;
    let mapping = cuthill_mckee(&orig_graph);
    let mut graph = CsrGraph::from_edges(
        orig_graph.number_of_nodes(),
        orig_graph
            .edges(true)
            .map(|Edge(u, v)| Edge(mapping.new_id_of(u).unwrap(), mapping.new_id_of(v).unwrap())),
    );

    let redundant_mapped = BitSet::new_with_bits_set(
        graph.number_of_nodes(),
        redundant
            .iter_set_bits()
            .map(|u| mapping.new_id_of(u).unwrap()),
    );

    let is_perm_covered = BitSet::new_with_bits_set(
        graph.number_of_nodes(),
        graph.vertices().filter(|&u| {
            let orig_u = mapping.old_id_of(u).unwrap();
            orig_graph
                .neighbors_of(orig_u)
                .any(|v| domset.is_fixed_node(v))
        }),
    );

    greedy_approximation(&graph, &mut domset, &redundant_mapped);
    let greedy_time = timer.elapsed().as_millis();
    timer = Instant::now();

    let mut rng = Pcg64Mcg::seed_from_u64(123u64);
    let mut search = GreedyReverseSearch::<_, _, 10, 10>::new(
        &mut graph,
        domset,
        is_perm_covered,
        redundant_mapped,
        &mut rng,
    );

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
        "{read_time},{greedy_time},{init_time},{counter},{search_time},{size}"
    )?;

    Ok(())
}
