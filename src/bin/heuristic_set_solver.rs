use dss::{
    graph::{
        AdjArray, AdjacencyList, BitSet, CsrGraph, CuthillMcKee, Edge, EdgeOps, ExtractCsrRepr,
        Getter, GraphEdgeEditing, GraphEdgeOrder, GraphFromReader, GraphNodeOrder, Node,
        NodeMapper, NumNodes,
    },
    heuristic::{greedy_approximation, reverse_greedy_search::GreedyReverseSearch},
    io::set_reader::SetPaceReader,
    log::build_pace_logger_for_level,
    prelude::{IterativeAlgorithm, TerminatingIterativeAlgorithm},
    reduction::{
        LongPathReduction, Reducer, RuleOneReduction, RuleSmallExactReduction, RuleSubsetReduction,
    },
    utils::{DominatingSet, signal_handling},
};
use itertools::Itertools;
use log::info;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use std::{path::PathBuf, time::Duration};
use structopt::StructOpt;

#[derive(StructOpt, Default)]
struct Opts {
    #[structopt(short = "i")]
    input: Option<PathBuf>,

    #[structopt(short = "T")]
    timeout: Option<f64>,

    #[structopt(short = "q")]
    no_output: bool,
}

fn load_graph(path: &Option<PathBuf>) -> anyhow::Result<(AdjArray, NumNodes)> {
    use dss::prelude::*;

    if let Some(path) = path {
        Ok(AdjArray::try_read_set_pace_file(path)?)
    } else {
        let stdin = std::io::stdin().lock();
        Ok(AdjArray::try_read_set_pace(stdin)?)
    }
}

struct State<G> {
    graph: G,
    domset: DominatingSet,
    covered: BitSet,
    redundant: BitSet,
}

fn apply_reduction_rules(
    mut graph: AdjArray,
    orig_number_nodes: NumNodes,
) -> (State<AdjArray>, Reducer<AdjArray>) {
    // Sets all original nodes as covered as we only want to cover the sets
    // Reductions/Algorithms should be fine with it as they assume that a removed fixed-node exists
    let mut covered = BitSet::new_with_bits_set(graph.number_of_nodes(), 0..orig_number_nodes);
    let mut domset = DominatingSet::new(graph.number_of_nodes());

    // singleton nodes need to be fixed
    // -> there should never exist a set-node that has degree 0
    domset.fix_nodes(graph.vertices().filter(|&u| graph.degree_of(u) == 0));

    let mut reducer = Reducer::new();
    let mut redundant = BitSet::new_with_bits_set(
        graph.number_of_nodes(),
        orig_number_nodes..graph.number_of_nodes(),
    );

    reducer.apply_rule_exhaustively::<RuleOneReduction<_>>(
        &mut graph,
        &mut domset,
        &mut covered,
        &mut redundant,
    );
    reducer.apply_rule::<LongPathReduction<_>>(
        &mut graph,
        &mut domset,
        &mut covered,
        &mut redundant,
    );

    {
        let csr_edges = graph.extract_csr_repr();
        RuleSubsetReduction::apply_rule(csr_edges, &covered, &mut redundant)
    }

    let mut num_removed_edges = 0;
    redundant.iter_set_bits().for_each(|u| {
        let redundant_neighbors = graph
            .neighbors_of(u)
            .filter(|&v| redundant.get_bit(v))
            .collect_vec();
        num_removed_edges += redundant_neighbors.len();
        for v in redundant_neighbors {
            graph.remove_edge(u, v);
        }
    });

    info!(
        "Subset n ~= {}, m -= {num_removed_edges}, |D| += 0, |covered| += 0",
        redundant.cardinality()
    );

    reducer.apply_rule::<RuleSmallExactReduction<_>>(
        &mut graph,
        &mut domset,
        &mut covered,
        &mut redundant,
    );

    info!("Preprocessing completed");

    (
        State {
            graph,
            domset,
            covered,
            redundant,
        },
        reducer,
    )
}

fn remap_state(org_state: &State<AdjArray>, mapping: &NodeMapper) -> State<CsrGraph> {
    let graph = CsrGraph::from_edges(
        mapping.len() as NumNodes,
        org_state.graph.edges(true).filter_map(|Edge(u, v)| {
            // usually edges between covered nodes should have been removed by reduction rules,
            // but let's make sure
            if org_state.covered.get_bit(u) && org_state.covered.get_bit(v) {
                return None;
            }

            // new id exist since the mapping only drops vertices of degree zero, which
            // by definition do not appear as endpoints of edges!
            let edge =
                Edge(mapping.new_id_of(u).unwrap(), mapping.new_id_of(v).unwrap()).normalized();

            debug_assert!(!edge.is_loop());
            Some(edge)
        }),
    );

    // ensure that we did not miss any nodes / edges
    assert_eq!(
        graph.number_of_nodes(),
        org_state.graph.vertices_with_neighbors().count() as NumNodes
    );

    let covered = BitSet::new_with_bits_set(
        graph.number_of_nodes(),
        mapping.get_filtered_new_ids(org_state.covered.iter_set_bits()),
    );

    let redundant = BitSet::new_with_bits_set(
        graph.number_of_nodes(),
        mapping.get_filtered_new_ids(org_state.redundant.iter_set_bits()),
    );

    let domset = DominatingSet::new(graph.number_of_nodes());

    info!(
        "Mapped graph n={:7} m={:8} covered={:7} redundant={:7} (remaining: {:7})",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        covered.cardinality(),
        redundant.cardinality(),
        graph.number_of_nodes() - redundant.cardinality(),
    );

    State {
        graph,
        domset,
        covered,
        redundant,
    }
}

fn run_search(rng: &mut impl Rng, mapped: State<CsrGraph>, timeout: Option<f64>) -> DominatingSet {
    let State {
        graph,
        domset,
        covered,
        redundant,
    } = mapped;

    let mut search = GreedyReverseSearch::<_, _, 10, 10>::new(
        graph.clone(),
        domset,
        covered.clone(),
        redundant.clone(),
        rng,
    );

    let domset = if let Some(seconds) = timeout {
        search.run_until_timeout(Duration::from_secs_f64(seconds));
        search.best_known_solution()
    } else {
        search.run_to_completion()
    }
    .unwrap();

    assert!(redundant.iter_set_bits().all(|u| !domset.is_in_domset(u)));
    assert!(domset.is_valid_given_previous_cover(&graph, &covered));

    domset
}

fn main() -> anyhow::Result<()> {
    build_pace_logger_for_level(log::LevelFilter::Info);
    signal_handling::initialize();

    #[cfg(feature = "optil")]
    let opts = Opts::default();

    #[cfg(not(feature = "optil"))]
    let opts = Opts::from_args();

    let mut rng = Pcg64Mcg::seed_from_u64(123u64);

    let (input_graph, orig_number_nodes) = load_graph(&opts.input).unwrap();
    info!(
        "Graph loaded n={:7} m={:8}",
        input_graph.number_of_nodes(),
        input_graph.number_of_edges()
    );

    let (mut state, mut reducer) = apply_reduction_rules(input_graph.clone(), orig_number_nodes);

    let mapping = state.graph.cuthill_mckee();
    if mapping.len() > 0 {
        info!("Start greedy");
        // if the reduction rules are VERY successful, no nodes remain
        let mut mapped = remap_state(&state, &mapping);
        greedy_approximation(
            &mapped.graph,
            &mut mapped.domset,
            &mapped.covered,
            &mapped.redundant,
        );
        info!("Greedy found solution size {}", mapped.domset.len());

        info!("Start local search");
        let domset_mapped = run_search(&mut rng, mapped, opts.timeout);
        info!("Local search found solution size {}", domset_mapped.len());

        let size_before = state.domset.len();
        state
            .domset
            .add_nodes(mapping.get_filtered_old_ids(domset_mapped.iter()));
        assert_eq!(size_before + domset_mapped.len(), state.domset.len());
    }

    let mut covered = state.domset.compute_covered(&input_graph);
    reducer.post_process(
        &mut input_graph.clone(),
        &mut state.domset,
        &mut covered,
        &mut state.redundant,
    );

    let switches: Vec<(Node, Node)> = state
        .domset
        .iter()
        .filter_map(|u| {
            if u < orig_number_nodes {
                return None;
            }

            // Must exist as a set-node only exists for non-empty sets
            let first_neighbor = input_graph.neighbors_of(u).next().unwrap();
            assert!(first_neighbor < orig_number_nodes);
            Some((u, first_neighbor))
        })
        .collect_vec();

    for (u, v) in switches {
        // This should never be true as by definition, the graph is bipartite and u can cover only
        // itself (as far as the algorithm knows)
        if !state.domset.is_in_domset(v) {
            state.domset.add_node(v);
        }

        state.domset.remove_node(u);
    }
    assert!(state.domset.iter().all(|u| u < orig_number_nodes));

    let reduction_cover =
        BitSet::new_with_bits_set(input_graph.number_of_nodes(), 0..orig_number_nodes);
    assert!(
        state
            .domset
            .is_valid_given_previous_cover(&input_graph, &reduction_cover)
    );

    // FIXME: remove later
    assert!((orig_number_nodes..input_graph.number_of_nodes()).all(|u| {
        input_graph
            .neighbors_of(u)
            .any(|v| state.domset.is_in_domset(v))
    }));
    if !opts.no_output {
        state.domset.write(std::io::stdout())?;
    }

    Ok(())
}
