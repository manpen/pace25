use dss::{
    graph::{
        AdjArray, AdjacencyList, BitSet, CsrGraph, CuthillMcKee, Edge, EdgeOps, ExtractCsrRepr,
        Getter, GraphEdgeOrder, GraphFromReader, GraphNodeOrder, NodeMapper, NumNodes,
    },
    heuristic::{greedy_approximation, reverse_greedy_search::GreedyReverseSearch},
    log::build_pace_logger_for_level,
    prelude::{IterativeAlgorithm, TerminatingIterativeAlgorithm},
    reduction::{
        LongPathReduction, Reducer, RuleOneReduction, RuleSmallExactReduction, RuleSubsetReduction,
    },
    utils::{DominatingSet, signal_handling},
};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use std::{path::PathBuf, time::Duration};
use structopt::{StructOpt, clap};

#[derive(StructOpt, Default)]
struct Opts {
    #[structopt(short = "i")]
    input: Option<PathBuf>,

    #[structopt(short = "T")]
    timeout: Option<f64>,
}

fn load_graph(path: &Option<PathBuf>) -> anyhow::Result<AdjArray> {
    use dss::prelude::*;

    if let Some(path) = path {
        Ok(AdjArray::try_read_pace_file(path)?)
    } else {
        let stdin = std::io::stdin().lock();
        Ok(AdjArray::try_read_pace(stdin)?)
    }
}

struct State<G> {
    graph: G,
    domset: DominatingSet,
    covered: BitSet,
    redundant: BitSet,
}

fn apply_reduction_rules(mut graph: AdjArray) -> (State<AdjArray>, Reducer<AdjArray>) {
    let mut covered = graph.vertex_bitset_unset();
    let mut domset = DominatingSet::new(graph.number_of_nodes());

    // singleton nodes need to be fixed
    domset.fix_nodes(graph.vertices().filter(|&u| graph.degree_of(u) == 0));

    let mut reducer = Reducer::new();
    reducer.apply_rule_exhaustively::<RuleOneReduction<_>>(&mut graph, &mut domset, &mut covered);
    reducer.apply_rule::<LongPathReduction<_>>(&mut graph, &mut domset, &mut covered);
    reducer.apply_rule::<RuleSmallExactReduction<_>>(&mut graph, &mut domset, &mut covered);
    let redundant = graph.vertex_bitset_unset(); // TODO: Add SubsetRule

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
            if !org_state.covered.get_bit(u) && !org_state.covered.get_bit(v) {
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
    assert_eq!(graph.number_of_edges(), org_state.graph.number_of_edges(),);

    // remap bitsets
    assert_eq!(
        mapping
            .get_filtered_new_ids(org_state.redundant.iter_set_bits())
            .count(),
        0
    );

    let covered = BitSet::new_with_bits_set(
        graph.number_of_nodes(),
        mapping.get_filtered_new_ids(org_state.covered.iter_set_bits()),
    );

    let mut domset = DominatingSet::new(graph.number_of_nodes());

    let redundant = {
        let csr_edges = graph.extract_csr_repr();
        RuleSubsetReduction::apply_rule(csr_edges, &covered, &mut domset)
    };

    State {
        graph,
        domset,
        covered,
        redundant,
    }
}

fn run_search(rng: &mut impl Rng, mapped: State<CsrGraph>, timeout: Option<f64>) -> DominatingSet {
    let State {
        mut graph,
        domset,
        covered,
        redundant,
    } = mapped;

    let mut search = GreedyReverseSearch::<_, _, 10, 10>::new(
        &mut graph,
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

    // while I love to have an error message here, this clashes with optil.io
    // hence, we ignore parsing errors and only terminate if the user explictly asks for help
    let opts = match Opts::from_args_safe() {
        Ok(x) => x,
        Err(e) if e.kind == clap::ErrorKind::HelpDisplayed => return Ok(()),
        _ => Default::default(),
    };

    let mut rng = Pcg64Mcg::seed_from_u64(123u64);

    let input_graph = load_graph(&opts.input).unwrap();

    let (mut state, mut reducer) = apply_reduction_rules(input_graph.clone());

    let mapping = state.graph.cuthill_mckee();
    if mapping.len() > 0 {
        // if the reduction rules are VERY successful, no nodes remain
        let mut mapped = remap_state(&state, &mapping);
        greedy_approximation(
            &mapped.graph,
            &mut mapped.domset,
            &mapped.covered,
            &mapped.redundant,
        );
        let domset_mapped = run_search(&mut rng, mapped, opts.timeout);

        let size_before = state.domset.len();
        state
            .domset
            .add_nodes(mapping.get_filtered_old_ids(domset_mapped.iter()));
        assert_eq!(size_before + domset_mapped.len(), state.domset.len());
    }

    let mut covered = state.domset.compute_covered(&input_graph);
    reducer.post_process(&mut input_graph.clone(), &mut state.domset, &mut covered);

    assert!(state.domset.is_valid(&input_graph));
    state.domset.write(std::io::stdout())?;

    Ok(())
}
