use dss::{
    graph::{
        AdjArray, AdjacencyList, BitSet, CsrGraph, CuthillMcKee, Edge, EdgeOps, Getter,
        GraphEdgeOrder, GraphFromReader, GraphNodeOrder, NumNodes,
    },
    heuristic::{greedy_approximation, reverse_greedy_search::GreedyReverseSearch},
    kernelization::{KernelizationRule, SubsetRule},
    prelude::{IterativeAlgorithm, TerminatingIterativeAlgorithm},
    reduction::{LongPathReduction, Reducer, RuleOneReduction},
    utils::{DominatingSet, signal_handling},
};
use rand::SeedableRng;
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

fn main() -> anyhow::Result<()> {
    signal_handling::initialize();

    // while I love to have an error message here, this clashes with optil.io
    // hence, we ignore parsing errors and only terminate if the user explictly asks for help
    let opts = match Opts::from_args_safe() {
        Ok(x) => x,
        Err(e) if e.kind == clap::ErrorKind::HelpDisplayed => return Ok(()),
        _ => Default::default(),
    };

    let input_graph = load_graph(&opts.input).unwrap();
    let mut graph = input_graph.clone();
    let mut covered = graph.vertex_bitset_unset();
    let mut domset = DominatingSet::new(graph.number_of_nodes());

    // singleton nodes need to be fixed
    domset.fix_nodes(graph.vertices().filter(|&u| graph.degree_of(u) == 0));

    let mut reducer = Reducer::new();
    reducer.apply_rule::<RuleOneReduction<_>>(&mut graph, &mut domset, &mut covered);
    reducer.apply_rule::<LongPathReduction<_>>(&mut graph, &mut domset, &mut covered);
    let redundant = SubsetRule::apply_rule(&mut graph, &mut domset);

    let mapping = graph.cuthill_mckee();
    let orig_graph = graph;

    let mut graph_mapped = CsrGraph::from_edges(
        mapping.len() as NumNodes,
        orig_graph.edges(true).map(|Edge(u, v)| {
            // new id exist since the mapping only drops vertices of degree zero, which
            // by definition do not appear as endpoints of edges!
            let edge =
                Edge(mapping.new_id_of(u).unwrap(), mapping.new_id_of(v).unwrap()).normalized();
            debug_assert!(!edge.is_loop());
            edge
        }),
    );

    // ensure that we did not miss any nodes / edges
    assert_eq!(
        graph_mapped.number_of_nodes(),
        orig_graph.vertices_with_neighbors().count() as NumNodes
    );
    assert_eq!(graph_mapped.number_of_edges(), orig_graph.number_of_edges(),);

    let redundant_mapped = BitSet::new_with_bits_set(
        graph_mapped.number_of_nodes(),
        redundant
            .iter_set_bits()
            .filter_map(|u| mapping.new_id_of(u)),
    );

    let is_perm_covered = BitSet::new_with_bits_set(
        graph_mapped.number_of_nodes(),
        covered.iter_set_bits().filter_map(|u| mapping.new_id_of(u)),
    );

    let mut domset_mapped = DominatingSet::new(graph_mapped.number_of_nodes());
    greedy_approximation(&graph_mapped, &mut domset_mapped, &redundant_mapped);

    let mut rng = Pcg64Mcg::seed_from_u64(123u64);
    let mut search = GreedyReverseSearch::<_, _, 10, 10>::new(
        &mut graph_mapped,
        domset_mapped,
        is_perm_covered,
        redundant_mapped.clone(),
        &mut rng,
    );

    let domset_mapped = if let Some(seconds) = opts.timeout {
        search.run_until_timeout(Duration::from_secs_f64(seconds));
        search.best_known_solution()
    } else {
        search.run_to_completion()
    }
    .unwrap();

    assert!(
        redundant_mapped
            .iter_set_bits()
            .all(|u| !domset.is_in_domset(u))
    );

    domset.add_nodes(mapping.get_old_ids(domset_mapped.iter()));
    let mut covered = domset.compute_covered(&input_graph);
    reducer.post_process(&mut input_graph.clone(), &mut domset, &mut covered);

    assert!(domset.is_valid(&input_graph));
    domset.write(std::io::stdout())?;

    Ok(())
}
