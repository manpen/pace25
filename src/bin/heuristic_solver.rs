use dss::{
    graph::{
        AdjArray, AdjacencyList, BitSet, CsrGraph, Edge, Getter, GraphFromReader, GraphNodeOrder,
        relabel::cuthill_mckee,
    },
    heuristic::{greedy_approximation, reverse_greedy_search::GreedyReverseSearch},
    kernelization::{KernelizationRule, SubsetRule, rule1::Rule1},
    prelude::{IterativeAlgorithm, TerminatingIterativeAlgorithm},
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

    let mut graph = load_graph(&opts.input).unwrap();

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

    let mut rng = Pcg64Mcg::seed_from_u64(123u64);
    let mut search = GreedyReverseSearch::<_, _, 10, 10>::new(
        &mut graph,
        domset,
        is_perm_covered,
        redundant_mapped,
        &mut rng,
    );

    let solution = if let Some(seconds) = opts.timeout {
        search.run_until_timeout(Duration::from_secs_f64(seconds));
        search.best_known_solution()
    } else {
        search.run_to_completion()
    }
    .unwrap();
    solution.write(std::io::stdout())?;

    Ok(())
}
