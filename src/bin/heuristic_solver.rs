use dss::{
    graph::{
        AdjArray, AdjacencyList, BitSet, Connectivity as _, CsrGraph, CuthillMcKee, Edge, EdgeOps,
        ExtractCsrRepr, Getter, GraphEdgeOrder, GraphFromReader, GraphNodeOrder, NodeMapper,
        NumNodes,
    },
    heuristic::{iterative_greedy::IterativeGreedy, reverse_greedy_search::GreedyReverseSearch},
    io::PaceWriter as _,
    log::build_pace_logger_for_level,
    prelude::{IterativeAlgorithm, TerminatingIterativeAlgorithm},
    reduction::{
        LongPathReduction, Reducer, RuleOneReduction, RuleSmallExactReduction, RuleSubsetReduction,
    },
    utils::{DominatingSet, signal_handling},
};
use log::info;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
use structopt::StructOpt;

#[derive(StructOpt)]
struct Opts {
    #[structopt(short = "i")]
    input: Option<PathBuf>,

    #[structopt(short = "T")]
    timeout: Option<f64>,

    #[structopt(short = "q")]
    no_output: bool,

    #[structopt(short = "l")]
    skip_local_search: bool,

    #[structopt(short = "g", default_value = "10")]
    greedy_timeout: f64,

    #[structopt(short = "G", default_value = "30")]
    greedy_iterations: u64,

    #[structopt(short = "c")]
    dump_ccs_lower_size: Option<NumNodes>,

    #[structopt(short = "C")]
    dump_ccs_upper_size: Option<NumNodes>,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            input: None,
            timeout: None,
            no_output: false,
            skip_local_search: false,
            greedy_timeout: 10.0,
            greedy_iterations: 30,
            dump_ccs_lower_size: None,
            dump_ccs_upper_size: None,
        }
    }
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

fn dump_ccs(graph: &AdjArray, opts: &Opts) {
    if opts.dump_ccs_lower_size.is_none() && opts.dump_ccs_upper_size.is_none() {
        return;
    }

    let size_lb = opts.dump_ccs_lower_size.unwrap_or(8);
    let size_ub = opts.dump_ccs_upper_size.unwrap_or(NumNodes::MAX);

    let graph_name = if let Some(path) = &opts.input {
        String::from(
            path.file_stem()
                .expect("Filestem")
                .to_str()
                .expect("String"),
        )
    } else if let Ok(iid) = std::env::var("STRIDE_IID") {
        format!(
            "{}",
            iid.parse::<u32>()
                .expect("STRIDE_IID needs to be an integer")
        )
    } else {
        panic!("Expect instances name either using -i or via env var STRIDE_IID");
    };

    info!(
        "Start export of CC with sizes between [{size_lb}, {size_ub}] into export_ccs/{graph_name}/"
    );

    let partition = graph.partition_into_connected_components(true);
    let subgraphs = partition.split_into_subgraphs(graph);

    let mut dir_checked = false;

    for (i, (subgraph, _)) in subgraphs.into_iter().enumerate() {
        if !(size_lb..=size_ub).contains(&subgraph.number_of_nodes()) {
            continue;
        }

        let store_path: PathBuf = format!(
            "export_ccs/{graph_name}/n{}_m{}-{i}.gr",
            subgraph.number_of_nodes(),
            subgraph.number_of_edges()
        )
        .into();

        if !dir_checked {
            std::fs::create_dir_all(store_path.parent().unwrap()).unwrap();
            dir_checked = true;
        }
        subgraph.try_write_pace_file(store_path).unwrap();
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
    let mut redundant = BitSet::new(graph.number_of_nodes());

    loop {
        let mut changed = false;

        changed |= reducer.apply_rule::<RuleOneReduction<_>>(
            &mut graph,
            &mut domset,
            &mut covered,
            &mut redundant,
        );
        changed |= reducer.apply_rule::<LongPathReduction<_>>(
            &mut graph,
            &mut domset,
            &mut covered,
            &mut redundant,
        );

        if changed {
            continue;
        }

        {
            let csr_edges = graph.extract_csr_repr();
            RuleSubsetReduction::apply_rule(csr_edges, &covered, &mut redundant);
            if reducer.remove_unnecessary_edges(&mut graph, &covered, &redundant) > 0 {
                continue;
            }
        }

        break;
    }

    reducer.apply_rule::<RuleSmallExactReduction<_>>(
        &mut graph,
        &mut domset,
        &mut covered,
        &mut redundant,
    );

    reducer.report_summary();

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

    #[cfg(feature = "optil")]
    let opts = Opts::default();

    #[cfg(not(feature = "optil"))]
    let opts = Opts::from_args();

    let mut rng = Pcg64Mcg::seed_from_u64(123u64);

    let input_graph = load_graph(&opts.input).unwrap();
    info!(
        "Graph loaded n={:7} m={:8}",
        input_graph.number_of_nodes(),
        input_graph.number_of_edges()
    );

    let (mut state, mut reducer) = apply_reduction_rules(input_graph.clone());
    dump_ccs(&state.graph, &opts);

    let mapping = state.graph.cuthill_mckee();
    if mapping.len() > 0 {
        info!("Start greedy");
        // if the reduction rules are VERY successful, no nodes remain
        let mut mapped = remap_state(&state, &mapping);
        assert!(mapped.domset.is_empty());

        // greedy
        {
            let mut algo =
                IterativeGreedy::new(&mut rng, &mapped.graph, &mapped.covered, &mapped.redundant);

            let mut remaining_iterations = opts.greedy_iterations.max(1);
            let start_time = Instant::now();
            algo.run_while(|_| {
                remaining_iterations -= 1;
                (remaining_iterations > 0)
                    && (start_time.elapsed().as_secs_f64() < opts.greedy_timeout)
            });

            mapped.domset = algo.best_known_solution().unwrap();
        }

        if !opts.skip_local_search {
            info!("Start local search");
            let domset_mapped = run_search(&mut rng, mapped, opts.timeout);
            info!("Local search found solution size {}", domset_mapped.len());

            let size_before = state.domset.len();
            state
                .domset
                .add_nodes(mapping.get_filtered_old_ids(domset_mapped.iter()));
            assert_eq!(size_before + domset_mapped.len(), state.domset.len());
        } else {
            state
                .domset
                .add_nodes(mapping.get_filtered_old_ids(mapped.domset.iter()));
        }
    }

    let mut covered = state.domset.compute_covered(&input_graph);
    reducer.post_process(
        &mut input_graph.clone(),
        &mut state.domset,
        &mut covered,
        &mut state.redundant,
    );

    assert!(state.domset.is_valid(&input_graph));
    if !opts.no_output {
        state.domset.write(std::io::stdout())?;
    }

    Ok(())
}
