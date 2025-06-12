use dss::{
    exact::{highs_advanced::HighsCache, highs_sub},
    graph::*,
    heuristic::{iterative_greedy::IterativeGreedy, reverse_greedy_search::GreedyReverseSearch},
    io::PaceWriter as _,
    log::build_pace_logger_for_level,
    prelude::{IterativeAlgorithm, TerminatingIterativeAlgorithm},
    reduction::*,
    utils::{DominatingSet, signal_handling},
};
use log::info;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use structopt::StructOpt;

#[derive(StructOpt)]
struct Opts {
    #[structopt(short = "i")]
    input: Option<PathBuf>,

    #[structopt(short = "T")]
    timeout: Option<f64>,

    #[structopt(short = "k")]
    preprocess_only: bool,

    #[structopt(short = "q")]
    no_output: bool,

    #[structopt(short = "v")]
    verbose: bool,

    #[structopt(short = "g", default_value = "6")]
    greedy_timeout: f64,

    #[structopt(short = "G", default_value = "20")]
    greedy_iterations: u64,

    #[structopt(short = "a", default_value = "6")]
    ls_attempts: u64,

    #[structopt(short = "p", default_value = "13")]
    ls_presolve_timeout: f64,

    #[structopt(short = "m", default_value = "2")]
    ls_presolve_max_gap: f64,

    #[structopt(
        short = "e",
        default_value = "0",
        help = "Use exact solver for bootstrapping if Greedy is better than; 0=no exact solver in bootstrapping"
    )]
    exact_presolve_threshold: NumNodes,

    #[structopt(
        short = "E",
        default_value = "60",
        help = "Seconds to spend on exact bootstrapping"
    )]
    exact_presolve_time: u64,

    #[structopt(
        short = "b",
        default_value = "120",
        help = "Attempt to stop reductions & bootstrapping before that time; soft limit"
    )]
    bootstrap_time: u64,

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
            greedy_timeout: 5.0,
            greedy_iterations: 20,
            dump_ccs_lower_size: None,
            dump_ccs_upper_size: None,
            verbose: false,
            ls_attempts: 6,
            ls_presolve_timeout: 13.0,
            ls_presolve_max_gap: 2.0,
            exact_presolve_threshold: 0,
            exact_presolve_time: 60,
            bootstrap_time: 120,
            preprocess_only: false,
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

#[derive(Clone)]
struct State<G: Clone> {
    graph: G,
    domset: DominatingSet,
    covered: BitSet,
    never_select: BitSet,
}

fn apply_reduction_rules(mut graph: AdjArray) -> (State<AdjArray>, Reducer<AdjArray>) {
    let mut covered = graph.vertex_bitset_unset();
    let mut domset = DominatingSet::new(graph.number_of_nodes());

    // singleton nodes need to be fixed
    domset.add_nodes(graph.vertices().filter(|&u| graph.degree_of(u) == 0));

    let mut reducer = Reducer::new();
    let mut never_select = BitSet::new(graph.number_of_nodes());

    macro_rules! apply {
        ($rule:expr) => {
            reducer.apply_rule(
                &mut $rule,
                &mut graph,
                &mut domset,
                &mut covered,
                &mut never_select,
            )
        };
    }

    let high_cache = Arc::new(HighsCache::default());

    let mut rule_vertex_cover = RuleVertexCover::new(graph.number_of_nodes());
    let mut rule_one = RuleOneReduction::new(graph.number_of_nodes());
    let mut rule_long_path = LongPathReduction::new(graph.number_of_nodes());
    let mut rule_isolated = RuleIsolatedReduction;
    let mut rule_red_cover = RuleRedundantCover::new(graph.number_of_nodes());
    let mut rule_articulation = RuleArticulationPoint::new_with_cache(high_cache.clone());
    let mut rule_subset = RuleSubsetReduction::new(graph.number_of_nodes());
    let mut rule_red_twin = RuleRedTwin::new(graph.number_of_nodes());
    let mut rule_subset_two = SubsetRuleTwoReduction::new(graph.number_of_nodes());

    loop {
        let mut changed = false;

        changed |= apply!(rule_one);
        changed |= apply!(rule_red_twin);
        changed |= apply!(rule_vertex_cover);
        changed |= apply!(rule_long_path);
        changed |= apply!(rule_isolated);
        changed |= apply!(rule_subset);
        changed |= apply!(rule_red_twin);
        changed |= apply!(rule_red_cover);
        changed |= apply!(rule_articulation);

        if !changed {
            changed |= apply!(rule_subset_two); // it is crucial that the other rules were applied exhaustively
        }

        if changed {
            continue;
        }

        break;
    }

    if graph.number_of_edges() > 0 {
        let mut rule_small_exact = RuleSmallExactReduction::new_with_cache(high_cache.clone());
        apply!(rule_small_exact);
    }

    reducer.report_summary();

    (
        State {
            graph,
            domset,
            covered,
            never_select,
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

    let never_select = BitSet::new_with_bits_set(
        graph.number_of_nodes(),
        mapping.get_filtered_new_ids(org_state.never_select.iter_set_bits()),
    );

    let domset = DominatingSet::new(graph.number_of_nodes());

    info!(
        "Mapped graph n={:7} m={:8} covered={:7} redundant={:7} (remaining: {:7})",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        covered.cardinality(),
        never_select.cardinality(),
        graph.number_of_nodes() - never_select.cardinality(),
    );

    State {
        graph,
        domset,
        covered,
        never_select,
    }
}

type MainHeuristic = GreedyReverseSearch<CsrGraph, 10, 10>;

fn initial_solution_with_greedy(
    rng: &mut impl Rng,
    mapped: &State<CsrGraph>,
    opts: &Opts,
    id: u32,
) -> DominatingSet {
    // Greedy
    assert!(mapped.domset.is_empty());
    info!("Start Greedy");

    let mut algo = IterativeGreedy::new(rng, &mapped.graph, &mapped.covered, &mapped.never_select);

    if id % 2 == 1 {
        algo.set_strategy(dss::heuristic::iterative_greedy::GreedyStrategy::DegreeValue);
    }

    let mut remaining_iterations = opts.greedy_iterations.max(1);
    let start_time = Instant::now();
    algo.run_while(|_| {
        remaining_iterations -= 1;
        (remaining_iterations > 0) && (start_time.elapsed().as_secs_f64() < opts.greedy_timeout)
    });

    algo.best_known_solution().unwrap()
}

fn initial_solution_with_external(
    mapped: &State<CsrGraph>,
    opts: &Opts,
) -> Option<(DominatingSet, bool)> {
    let resp = highs_sub::solve_with_subprocess_find_binary(
        &mapped.graph,
        &mapped.covered,
        &mapped.never_select,
        Duration::from_secs(opts.exact_presolve_time),
        Duration::from_secs(5),
    );

    use dss::exact::highs_advanced::SolverResult;
    match resp {
        Ok(SolverResult::Optimal(items)) => {
            info!(
                "External solver found optimal solution. Size {}",
                items.len()
            );
            let mut ds = DominatingSet::new(mapped.graph.number_of_nodes());
            ds.add_nodes(items);
            Some((ds, true))
        }

        Ok(SolverResult::Suboptimal(items)) => {
            info!("External solver found some solution. Size {}", items.len());
            let mut ds = DominatingSet::new(mapped.graph.number_of_nodes());
            ds.add_nodes(items);
            Some((ds, false))
        }

        Ok(SolverResult::Timeout) | Ok(SolverResult::Infeasible) => {
            info!("External solver found no solution");
            None
        }

        Err(e) => {
            info!("External solver error: {e:?}");
            None
        }
    }
}

fn quick_optimization_run(search: &mut MainHeuristic, opts: &Opts) {
    let start = Instant::now();
    let mut last_update_time = start;
    let mut last_update_score = search.current_score();
    search.run_while(|a| {
        let now = Instant::now();
        if now.duration_since(start) > Duration::from_secs_f64(opts.ls_presolve_timeout) {
            info!(
                "Stop bootstrapping due to time limit. Size: {}",
                a.current_score()
            );
            return false;
        }

        if a.current_score() < last_update_score {
            last_update_score = a.current_score();
            last_update_time = now;
        } else if now.duration_since(last_update_time)
            > Duration::from_secs_f64(opts.ls_presolve_max_gap)
        {
            info!(
                "Stop bootstrapping run due to lack of improvements. Size: {}",
                a.current_score()
            );
            return false;
        }

        true
    });
}

fn run_main_solve(
    rng: &mut impl Rng,
    mapped: &State<CsrGraph>,
    opts: &Opts,
    start_time: Instant,
) -> DominatingSet {
    let mut best_boot: Option<MainHeuristic> = None;
    let mut exact_attempts_left = 1;

    // Phase 1: Run several initial solvers (greedy + a short LS, external solver, etc)
    // for bootstrapping. Select the most promising of them
    for ls_attempt in 0..opts.ls_attempts as u32 {
        let mut initial_solution = None;

        // We may bootstrap using HiGHS ...
        if let Some(best) = best_boot.as_ref()
            && exact_attempts_left > 0
            && best.current_score() < opts.exact_presolve_threshold
            && start_time.elapsed().as_secs() + opts.exact_presolve_time / 2 < opts.bootstrap_time
        {
            if let Some((solution, opt)) = initial_solution_with_external(mapped, opts) {
                if opt {
                    return solution;
                }

                initial_solution = Some(solution);
            }
            exact_attempts_left -= 1;
        }

        // ... or using greedy
        if initial_solution.is_none() {
            initial_solution = Some(initial_solution_with_greedy(rng, mapped, opts, ls_attempt));
        }

        let mut search = MainHeuristic::new(
            mapped.graph.clone(),
            initial_solution.unwrap(),
            mapped.covered.clone(),
            mapped.never_select.clone(),
            rng,
        );

        if opts.verbose {
            info!("Start LS");
            search.enable_verbose_logging();
        }

        quick_optimization_run(&mut search, opts);

        // Update solution
        if best_boot
            .as_ref()
            .is_none_or(|x| x.current_score() > search.current_score())
        {
            info!(
                "Current best phase1 solution with score {} in attempt {ls_attempt}",
                search.current_score()
            );
            best_boot = Some(search);
        }
    }

    let mut best_boot = best_boot.unwrap();

    // Phase 2: Try to optimize best initial solution
    info!("Start final run at score {}", best_boot.current_score());
    if let Some(timeout) = opts.timeout {
        best_boot.run_until_timeout(Duration::from_secs_f64(timeout));
        best_boot.best_known_solution().unwrap()
    } else {
        best_boot.run_to_completion().unwrap()
    }
}

fn main() -> anyhow::Result<()> {
    build_pace_logger_for_level(log::LevelFilter::Info);
    signal_handling::initialize();

    let start_time = Instant::now();

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
        // if the reduction rules are VERY successful, no nodes remain
        let mapped = remap_state(&state, &mapping);

        if opts.preprocess_only {
            // we terminate only after mapping to have some more statistics in the logs
            return Ok(());
        }

        // run solver
        let domset_mapped = run_main_solve(&mut rng, &mapped, &opts, start_time);
        info!("Local search found solution size {}", domset_mapped.len());

        // integrate mapped solution into global state
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
        &mut state.never_select,
    );

    assert!(state.domset.is_valid(&input_graph));
    if !opts.no_output {
        state.domset.write(std::io::stdout())?;
    }

    Ok(())
}
