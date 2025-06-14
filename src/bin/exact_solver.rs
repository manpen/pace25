use std::time::Duration;
use std::{fs::File, path::PathBuf, sync::Arc};

use dss::exact::{ext_maxsat, highs_advanced::*, search_binary_path};

use dss::reduction::{
    RuleIsolatedReduction, RuleRedundantCover, RuleVertexCover, SubsetRuleTwoReduction,
};
#[allow(unused_imports)]
use dss::{exact::naive::naive_solver, log::build_pace_logger_for_level, prelude::*, reduction::*};
use log::info;
use structopt::StructOpt;

#[derive(StructOpt)]
pub struct SatSolverOpts {
    #[structopt(short = "p")]
    conc_solvers: bool,
}

#[allow(clippy::derivable_impls)]
impl Default for SatSolverOpts {
    fn default() -> Self {
        Self {
            #[cfg(feature = "conc_solvers")]
            conc_solvers: true,
            #[cfg(not(feature = "conc_solvers"))]
            conc_solvers: false,
        }
    }
}

#[derive(StructOpt)]
pub enum SatSolverOptsEnum {
    Sat(SatSolverOpts),
}

#[derive(StructOpt)]
pub struct NaiveSolver {}

#[derive(StructOpt, Default)]

pub enum NaiveSolverOptsEnum {
    #[default]
    Naive,
}

#[derive(StructOpt)]
pub struct HighsSolver {}

#[derive(StructOpt, Default)]

pub enum HighsSolverSolverOptsEnum {
    #[default]
    Highs,
}

#[derive(StructOpt)]
#[allow(clippy::enum_variant_names)]
pub enum Commands {
    #[structopt(flatten)]
    SatSolverEnum(SatSolverOptsEnum),
    #[structopt(flatten)]
    NaiveSolverEnum(NaiveSolverOptsEnum),
    #[structopt(flatten)]
    HighsSolverEnum(HighsSolverSolverOptsEnum),
}

impl Default for Commands {
    fn default() -> Self {
        Commands::SatSolverEnum(SatSolverOptsEnum::Sat(Default::default()))
    }
}

#[derive(Default, StructOpt)]
struct Opts {
    #[structopt(short, long)]
    instance: Option<PathBuf>,

    #[structopt(short, long)]
    output: Option<PathBuf>,

    #[structopt(subcommand)]
    cmd: Option<Commands>,

    #[structopt(short = "q")]
    no_output: bool,
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

fn write_solution(ds: &DominatingSet, path: &Option<PathBuf>) -> anyhow::Result<()> {
    if let Some(path) = path {
        let file = File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        ds.write(writer)?;
    } else {
        let writer = std::io::stdout();
        ds.write(writer)?;
    }

    Ok(())
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

fn remap_state_to_csr(org_state: &State<AdjArray>, mapping: &NodeMapper) -> CsrGraph {
    CsrGraph::from_edges(
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
    )
}

fn remap_state_to_adj(org_state: &State<AdjArray>, mapping: &NodeMapper) -> AdjArray {
    AdjArray::from_edges(
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
    )
}

fn args_uwrmaxsat() -> Vec<String> {
    vec![
        "-v0".into(),
        "-no-bin".into(),
        "-no-sat".into(),
        "-no-par".into(),
        "-maxpre-time=60".into(),
        "-scip-cpu=800".into(),
        "-scip-delay=300".into(),
        "-m".into(),
        "-bm".into(),
    ]
}

fn args_evalmaxsat() -> Vec<String> {
    vec!["--TCT".into(), "1100".into()]
}

fn solve_staged_maxsat(
    graph: &(impl StaticGraph + SelfLoop),
    covered: &BitSet,
    never_select: &BitSet,
) -> anyhow::Result<DominatingSet> {
    if let Ok(solver_binary) = search_binary_path(&PathBuf::from("uwrmaxsat"))
        && let Ok(d) = ext_maxsat::solve(
            &solver_binary,
            args_uwrmaxsat(),
            graph,
            covered,
            never_select,
            Some(Duration::from_secs(600)),
        )
    {
        return Ok(d);
    }

    if let Ok(solver_binary) = search_binary_path(&PathBuf::from("EvalMaxSAT_bin"))
        && let Ok(d) = ext_maxsat::solve(
            &solver_binary,
            args_evalmaxsat(),
            graph,
            covered,
            never_select,
            None,
        )
    {
        return Ok(d);
    }

    anyhow::bail!("No solver succeeded");
}

fn solve_staged_maxsat_concurrent(
    graph: &(impl StaticGraph + SelfLoop),
    covered: &BitSet,
    never_select: &BitSet,
) -> anyhow::Result<DominatingSet> {
    let mut solvers = vec![];

    if let Ok(solver_binary) = search_binary_path(&PathBuf::from("uwrmaxsat")) {
        solvers.push((
            solver_binary,
            args_uwrmaxsat(),
            Some(Duration::from_secs(1200)),
        ));
    }

    if let Ok(solver_binary) = search_binary_path(&PathBuf::from("EvalMaxSAT_bin")) {
        solvers.push((solver_binary, args_evalmaxsat(), None));
    }

    if solvers.is_empty() {
        anyhow::bail!("No solvers found");
    }

    if solvers.len() == 1 {
        solvers[0].2 = None;
    }

    ext_maxsat::solve_multiple(solvers, graph, covered, never_select)
}

fn map_and_solve_kernel_exact(
    state: &State<AdjArray>,
    mapping: &NodeMapper,
    opts: &mut Opts,
) -> DominatingSet {
    let n = mapping.len();

    let covered = BitSet::new_with_bits_set(
        n,
        mapping.get_filtered_new_ids(state.covered.iter_set_bits()),
    );

    let never_select = BitSet::new_with_bits_set(
        n,
        mapping.get_filtered_new_ids(state.never_select.iter_set_bits()),
    );

    let cmd = std::mem::take(&mut opts.cmd).unwrap_or_default();
    match cmd {
        Commands::SatSolverEnum(SatSolverOptsEnum::Sat(o)) => {
            let csr_graph = remap_state_to_csr(state, mapping);
            assert_eq!(csr_graph.number_of_nodes(), n);
            if o.conc_solvers {
                solve_staged_maxsat_concurrent(&csr_graph, &covered, &never_select).unwrap()
            } else {
                solve_staged_maxsat(&csr_graph, &covered, &never_select).unwrap()
            }
        }
        Commands::NaiveSolverEnum(_) => {
            let adj_graph = remap_state_to_adj(state, mapping);
            assert_eq!(adj_graph.number_of_nodes(), n);
            info!("Start Naive Solver");
            naive_solver(&adj_graph, &covered, &never_select, None, None).unwrap()
        }
        Commands::HighsSolverEnum(_) => {
            let adj_graph = remap_state_to_adj(state, mapping);
            assert_eq!(adj_graph.number_of_nodes(), n);
            info!("Start Highs Solver");

            let mut solver = HighsDominatingSetSolver::new(adj_graph.number_of_nodes());
            let problem = solver.build_problem(&adj_graph, &covered, &never_select, unit_weight);
            let local_sol = problem.solve_exact(None).take_solution().unwrap();
            let mut domset = DominatingSet::new(n);
            domset.add_nodes(local_sol.iter().copied());
            domset
        }
    }
}

fn main() -> anyhow::Result<()> {
    build_pace_logger_for_level(log::LevelFilter::Info);
    #[cfg(feature = "optil")]
    let mut opts = Opts::default();

    #[cfg(not(feature = "optil"))]
    let mut opts = Opts::from_args();

    let input_graph = load_graph(&opts.instance)?;

    let (mut state, mut reducer) = apply_reduction_rules(input_graph.clone());

    let mapping = state.graph.cuthill_mckee();
    if mapping.len() > 0 {
        // if the reduction rules are VERY successful, no nodes remain
        let domset_mapped = map_and_solve_kernel_exact(&state, &mapping, &mut opts);

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

    assert!(
        state.domset.is_valid(&input_graph),
        "Produced DS is not valid"
    );
    if opts.no_output {
        info!("Final solution size: {}", state.domset.len());
    } else {
        write_solution(&state.domset, &opts.output)?;
    }

    Ok(())
}
