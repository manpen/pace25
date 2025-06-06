use std::{fs::File, path::PathBuf};

use dss::{exact::highs_advanced::*, reduction::*};

use dss::reduction::{
    RuleIsolatedReduction, RuleRedundantCover, RuleVertexCover, SubsetRuleTwoReduction,
};
#[allow(unused_imports)]
use dss::{
    exact::{naive::naive_solver, sat_solver::SolverBackend},
    log::build_pace_logger_for_level,
    prelude::*,
    reduction::{
        LongPathReduction, Reducer, RuleOneReduction, RuleSmallExactReduction, RuleSubsetReduction,
    },
};
use log::info;
use structopt::StructOpt;

#[derive(StructOpt)]
pub struct SatSolverOpts {}

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
        Commands::HighsSolverEnum(Default::default())
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

fn main() -> anyhow::Result<()> {
    build_pace_logger_for_level(log::LevelFilter::Info);
    #[cfg(feature = "optil")]
    let opts = Opts::default();

    #[cfg(not(feature = "optil"))]
    let opts = Opts::from_args();

    let mut graph = load_graph(&opts.instance)?;
    let org_graph = graph.clone();

    let mut covered = graph.vertex_bitset_unset();
    let mut domset = DominatingSet::new(graph.number_of_nodes());

    // singleton nodes need to be fixed
    domset.fix_nodes(graph.vertices().filter(|&u| graph.degree_of(u) == 0));

    let mut reducer = Reducer::new();
    let mut never_select = BitSet::new(graph.number_of_nodes());

    let mut rule_vertex_cover = RuleVertexCover::new(graph.number_of_nodes());
    let mut rule_one = RuleOneReduction::new(graph.number_of_nodes());
    let mut rule_long_path = LongPathReduction;
    let mut rule_isolated = RuleIsolatedReduction;
    let mut rule_redundant = RuleRedundantCover::new(graph.number_of_nodes());
    let mut rule_articulation = RuleArticulationPoint::new(graph.number_of_nodes());
    let mut rule_subset_two = SubsetRuleTwoReduction::new(graph.number_of_nodes());

    loop {
        let mut changed = false;

        changed |= reducer.apply_rule(
            &mut rule_vertex_cover,
            &mut graph,
            &mut domset,
            &mut covered,
            &mut never_select,
        );

        changed |= reducer.apply_rule(
            &mut rule_one,
            &mut graph,
            &mut domset,
            &mut covered,
            &mut never_select,
        );

        changed |= reducer.apply_rule(
            &mut rule_long_path,
            &mut graph,
            &mut domset,
            &mut covered,
            &mut never_select,
        );

        changed |= reducer.apply_rule(
            &mut rule_isolated,
            &mut graph,
            &mut domset,
            &mut covered,
            &mut never_select,
        );

        changed |= reducer.apply_rule(
            &mut rule_redundant,
            &mut graph,
            &mut domset,
            &mut covered,
            &mut never_select,
        );

        if changed {
            continue;
        }

        changed |= reducer.apply_rule(
            &mut rule_articulation,
            &mut graph,
            &mut domset,
            &mut covered,
            &mut never_select,
        );

        if changed {
            continue;
        }

        changed |= reducer.apply_rule(
            &mut rule_subset_two,
            &mut graph,
            &mut domset,
            &mut covered,
            &mut never_select,
        );

        if changed {
            continue;
        }

        break;
    }

    let mut rule_small_exact = RuleSmallExactReduction;

    if graph.number_of_edges() > 0 {
        reducer.apply_rule(
            &mut rule_small_exact,
            &mut graph,
            &mut domset,
            &mut covered,
            &mut never_select,
        );
    }

    let mut domset = if graph.number_of_edges() > 0 {
        let csr_graph = CsrGraph::from_edges(graph.number_of_nodes(), graph.edges(true));
        let cmd = opts.cmd.unwrap_or_default();

        match cmd {
            Commands::SatSolverEnum(SatSolverOptsEnum::Sat(_)) => dss::exact::sat_solver::solve(
                &csr_graph,
                covered,
                Some(domset),
                SolverBackend::MAXSAT,
            )?,
            Commands::NaiveSolverEnum(_) => {
                info!("Start Naive Solver");
                signal_handling::initialize();
                let local_sol = naive_solver(&graph, &covered, &never_select, None, None).unwrap();
                domset.add_nodes(local_sol.iter());
                domset
            }
            Commands::HighsSolverEnum(_) => {
                info!("Start Highs Solver");

                let mut solver = HighsDominatingSetSolver::new(graph.number_of_nodes());
                let problem = solver.build_problem(&graph, &covered, &never_select, unit_weight);
                let local_sol = problem.solve_exact(None).take_solution().unwrap();
                domset.add_nodes(local_sol.iter().cloned());
                domset
            }
        }
    } else {
        domset
    };

    let mut covered = domset.compute_covered(&org_graph);
    reducer.post_process(&mut graph, &mut domset, &mut covered, &mut never_select);

    assert!(domset.is_valid(&org_graph), "Produced DS is not valid");
    write_solution(&domset, &opts.output)?;

    Ok(())
}
