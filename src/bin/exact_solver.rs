use std::{fs::File, path::PathBuf};

use dss::{
    exact::{naive::naive_solver, sat_solver::SolverBackend},
    log::build_pace_logger_for_level,
    prelude::*,
    reduction::{
        LongPathReduction, Reducer, RuleOneReduction, RuleSmallExactReduction, RuleSubsetReduction,
    },
};
use itertools::Itertools;
use structopt::StructOpt;

#[derive(StructOpt)]
pub struct SatSolverOpts {}

#[derive(StructOpt)]
pub enum SatSolverOptsEnum {
    Sat(SatSolverOpts),
}

#[derive(StructOpt)]
pub struct NaiveSolver {}

#[derive(StructOpt)]

pub enum NaiveSolverOptsEnum {
    Naive,
}

#[derive(StructOpt)]
#[allow(clippy::enum_variant_names)]
pub enum Commands {
    #[structopt(flatten)]
    SatSolverEnum(SatSolverOptsEnum),
    #[structopt(flatten)]
    NaiveSolverEnum(NaiveSolverOptsEnum),
}

#[derive(StructOpt)]
struct Opts {
    #[structopt(short, long)]
    instance: Option<PathBuf>,

    #[structopt(short, long)]
    output: Option<PathBuf>,

    #[structopt(subcommand)]
    cmd: Commands,
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
    let opt = Opts::from_args();

    let mut graph = load_graph(&opt.instance)?;
    let org_graph = graph.clone();

    let mut covered = graph.vertex_bitset_unset();
    let mut solution = DominatingSet::new(graph.number_of_nodes());

    // singleton nodes need to be fixed
    solution.fix_nodes(graph.vertices().filter(|&u| graph.degree_of(u) == 0));

    let mut reducer = Reducer::new();

    reducer.apply_rule_exhaustively::<RuleOneReduction<_>>(&mut graph, &mut solution, &mut covered);
    reducer.apply_rule::<LongPathReduction<_>>(&mut graph, &mut solution, &mut covered);
    reducer.apply_rule::<RuleSmallExactReduction<_>>(&mut graph, &mut solution, &mut covered);

    let redundant = {
        let csr_graph = CsrGraph::from_edges(graph.number_of_nodes(), graph.edges(true));
        let csr_edges = csr_graph.extract_csr_repr();
        RuleSubsetReduction::apply_rule(csr_edges, &covered, &mut solution)
    };

    let mut solution = {
        let csr_graph = CsrGraph::from_edges(graph.number_of_nodes(), graph.edges(true));
        match opt.cmd {
            Commands::SatSolverEnum(SatSolverOptsEnum::Sat(_)) => dss::exact::sat_solver::solve(
                &csr_graph,
                covered,
                Some(solution),
                SolverBackend::MAXSAT,
            )?,
            Commands::NaiveSolverEnum(_) => {
                let local_sol = naive_solver(&graph, &covered, &redundant, None).unwrap();
                solution.add_nodes(local_sol.iter());
                solution
            }
        }
    };

    let mut covered = solution.compute_covered(&org_graph);
    reducer.post_process(&mut graph, &mut solution, &mut covered);

    assert!(solution.is_valid(&org_graph), "Produced DS is not valid");
    write_solution(&solution, &opt.output)?;

    Ok(())
}
