use std::{fs::File, path::PathBuf};

#[allow(unused_imports)]
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
    let mut redundant = BitSet::new(graph.number_of_nodes());

    loop {
        let mut changed = false;

        changed |= reducer.apply_rule::<RuleOneReduction<_>>(
            &mut graph,
            &mut solution,
            &mut covered,
            &mut redundant,
        );
        changed |= reducer.apply_rule::<LongPathReduction<_>>(
            &mut graph,
            &mut solution,
            &mut covered,
            &mut redundant,
        );

        if changed {
            continue;
        }

        if true {
            let csr_edges = graph.extract_csr_repr();
            RuleSubsetReduction::apply_rule(csr_edges, &covered, &mut redundant);
            assert!(!solution.iter().any(|u| redundant.get_bit(u)));
            if reducer.remove_unnecessary_edges(&mut graph, &covered, &redundant) > 0 {
                continue;
            }
        }

        break;
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

    if graph.number_of_edges() > 0 {
        reducer.apply_rule::<RuleSmallExactReduction<_>>(
            &mut graph,
            &mut solution,
            &mut covered,
            &mut redundant,
        );
    }

    let mut solution = if graph.number_of_edges() > 0 {
        let csr_graph = CsrGraph::from_edges(graph.number_of_nodes(), graph.edges(true));
        match opt.cmd {
            Commands::SatSolverEnum(SatSolverOptsEnum::Sat(_)) => dss::exact::sat_solver::solve(
                &csr_graph,
                covered,
                Some(solution),
                SolverBackend::MAXSAT,
            )?,
            Commands::NaiveSolverEnum(_) => {
                let local_sol = naive_solver(&graph, &covered, &redundant, None, None).unwrap();
                solution.add_nodes(local_sol.iter());
                solution
            }
        }
    } else {
        solution
    };

    let mut covered = solution.compute_covered(&org_graph);
    reducer.post_process(&mut graph, &mut solution, &mut covered, &mut redundant);

    assert!(solution.is_valid(&org_graph), "Produced DS is not valid");
    write_solution(&solution, &opt.output)?;

    Ok(())
}
