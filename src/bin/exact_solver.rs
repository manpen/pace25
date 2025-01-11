use std::{fs::File, path::PathBuf};

use dss::prelude::*;
use structopt::StructOpt;

#[derive(StructOpt)]
pub struct SatSolverOpts {}

#[derive(StructOpt)]
pub enum SatSolverOptsEnum {
    Sat(SatSolverOpts),
}

#[derive(StructOpt)]
#[allow(clippy::enum_variant_names)]
pub enum Commands {
    #[structopt(flatten)]
    SatSolverEnum(SatSolverOptsEnum),
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

fn load_graph(path: &Option<PathBuf>) -> anyhow::Result<AdjMatrix> {
    use dss::prelude::*;

    if let Some(path) = path {
        Ok(AdjMatrix::try_read_pace_file(path)?)
    } else {
        let stdin = std::io::stdin().lock();
        Ok(AdjMatrix::try_read_pace(stdin)?)
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
    let opt = Opts::from_args();

    let graph = load_graph(&opt.instance)?;

    let result = match opt.cmd {
        Commands::SatSolverEnum(_) => dss::exact::sat_solver::solve(&graph, None)?,
    };

    assert!(result.is_valid(&graph), "Produced DS is not valid");
    write_solution(&result, &opt.output)?;

    Ok(())
}
