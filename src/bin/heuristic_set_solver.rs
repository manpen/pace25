use dss::{
    graph::{CsrGraph, NumNodes}, heuristic::{greedy_approximation, reverse_greedy_search::GreedyReverseSearch}, io::set_reader::SetPaceReader, prelude::{IterativeAlgorithm, TerminatingIterativeAlgorithm}, utils::signal_handling
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

fn load_graph(path: &Option<PathBuf>) -> anyhow::Result<(CsrGraph, NumNodes)> {
    use dss::prelude::*;

    if let Some(path) = path {
        Ok(CsrGraph::try_read_set_pace_file(path)?)
    } else {
        let stdin = std::io::stdin().lock();
        Ok(CsrGraph::try_read_set_pace(stdin)?)
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

    let (mut graph, num_orig_nodes) = load_graph(&opts.input).unwrap();

    let mut rng = Pcg64Mcg::seed_from_u64(123u64);
    let domset = greedy_approximation(&graph);

    let mut search = GreedyReverseSearch::<_, _, 8, 10>::new(&mut graph, domset, &mut rng);

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
