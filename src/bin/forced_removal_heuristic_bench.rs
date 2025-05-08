use std::{
    fs::File,
    io::{BufReader, Write},
    path::PathBuf,
    time::Instant,
};

use dss::{
    graph::CsrGraph,
    heuristic::{
        greedy_approximation, reverse_greedy_search::GreedyReverseSearch, ForcedRemovalRuleType,
    },
    io::GraphPaceReader,
    prelude::IterativeAlgorithm,
    utils::signal_handling,
};
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

use structopt::StructOpt;

#[derive(StructOpt)]
struct Args {
    #[structopt(long, parse(from_os_str))]
    path: PathBuf,

    #[structopt(short, long, default_value = "0")]
    forced: u8,
}

fn main() -> anyhow::Result<()> {
    signal_handling::initialize();

    let args = Args::from_args();

    let mut timer = Instant::now();

    let mut graph = CsrGraph::try_read_pace(BufReader::new(File::open(args.path)?)).unwrap();

    let read_time = timer.elapsed().as_millis();
    timer = Instant::now();

    let mut rng = Pcg64Mcg::seed_from_u64(123u64);
    let domset = greedy_approximation(&graph);

    let greedy_time = timer.elapsed().as_millis();
    timer = Instant::now();

    let rule = match args.forced {
        0 => ForcedRemovalRuleType::DMS,
        1 => ForcedRemovalRuleType::BFS2,
        2 => ForcedRemovalRuleType::BFS3,
        3 => ForcedRemovalRuleType::BFS4,
        4 => ForcedRemovalRuleType::BFSP2,
        5 => ForcedRemovalRuleType::BFSP3,
        6 => ForcedRemovalRuleType::BFSP4,
        _ => panic!("Only 7 forced rules are implemented"),
    };
    let mut search = GreedyReverseSearch::<_, _, 10, 10>::new(&mut graph, domset, &mut rng, rule);

    let init_time = timer.elapsed().as_millis();
    timer = Instant::now();

    let mut counter = 1usize;
    while !signal_handling::received_ctrl_c() && !search.is_completed() {
        counter += 1;
        search.step();
    }

    let search_time = timer.elapsed().as_millis();

    let size = search.best_known_solution().unwrap().len();

    writeln!(
        std::io::stdout(),
        "{read_time},{greedy_time},{init_time},{counter},{search_time},{size}"
    )?;

    Ok(())
}
