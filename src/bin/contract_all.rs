use log::LevelFilter;
use serde::Serialize;
use std::{fs::File, io::BufWriter, io::Write, path::PathBuf, time::Duration};
use structopt::StructOpt;
use tww::{
    log::build_pace_logger_for_level,
    prelude::{contract_all::ContractAll, *},
};

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short = "g", parse(from_os_str))]
    graph: PathBuf,

    #[structopt(short = "k")]
    tww: NumNodes,

    #[structopt(short = "o", parse(from_os_str))]
    output: Option<PathBuf>,

    #[structopt(short = "s")]
    skip_if_output_exists: bool,

    #[structopt(short = "t")]
    timeout_ms: Option<u64>,
}

#[derive(Serialize)]
struct Frame {
    nodes: NumNodes,
    edges: NumEdges,
    tww: NumNodes,
    black_edges: Vec<(Node, Node)>,
    red_edges: Vec<(Node, Node)>,
    contractions: Vec<(Node, Node)>,
}

fn main() {
    build_pace_logger_for_level(LevelFilter::Info);

    let opt = Opt::from_args();
    if opt.skip_if_output_exists {
        if let Some(path) = opt.output.as_ref() {
            if path.exists() {
                eprintln!("Output file already exists and --skip-if-output-exists was passed");
                return;
            }
        } else {
            panic!("--skip-if-output-exists requires --output to be set");
        }
    }

    let graph = AdjArray::try_read_pace_file(opt.graph).unwrap();

    let timeout = opt.timeout_ms.map(Duration::from_millis);
    let mut solver = ContractAll::<AdjArray>::new(opt.tww, timeout);

    match solver.solve(&graph) {
        Ok(()) => {}
        Err(contract_all::Error::Infeasible) => {
            panic!("Instance could not be solved for the TWW provided")
        }
        Err(contract_all::Error::Timeout) => {
            eprintln!("Timeout reached.");
            return;
        }
    }

    let result = solver.get_result();

    let mut output_file = opt
        .output
        .map(|path| BufWriter::new(File::create(path).expect("Could not open output file")));

    for (graph, contractions) in result {
        let edges: Vec<_> = graph.ordered_colored_edges(true).collect();
        let num_nodes = graph.vertices_with_neighbors().last().unwrap() + 1;

        let frame = Frame {
            black_edges: edges
                .iter()
                .filter_map(|ColoredEdge(u, v, c)| (c.is_black()).then_some((*u, *v)))
                .collect(),
            red_edges: edges
                .iter()
                .filter_map(|ColoredEdge(u, v, c)| (c.is_red()).then_some((*u, *v)))
                .collect(),
            nodes: num_nodes,
            edges: graph.number_of_edges(),
            tww: opt.tww,
            contractions,
        };

        if let Some(writer) = output_file.as_mut() {
            writeln!(*writer, "{}", serde_json::to_string(&frame).unwrap())
                .expect("Could not write to result file");
        } else {
            println!("{}", serde_json::to_string(&frame).unwrap());
        }
    }
}
