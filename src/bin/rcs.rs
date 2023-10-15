use log::{debug, info, LevelFilter};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use serde::Serialize;
use std::{
    collections::HashSet,
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
    time::Instant,
};
use structopt::StructOpt;
use tww::{log::build_pace_logger_for_level, prelude::*, utils::ContractionSequence};

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short = "g", parse(from_os_str))]
    graph: PathBuf,

    #[structopt(short = "i", parse(from_os_str))]
    sequences: Vec<PathBuf>,

    #[structopt(short = "o", parse(from_os_str))]
    output: PathBuf,

    #[structopt(short = "s")]
    random_seed: Option<u64>,

    #[structopt(short = "n")]
    max_sequences: Option<usize>,

    #[structopt(short = "t")]
    timeout_ms: Option<u64>,
}

type CSDigest = digest::Output<sha2::Sha256>;

struct State<G: FullfledgedGraph> {
    graph: G,

    /// invariant: all sequences stored here have acceptable tww
    sequences: Vec<ContractionSequence>,

    /// stores sequences that have been previously been processed and do not have to be addressed again
    cache: HashSet<CSDigest>,

    first_pending: usize,
    tww: Node,

    urng: Pcg64Mcg,
}

#[derive(Serialize)]
struct Export<'a> {
    seconds_shuffled: f64,
    seqs_found: usize,
    seqs_processed: usize,
    seqs_cached: usize,
    solutions: &'a [ContractionSequence],
}

impl<G: FullfledgedGraph> State<G> {
    pub fn from_graph(graph: G) -> Self {
        let n = graph.number_of_nodes();
        State {
            graph,
            sequences: Default::default(),

            cache: Default::default(),

            first_pending: 0,
            tww: n,
            urng: Pcg64Mcg::from_entropy(),
        }
    }

    pub fn load_initial_sequences(&mut self, paths: &[PathBuf]) {
        let number_of_nodes = self.graph.number_of_nodes();

        let mut sequences: Vec<ContractionSequence> = paths
            .iter()
            .map(|path| {
                info!("Attempt to load cs {:?}", path);
                let reader = File::open(path).expect("Cannot open file");
                let buf_reader = BufReader::new(reader);
                let mut cs = ContractionSequence::pace_reader(buf_reader, number_of_nodes)
                    .expect("Cannot read contraction sequence");
                cs.normalize();
                cs
            })
            .collect();

        self.tww = sequences
            .iter()
            .map(|cs| {
                cs.compute_twin_width(self.graph.clone())
                    .expect("Input CS does not contract graph")
            })
            .min()
            .unwrap();

        while let Some(cs) = sequences.pop() {
            let digest = cs.binary_digest();
            if !self.cache.insert(digest) {
                // duplicate!
                continue;
            }

            let tww = cs.compute_twin_width(self.graph.clone()).unwrap();
            if tww > self.tww {
                continue;
            }
            self.sequences.push(cs);
        }

        info!(
            "Found {} unique contraction sequences with tww {}",
            self.sequences.len(),
            self.tww
        );
    }

    pub fn process_pending(&mut self) {
        if self.first_pending >= self.sequences.len() {
            return;
        }

        {
            // select a random non processed sequence
            let partner = self
                .urng
                .gen_range(self.first_pending..self.sequences.len());
            self.sequences.swap(self.first_pending, partner);
        }

        let end = self.graph.number_of_nodes().saturating_sub(1) as usize;
        for i in 0..end {
            for j in i + 1..end {
                let mut candidate = self.sequences[self.first_pending].clone();
                candidate.swap_merges(i, j);
                candidate.normalize();

                let digest = candidate.binary_digest();

                if !self.cache.insert(digest) {
                    debug!(" Skip swap with {i} <-> {j} due to cache hit");
                    // sequence was already considered; let's not do it again
                    continue;
                }

                let tww = candidate
                    .compute_twin_width_upto(self.graph.clone(), self.tww + 1)
                    .unwrap();

                if tww > self.tww {
                    debug!(" Skip swap with {i} <-> {j} due to TWW excess");
                    continue;
                }

                debug!("Found new optimal sequence");
                self.sequences.push(candidate);
            }
        }

        self.first_pending += 1;
    }
}

fn main() {
    build_pace_logger_for_level(LevelFilter::Info);

    let opt = Opt::from_args();
    if opt.sequences.is_empty() {
        panic!("Need at least one contraction sequence");
    }

    let graph = AdjArray::try_read_pace_file(opt.graph).unwrap();

    let mut state = State::from_graph(graph);
    if let Some(seed) = opt.random_seed {
        state.urng = Pcg64Mcg::seed_from_u64(seed);
    }

    state.load_initial_sequences(&opt.sequences);
    if state.sequences.is_empty() {
        panic!("No valid contraction sequences found");
    }

    let start = Instant::now();

    while state.first_pending < state.sequences.len() {
        state.process_pending();

        if state.first_pending % 1000 == 0 {
            info!(
                "Processed {:>5} rounds with {:>6} sequences found ({:>3.3} % of known) | cached: {:>8}",
                state.first_pending,
                state.sequences.len(),
                100.0 * state.first_pending as f64 / state.sequences.len() as f64,
                state.cache.len(),
            );

            if let Some(timeout) = opt.timeout_ms {
                if start.elapsed().as_millis() > timeout as u128 {
                    info!("Timeout. Stop computation");
                    break;
                }
            }
        }

        if state.sequences.len() >= opt.max_sequences.unwrap_or(usize::MAX) {
            break;
        }
    }

    let elapsed = start.elapsed();

    state.sequences.sort_by(|a, b| a.merges().cmp(b.merges()));

    let data = Export {
        seconds_shuffled: elapsed.as_secs_f64(),
        seqs_found: state.sequences.len(),
        seqs_processed: state.first_pending,
        seqs_cached: state.cache.len(),
        solutions: &state.sequences,
    };

    let writer = BufWriter::new(File::create(opt.output).expect("Could not create output file"));
    serde_json::to_writer(writer, &data).expect("Cannot serialize output");
}
