#![allow(rustdoc::private_intra_doc_links)]

use crate::prelude::*;
use itertools::Itertools;
use log::info;
use std::fmt::Debug;

mod frame;
use frame::*;

mod kernelization;
use kernelization::Kernelization;

#[derive(Clone, Debug)]
pub struct TwwSolution {
    tww: NumNodes,
    sequence: ContractionSequence,
}

pub type OptionalTwwSolution = Option<TwwSolution>;

pub struct BranchAndBound<G: FullfledgedGraph> {
    stack: Vec<Frame<G>>,
    paranoia_stack: Vec<G>,
    from_child: Option<OptionalTwwSolution>,
    solution: Option<OptionalTwwSolution>,

    number_of_nodes: Node,
    iterations: usize,

    paranoid: bool,
    super_verbose: bool,
    drop_output: bool,
    cross_validation: bool,
}

enum BBResult<G> {
    Result(OptionalTwwSolution),
    Branch(Frame<G>),
}

pub type Solution = Vec<Node>;
pub type OptSolution = Option<Solution>;

impl<G: FullfledgedGraph> BranchAndBound<G> {
    pub fn new(graph: G) -> Self {
        let mut stack: Vec<Frame<G>> = Vec::with_capacity(3 * (graph.len() + 2));
        let number_of_nodes = graph.number_of_nodes();
        stack.push(Frame::new(graph, 0, number_of_nodes));

        Self {
            stack,
            paranoia_stack: Vec::new(),
            solution: None,
            from_child: None,
            number_of_nodes,
            iterations: 0,
            paranoid: false,
            super_verbose: false,
            drop_output: false,
            cross_validation: false,
        }
    }

    pub fn with_paranoia(graph: G) -> Self {
        let mut graph_stack = Vec::with_capacity(3 * (graph.len() + 2));
        graph_stack.push(graph.clone());
        let mut res = Self::new(graph);
        res.paranoid = true;
        res.paranoia_stack = graph_stack;
        res
    }

    /// Sets an inclusive lower bound on the TWW. This is a hint which may or may
    /// not be used to prune the search tree. It is undefined behaviour if `lower_bound` exceeds
    /// the minimum TWW.
    ///
    /// # Warning
    /// This method may only be called before the first execution of the algorithm.
    pub fn set_lower_bound(&mut self, lower_bound: Node) {
        assert_eq!(self.iterations, 0);
        self.stack.last_mut().unwrap().slack = lower_bound;
        self.stack.last_mut().unwrap().initial_slack = lower_bound;
    }

    /// Sets an inclusive upper bound on the TWW. This is a hint which may or may
    /// not be used to prune the search tree. If `upper_bound` is smaller than the size of the
    /// minimum DFVS no solution can be produced.
    ///
    /// # Warning
    /// This method may only be called before the first execution of the algorithm.
    pub fn set_upper_bound(&mut self, upper_bound: Node) {
        assert_eq!(self.iterations, 0);
        assert!(upper_bound < self.number_of_nodes);

        self.stack.last_mut().unwrap().not_above = upper_bound;
        self.stack.last_mut().unwrap().initial_not_above = upper_bound;
    }

    /// Returns the number of recursive calls (i.e. calls direct or indirect calls to
    /// [`BranchAndBound::execute_step`]) processed so far
    pub fn number_of_iterations(&self) -> usize {
        self.iterations
    }

    /// Prints a backtrace if the solver instance is dropped without completing.
    /// This is useful if an assertion fired during the computation. However, be aware,
    /// that if you're using the solver with a time limit, this functionality might also
    /// fire if the solver did not finish within the time budget.
    pub fn set_drop_output(&mut self, enabled: bool) {
        self.drop_output = enabled;
    }

    /// Prints a call stack for each call into frame. Requires paranoia.
    ///
    /// # Warning
    /// This might produces millions lines of output.
    pub fn set_print_call_stack(&mut self, enabled: bool) {
        assert!(self.paranoid || !enabled);
        self.super_verbose = enabled
    }

    /// Uses the matrix solver to cross check all results obtained by the solver for small
    /// graphs. Requires paranoia.
    ///
    /// # Warning
    /// This might be extremely slow! Never use this feature in production.
    pub fn set_cross_validation(&mut self, enabled: bool) {
        assert!(self.paranoid || !enabled);
        self.cross_validation = enabled;
    }
}

impl<G: FullfledgedGraph> TerminatingIterativeAlgorithm<OptionalTwwSolution> for BranchAndBound<G> {}

impl<G: FullfledgedGraph> IterativeAlgorithm<OptionalTwwSolution> for BranchAndBound<G> {
    fn execute_step(&mut self) {
        assert!(self.solution.is_none());

        self.iterations += 1;

        // we execute the last frame on the stack but do not remove it yet, as it will remain there
        // in case it branches
        let result = {
            let current = self.stack.last_mut().unwrap();
            if let Some(from_child) = self.from_child.replace(None) {
                current.resume(from_child)
            } else {
                current.initialize()
            }
        };

        match result {
            BBResult::Result(res) => {
                let frame = self.stack.last().unwrap();

                if self.super_verbose {
                    info!(
                        "{} lb: {:>4}  ub: {:>4} sol: {:>4}",
                        (0..self.stack.len()).into_iter().map(|_| ' ').join(""),
                        frame.slack,
                        frame.not_above,
                        res.as_ref().map_or(-1, |r| r.tww as i64)
                    );
                }

                // make sure that the solution satisfies the required lower/upper bounds;
                // this check is sufficiently cheap that we do it even in the non-paranoid mode
                if let Some(solution) = res.as_ref() {
                    assert!(solution.tww >= frame.initial_slack);
                    assert!(solution.tww <= frame.initial_not_above);
                }

                // frame completed, so it's finally time to remove it from stack
                self.stack.pop();

                if self.stack.is_empty() {
                    self.solution = Some(res);
                    return;
                }

                self.from_child = Some(res);
            }

            BBResult::Branch(frame) => {
                if self.paranoid {
                    self.paranoia_stack.push(frame.graph.clone());
                }

                self.stack.push(frame);
                self.from_child = None;
            }
        }
    }

    fn is_completed(&self) -> bool {
        self.stack.is_empty()
    }

    fn best_known_solution(&mut self) -> Option<OptionalTwwSolution> {
        self.solution.clone()
    }
}

impl<G: FullfledgedGraph> Drop for BranchAndBound<G> {
    fn drop(&mut self) {
        while let Some(frame) = self.stack.pop() {
            if self.drop_output {
                info!(
                    "{} {} (dbg-lower={}, dbg-upper={})",
                    (0..self.stack.len()).into_iter().map(|_| ' ').join(""),
                    frame
                        .resume_with
                        .map_or(String::from("UNINITIALIZED"), |x| x.describe()),
                    frame.initial_slack,
                    frame.initial_not_above,
                );
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{log::build_pace_logger_for_level, testing::get_test_graphs_with_tww};

    use super::*;
    use std::{fs::File, io::BufReader};

    #[test]
    fn tiny() {
        build_pace_logger_for_level(log::LevelFilter::Info);

        for (i, tww) in [1, 2, 0, 0, 3, 0, 2, 4, 1, 2].into_iter().enumerate() {
            if i == 4 {
                continue;
            }

            let filename = format!("instances/tiny/tiny{:>03}.gr", i + 1);
            let reader = File::open(filename.clone())
                .unwrap_or_else(|_| panic!("Cannot open file {}", &filename));
            let buf_reader = BufReader::new(reader);

            let pace_reader =
                PaceReader::try_new(buf_reader).expect("Could not construct PaceReader");

            let mut graph = AdjArray::new(pace_reader.number_of_nodes());
            graph.add_edges(pace_reader, EdgeColor::Black);

            let mut algo = BranchAndBound::with_paranoia(graph.clone());
            //algo.super_verbose = true;

            let TwwSolution {
                tww: size,
                sequence: mut seq,
            } = algo.run_to_completion().unwrap().unwrap();

            seq.add_unmerged_singletons(&graph);
            let tww_of_seq = seq.compute_twin_width(graph).unwrap();

            assert_eq!(tww_of_seq, tww, "file: {filename}");
            assert_eq!(size, tww, "file: {filename}");
        }
    }

    #[test]
    fn small_random() {
        let mut total_iters = 0;
        for (filename, graph, presolved_tww) in
            get_test_graphs_with_tww("instances/small-random/*.gr").step_by(3)
        {
            if graph.number_of_nodes() > 15 {
                continue;
            }
            println!(" Test {filename}");

            let mut algo = BranchAndBound::new(graph.clone());
            let TwwSolution { tww, mut sequence } = algo.run_to_completion().unwrap().unwrap();
            total_iters += algo.number_of_iterations();

            sequence.add_unmerged_singletons(&graph);
            let tww_of_seq = sequence.compute_twin_width(graph).unwrap();
            assert_eq!(tww_of_seq, tww, "file: {filename}");
            assert_eq!(tww, presolved_tww, "file: {filename}");
        }
        println!("Num Iters: {total_iters}");
    }
}

// with guards: 32'719'702
// without:     37'865'316