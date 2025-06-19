use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread::sleep,
    time::{Duration, Instant},
};

use crate::{
    exact::{
        highs_advanced::{HighsDominatingSetSolver, SolverResult, unit_weight},
        search_binary_path,
    },
    graph::*,
    utils::signal_handling,
};
use itertools::Itertools;
use log::info;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighsSubprocessProblem {
    pub timeout: u64,
    graph: CsrGraph,
    covered: Vec<Node>,
    never_select: Vec<Node>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighsSubprocessResponse {
    solution: Option<(Vec<Node>, bool)>,
}

impl HighsSubprocessProblem {
    pub fn solve(mut self) -> HighsSubprocessResponse {
        let mut solver = HighsDominatingSetSolver::new(self.graph.number_of_nodes());

        let covered = BitSet::new_with_bits_set(
            self.graph.number_of_nodes(),
            std::mem::take(&mut self.covered),
        );

        let redundant = BitSet::new_with_bits_set(
            self.graph.number_of_nodes(),
            std::mem::take(&mut self.never_select),
        );

        let problem = solver.build_problem(&self.graph, &covered, &redundant, unit_weight);

        HighsSubprocessResponse {
            solution: match problem.solve_allow_subopt(Some(Duration::from_secs(self.timeout))) {
                super::highs_advanced::SolverResult::Optimal(items) => Some((items, true)),
                super::highs_advanced::SolverResult::Suboptimal(items) => Some((items, false)),
                super::highs_advanced::SolverResult::Timeout => None,
                super::highs_advanced::SolverResult::Infeasible => None,
            },
        }
    }
}

pub fn solve_with_subprocess(
    binary: &Path,
    graph: &CsrGraph,
    covered: &BitSet,
    never_select: &BitSet,
    timeout: Duration,
    grace: Duration,
) -> anyhow::Result<SolverResult> {
    let mut child = Command::new(binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    let start = Instant::now();

    // feed data
    {
        let stdin = child.stdin.take().expect("Failed to take STDIN");
        let problem = HighsSubprocessProblem {
            timeout: timeout.as_secs_f64().ceil() as u64,
            graph: graph.clone(),
            covered: covered.iter_set_bits().collect_vec(),
            never_select: never_select.iter_set_bits().collect_vec(),
        };
        serde_json::to_writer(stdin, &problem)?
    }

    let timeout_with_grace = timeout + grace;
    let mut stdout = child.stdout.take().expect("Failed to take STDOUT");
    let mut received = Vec::new();
    loop {
        let _ = stdout.read(&mut received);
        match child.try_wait()? {
            Some(_status) => {
                let _ = stdout.read_to_end(&mut received);

                let response: HighsSubprocessResponse =
                    serde_json::from_slice(received.as_slice())?;

                return Ok(match response.solution {
                    Some((items, true)) => SolverResult::Optimal(items),
                    Some((items, false)) => SolverResult::Suboptimal(items),
                    None => SolverResult::Timeout,
                });
            }
            None => {
                if start.elapsed() > timeout_with_grace {
                    info!("Kill subprocess");
                    child.kill()?;
                    return Ok(SolverResult::Timeout);
                }

                if signal_handling::received_ctrl_c() {
                    info!("Kill subprocess due to received ctrl_c");
                    child.kill()?;
                    anyhow::bail!("Timeout / Signal");
                }

                sleep(Duration::from_millis(500));
            }
        }
    }
}

const HIGHS_CHILD: &str = "highs_child";

pub fn solve_with_subprocess_find_binary(
    graph: &CsrGraph,
    covered: &BitSet,
    never_select: &BitSet,
    timeout: Duration,
    grace: Duration,
) -> anyhow::Result<SolverResult> {
    let path = search_binary_path(&PathBuf::from(HIGHS_CHILD))?;
    info!("Start subprocess using {path:?}");
    solve_with_subprocess(&path, graph, covered, never_select, timeout, grace)
}
