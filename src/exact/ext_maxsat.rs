use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Child, ChildStdout, Command, Stdio},
    thread::sleep,
    time::{Duration, Instant},
};

use crate::prelude::*;
use itertools::Itertools as _;
use stream_bitset::prelude::*;
use tempfile::NamedTempFile;

fn compute_skip_constraints<G>(graph: &G, covered: &BitSet, redundant: &BitSet) -> BitSet
where
    G: AdjacencyList + AdjacencyTest,
{
    let mut skip_constraints_of = covered.clone();

    for u in (redundant.bitmask_stream() - covered).iter_set_bits() {
        if let Some((a, b)) = graph
            .neighbors_of(u)
            .filter(|&v| v != u && !redundant.get_bit(v))
            .collect_tuple()
            && graph.has_edge(a, b)
        {
            skip_constraints_of.set_bit(a);
            skip_constraints_of.set_bit(b);
        }
    }

    skip_constraints_of
}

fn write_to_input_maxsat(
    writer: &mut impl std::io::Write,
    graph: &(impl StaticGraph + SelfLoop),
    no_constraints: &BitSet,
    never_select: &BitSet,
) -> std::io::Result<()> {
    let mut mapping = vec![0; graph.len()];
    let mut num_mapped_nodes: Node = 1;
    for u in never_select.iter_cleared_bits() {
        mapping[u as usize] = num_mapped_nodes;
        num_mapped_nodes += 1;
    }

    // write a hard constraint for each uncovered node
    for u in graph.vertices() {
        if no_constraints.get_bit(u) {
            continue;
        }

        write!(writer, "h ")?;
        for v in graph.neighbors_of(u) {
            if never_select.get_bit(v) {
                continue;
            }

            write!(writer, "{} ", mapping[v as usize])?;
        }
        writeln!(writer, "0")?;
    }

    // write a soft constraint for each non-redundant node
    for u in 1..num_mapped_nodes {
        writeln!(writer, "1 -{u} 0")?; // "1" is weight, - indicated negation, 0 is end of clause
    }

    Ok(())
}

fn read_solver_response(
    stdout_string: String,
    graph: &(impl StaticGraph + SelfLoop),
    _covered: &BitSet,
    never_select: &BitSet,
) -> anyhow::Result<DominatingSet> {
    let mut domset = DominatingSet::new(graph.number_of_nodes());

    for line in stdout_string.lines() {
        let line: &str = line;
        let mut parts = line.split(' ').filter(|t| !t.is_empty());

        match parts.next() {
            Some("s") => {
                if !line.contains("OPTIMUM FOUND") {
                    anyhow::bail!("suboptimal solution");
                }
            }

            Some("o") => {}

            Some("v") => {
                let values_in_ascii: &str = parts.next().expect("Variable assignment");
                assert_eq!(
                    values_in_ascii.len(),
                    (graph.number_of_nodes() - never_select.cardinality()) as usize
                );

                domset.add_nodes(
                    never_select
                        .iter_cleared_bits()
                        .zip(values_in_ascii.chars())
                        .filter_map(|(i, c)| (c == '1').then_some(i as Node)),
                );
            }

            _ => {}
        }
    }

    Ok(domset)
}

pub fn solve(
    solver_binary: &Path,
    mut args: Vec<String>,
    graph: &(impl StaticGraph + SelfLoop),
    covered: &BitSet,
    never_select: &BitSet,
    timeout: Option<Duration>,
) -> anyhow::Result<DominatingSet> {
    let start = Instant::now();
    let no_constraints = compute_skip_constraints(graph, covered, never_select);

    let mut maxsat_file = NamedTempFile::new()?;

    {
        use std::io::Write;
        let mut writer = &mut maxsat_file;
        write_to_input_maxsat(&mut writer, graph, &no_constraints, never_select)?;
        writer.flush()?;
    }

    args.push(maxsat_file.path().to_str().expect("Path").into());

    let mut child = Command::new(solver_binary)
        .args(args)
        .stdout(Stdio::piped())
        .spawn()?;

    let mut stdout = child.stdout.take().expect("Failed to take STDOUT");
    let mut received = Vec::new();
    loop {
        let _ = stdout.read(&mut received);
        let elapsed = start.elapsed();
        match child.try_wait()? {
            Some(_status) => {
                let _ = stdout.read_to_end(&mut received);
                break;
            }
            None => {
                if let Some(timeout) = timeout
                    && elapsed > timeout
                {
                    info!("Kill subprocess");
                    child.kill()?;
                    anyhow::bail!("Timeout");
                }

                if signal_handling::received_ctrl_c() {
                    info!("Kill subprocess due to received ctrl_c");
                    child.kill()?;
                    anyhow::bail!("Timeout / Signal");
                }

                if elapsed.as_millis() < 1000 {
                    sleep(Duration::from_millis(5));
                } else {
                    sleep(Duration::from_millis(500));
                }
            }
        }
    }

    let stdout_string = String::from_utf8(received)?;
    read_solver_response(stdout_string, graph, covered, never_select)
}

struct ChildContext {
    child: Child,
    stdout: ChildStdout,
    received: Vec<u8>,
    timeout: Option<Duration>,
    done: bool,
}

pub fn solve_multiple(
    solvers: Vec<(PathBuf, Vec<String>, Option<Duration>)>,
    graph: &(impl StaticGraph + SelfLoop),
    covered: &BitSet,
    never_select: &BitSet,
) -> anyhow::Result<DominatingSet> {
    let start = Instant::now();
    let no_constraints = compute_skip_constraints(graph, covered, never_select);

    let mut maxsat_file = NamedTempFile::new()?;

    {
        use std::io::Write;
        let mut writer = &mut maxsat_file;
        write_to_input_maxsat(&mut writer, graph, &no_constraints, never_select)?;
        writer.flush()?;
    }

    let mut children = solvers
        .into_iter()
        .filter_map(|(path, mut args, timeout)| {
            args.push(maxsat_file.path().to_str().expect("Path").into());
            if let Ok(mut child) = Command::new(path).args(args).stdout(Stdio::piped()).spawn() {
                let stdout = child.stdout.take().expect("Failed to take STDOUT");
                Some(ChildContext {
                    child,
                    stdout,
                    received: Vec::with_capacity(graph.number_of_nodes() as usize + 2000),
                    timeout,
                    done: false,
                })
            } else {
                None
            }
        })
        .collect_vec();

    if children.is_empty() {
        anyhow::bail!("Could not start any solvers");
    }

    let received_buffer: Vec<u8> = 'outer: loop {
        let elapsed = start.elapsed();
        for child in children.iter_mut() {
            if child.done {
                continue;
            }

            let _ = child.stdout.read(&mut child.received);
            match child.child.try_wait()? {
                Some(_status) => {
                    let _ = child.stdout.read_to_end(&mut child.received);
                    info!("Child completed");
                    child.done = true;
                    break 'outer std::mem::take(&mut child.received);
                }
                None => {
                    if let Some(timeout) = child.timeout
                        && elapsed > timeout
                    {
                        info!("Kill subprocess");
                        child.child.kill()?;

                        child.done = true;
                    }
                }
            }
        }
        children.retain(|c| !c.done);
        if children.is_empty() {
            anyhow::bail!("No solution obtained");
        }

        if elapsed.as_millis() < 1000 {
            sleep(Duration::from_millis(5));
        } else {
            sleep(Duration::from_millis(500));
        }
    };

    for child in children.iter_mut() {
        if !child.done {
            let _ = child.child.kill();
        }
    }

    let stdout_string = String::from_utf8(received_buffer)?;
    read_solver_response(stdout_string, graph, covered, never_select)
}

#[cfg(test)]
mod test {
    use super::*;
    use std::path::PathBuf;

    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    use crate::{
        exact::{
            ext_maxsat::{solve, solve_multiple},
            naive::naive_solver,
        },
        graph::{AdjacencyList as _, CsrGraph, GraphFromReader, GraphNodeOrder, NumNodes},
        testing::generate_random_graph_stream,
    };

    const EVAL_BIN: &str = "./EvalMaxSAT_bin";
    const UWR_BIN: &str = "./uwrmaxsat";

    fn eval_args() -> Vec<String> {
        vec!["--TCT".into(), "1".into()]
    }

    fn uwr_args() -> Vec<String> {
        vec![
            "-v0".into(),
            "-no-bin".into(),
            "-no-sat".into(),
            "-no-par".into(),
            "-maxpre-time=60".into(),
            "-scip-cpu=800".into(),
            "-scip-delay=400".into(),
            "-m".into(),
            "-bm".into(),
        ]
    }

    fn test_single(rng: &mut impl Rng, solver_binary: &Path, args: Vec<String>) {
        const NODES: NumNodes = 20;

        if !solver_binary.is_file() {
            return;
        }

        for (graph, covered, never_select) in generate_random_graph_stream(rng, NODES).take(100) {
            let naive = naive_solver(&graph, &covered, &never_select, None, None).unwrap();

            let csr_graph = CsrGraph::from_edges(graph.number_of_nodes(), graph.edges(true));

            let result = solve(
                &solver_binary,
                args.clone(),
                &csr_graph,
                &covered,
                &never_select,
                None,
            )
            .unwrap();

            assert!(result.is_valid_given_previous_cover(&graph, &covered));
            assert_eq!(naive.len(), result.len());
            assert!(result.iter().all(|u| !never_select.get_bit(u)));
        }
    }

    #[test]
    fn cross_with_naive_eval_maxsat() {
        let solver_binary: PathBuf = EVAL_BIN.into();
        let mut rng = Pcg64Mcg::seed_from_u64(0x123612873);
        test_single(&mut rng, &solver_binary, eval_args());
    }

    #[test]
    fn cross_with_naive_eval_uwrmaxsat() {
        let solver_binary: PathBuf = UWR_BIN.into();
        let mut rng = Pcg64Mcg::seed_from_u64(0x121612873);
        test_single(&mut rng, &solver_binary, uwr_args());
    }

    fn test_multi(rng: &mut impl Rng, mut solvers: Vec<(PathBuf, Vec<String>, Option<Duration>)>) {
        const NODES: NumNodes = 20;

        // delete solvers that could not be found
        solvers.retain(|s| s.0.is_file());
        if solvers.is_empty() {
            return;
        }

        for (graph, covered, never_select) in generate_random_graph_stream(rng, NODES).take(100) {
            let naive = naive_solver(&graph, &covered, &never_select, None, None).unwrap();

            let csr_graph = CsrGraph::from_edges(graph.number_of_nodes(), graph.edges(true));

            let result =
                solve_multiple(solvers.clone(), &csr_graph, &covered, &never_select).unwrap();

            assert!(result.is_valid_given_previous_cover(&graph, &covered));
            assert_eq!(naive.len(), result.len());
            assert!(result.iter().all(|u| !never_select.get_bit(u)));
        }
    }

    #[test]
    fn cross_with_naive_eval_maxsat_mult() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x1216112873);
        test_multi(&mut rng, vec![(PathBuf::from(EVAL_BIN), eval_args(), None)]);
    }

    #[test]
    fn cross_with_naive_eval_uwrmaxsat_mult() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x12161173);
        test_multi(&mut rng, vec![(PathBuf::from(UWR_BIN), uwr_args(), None)]);
    }

    #[test]
    fn cross_with_naive_both_mult() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x12112873);
        test_multi(
            &mut rng,
            vec![
                (PathBuf::from(EVAL_BIN), eval_args(), None),
                (PathBuf::from(UWR_BIN), uwr_args(), None),
            ],
        );
    }
}
