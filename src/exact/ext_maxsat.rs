use std::{
    io::Read,
    path::Path,
    process::{Command, Stdio},
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

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    use crate::{
        exact::{ext_maxsat::solve, naive::naive_solver},
        graph::{AdjacencyList as _, CsrGraph, GraphFromReader, GraphNodeOrder, NumNodes},
        testing::generate_random_graph_stream,
    };

    #[test]
    fn cross_with_naive_eval_maxsat() {
        const NODES: NumNodes = 20;
        let solver_binary: PathBuf = "./EvalMaxSAT_bin".into();
        if !solver_binary.is_file() {
            return;
        }

        let mut rng = Pcg64Mcg::seed_from_u64(0x123612873);

        for (graph, covered, never_select) in
            generate_random_graph_stream(&mut rng, NODES).take(100)
        {
            let naive = naive_solver(&graph, &covered, &never_select, None, None).unwrap();

            let csr_graph = CsrGraph::from_edges(graph.number_of_nodes(), graph.edges(true));

            let result = solve(
                &solver_binary,
                vec!["--TCT".into(), "1".into()],
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
    fn cross_with_naive_eval_uwrmaxsat() {
        const NODES: NumNodes = 20;
        let solver_binary: PathBuf = "./uwrmaxsat".into();
        if !solver_binary.is_file() {
            return;
        }

        let mut rng = Pcg64Mcg::seed_from_u64(0x12362873);

        for (graph, covered, never_select) in
            generate_random_graph_stream(&mut rng, NODES).take(100)
        {
            let naive = naive_solver(&graph, &covered, &never_select, None, None).unwrap();

            let csr_graph = CsrGraph::from_edges(graph.number_of_nodes(), graph.edges(true));

            let result = solve(
                &solver_binary,
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
                ],
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
}
