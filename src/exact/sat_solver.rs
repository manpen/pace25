use std::process::{Command, Stdio};

use crate::prelude::*;
use tempfile::NamedTempFile;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum SolverBackend {
    GOODLP,
    MAXSAT,
}

pub fn solve(
    graph: &(impl StaticGraph + SelfLoop),
    covered: BitSet,
    partial_solution: Option<DominatingSet>,
    backend: SolverBackend,
) -> anyhow::Result<DominatingSet> {
    let mut domset =
        partial_solution.unwrap_or_else(|| DominatingSet::new(graph.number_of_nodes()));

    // TODO: compute redundant nodes
    let redundant = graph.vertex_bitset_unset();

    match backend {
        SolverBackend::GOODLP => {
            #[cfg(feature = "goodlp")]
            good_lp_solver(graph, &mut domset, covered)?;
            #[cfg(not(feature = "goodlp"))]
            panic!("goodlp not supported");
        }
        SolverBackend::MAXSAT => {
            maxsat_solver(graph, &mut domset, covered, redundant)?;
        }
    }

    Ok(domset)
}

fn maxsat_solver(
    graph: &(impl StaticGraph + SelfLoop),
    domset: &mut DominatingSet,
    covered: BitSet,
    redundant: BitSet,
) -> std::result::Result<(), anyhow::Error> {
    let mut maxsat_file = NamedTempFile::new()?;

    {
        use std::io::Write;
        // TODO: Add BufWriter!
        //let mut writer = BufWriter::new(maxsat_file);
        let mut writer = &mut maxsat_file;

        // write a hard constraint for each uncovered node
        for u in graph.vertices() {
            if covered.get_bit(u) {
                continue;
            }

            write!(&mut writer, "h ")?;
            for v in graph.neighbors_of(u) {
                if redundant.get_bit(v) {
                    continue;
                }

                write!(&mut writer, "{} ", v + 1)?;
            }
            writeln!(&mut writer, "0")?;
        }

        // write a soft constraint for each non-redundant node
        for u in graph.vertices() {
            //if redundant.get_bit(u) || domset.is_in_domset(u) {
            //    continue;
            //}
            writeln!(&mut writer, "1 -{} 0", u + 1)?; // "1" is weight, - indicated negation, 0 is end of clause
        }

        writer.flush()?;
    }

    let solution_path = maxsat_file.path().to_str().expect("Path");

    // start solver
    {
        let process_output = Command::new("./EvalMaxSAT_bin")
            .args(vec!["--TCT", "1800", solution_path])
            .stdout(Stdio::piped())
            .output()?;
        let stdout_string = String::from_utf8(process_output.stdout)?;

        let mut expected_size: Option<usize> = None;
        let mut bits_set: Option<usize> = None;

        for line in stdout_string.lines() {
            let line: &str = line;
            let mut parts = line.split(' ').filter(|t| !t.is_empty());

            match parts.next() {
                Some("s") => {
                    if !line.contains("OPTIMUM FOUND") {
                        anyhow::bail!("suboptimal solution");
                    }
                }

                Some("o") => {
                    expected_size = Some(parts.next().expect("Size info").parse()?);
                }

                Some("v") => {
                    assert!(bits_set.is_none());

                    let values_in_ascii: &str = parts.next().expect("Variable assignment");
                    assert_eq!(values_in_ascii.len(), graph.number_of_nodes() as usize);

                    let card_before = domset.len();
                    domset.add_nodes(
                        values_in_ascii
                            .chars()
                            .enumerate()
                            .filter_map(|(i, c)| (c == '1').then_some(i as Node)),
                    );

                    bits_set = Some(domset.len() - card_before);
                }

                _ => {}
            }
        }

        assert!(expected_size.is_some());
        assert_eq!(expected_size, bits_set);
    }

    Ok(())
}

#[cfg(feature = "goodlp")]
fn good_lp_solver(
    graph: &(impl StaticGraph + SelfLoop),
    domset: &mut DominatingSet,
    covered: stream_bitset::prelude::BitSetImpl<u32>,
) -> std::result::Result<(), anyhow::Error> {
    use good_lp::{Expression, ProblemVariables, Solution, SolverModel, default_solver, variable};
    use log::info;
    use std::time::Instant;

    let mut problem: ProblemVariables = ProblemVariables::new();
    let vec = problem.add_vector(variable().binary(), graph.number_of_nodes() as usize);
    let mut sum = Expression::from(vec[0]);
    for x in &vec[1..] {
        sum += x;
    }
    let mut model = problem.minimise(sum).using(default_solver);
    info!("Invoke MinSAT solver");
    info!(
        "Previously covered: {}/{}",
        covered.cardinality(),
        graph.number_of_nodes()
    );
    for u in graph.vertices() {
        if covered.get_bit(u) {
            continue;
        }

        let mut expr = Expression::with_capacity(graph.degree_of(u) as usize);
        for v in graph.neighbors_of(u) {
            expr += vec[v as usize];
        }

        model = model.with(expr.geq(1));
    }
    model.set_parameter("log", "0");
    let start_time = Instant::now();
    let solution = model.solve()?;
    info!("MinSat solver took {:?}", start_time.elapsed());
    let size_before = domset.len();
    domset.add_nodes(
        vec.into_iter()
            .enumerate()
            .filter(|(_, var)| solution.value(*var) > 0.9)
            .map(|(i, _)| i as Node),
    );
    info!("Added {} nodes into DS", domset.len() - size_before);
    Ok(())
}
