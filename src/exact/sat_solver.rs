use std::time::Instant;

use crate::prelude::*;
use good_lp::{default_solver, variable, Expression, ProblemVariables, Solution, SolverModel};
use log::info;

pub fn solve(
    graph: &impl FullfledgedGraph,
    partial_solution: Option<DominatingSet>,
) -> anyhow::Result<DominatingSet> {
    let mut problem: ProblemVariables = ProblemVariables::new();

    // TODO: if we use large partial_solutions, we can reduce the number of variables (only one for each uncovered node)
    let vec = problem.add_vector(variable().binary(), graph.number_of_nodes() as usize);

    let mut sum = Expression::from(vec[0]);
    for x in &vec[1..] {
        sum += x;
    }

    let mut model = problem.minimise(sum).using(default_solver);

    let mut domset =
        partial_solution.unwrap_or_else(|| DominatingSet::new(graph.number_of_nodes()));
    let covered = domset.compute_covered(graph);

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

        let mut expr = Expression::from(vec[u as usize]);
        for v in graph.neighbors_of(u) {
            assert_ne!(u, v);
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

    Ok(domset)
}
