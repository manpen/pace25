use std::time::Instant;

use crate::{
    kernelization::{rule1::Rule1, KernelizationRule},
    prelude::*,
};
use log::info;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum SolverBackend {
    SCIP,
    GOODLP,
}

pub fn solve(
    graph: &(impl StaticGraph + SelfLoop),
    partial_solution: Option<DominatingSet>,
    backend: SolverBackend,
) -> anyhow::Result<DominatingSet> {
    let mut domset =
        partial_solution.unwrap_or_else(|| DominatingSet::new(graph.number_of_nodes()));

    let redundant = Rule1::apply_rule(graph, &mut domset);
    let covered = domset.compute_covered(graph);

    match backend {
        SolverBackend::SCIP => {
            use russcip::prelude::*;

            let mut model = Model::default().minimize();
            model = model.set_int_param("display/verblevel", 0).unwrap();

            let vars: Vec<_> = graph
                .vertices_range()
                .map(|u| (!redundant.get_bit(u)).then(|| model.add(var().bin().obj(1.0))))
                .collect();

            for u in graph.vertices() {
                if covered.get_bit(u) {
                    continue;
                }

                let mut expr = cons().ge(1.0);
                for v in graph.neighbors_of(u) {
                    if let Some(v) = vars[v as usize].as_ref() {
                        expr = expr.coef(v, 1.0);
                    }
                }

                model.add(expr);
            }

            let solved_model = model.solve();

            let sol = solved_model.best_sol().unwrap();

            domset.add_nodes(
                vars.into_iter()
                    .enumerate()
                    .filter(|(_, var)| var.as_ref().is_some_and(|x| sol.val(x) > 0.5))
                    .map(|(i, _)| i as Node),
            );
        }
        SolverBackend::GOODLP => {
            use good_lp::{
                default_solver, variable, Expression, ProblemVariables, Solution, SolverModel,
            };

            let mut problem: ProblemVariables = ProblemVariables::new();

            // TODO: if we use large partial_solutions, we can reduce the number of variables (only one for each uncovered node)
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
        }
    }

    Ok(domset)
}
