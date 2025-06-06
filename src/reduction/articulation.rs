use std::{marker::PhantomData, time::Duration};

use itertools::Itertools;
#[allow(unused_imports)]
use log::{debug, info};

use crate::{
    exact::highs_advanced::{HighsDominatingSetSolver, SolverResult},
    graph::*,
};

use super::*;

const SOLVER_TIMEOUT: Duration = Duration::from_secs(1);
const MAX_CC_SIZE: Node = 200;

pub struct RuleArticulationPoint<G> {
    _graph: PhantomData<G>,
}

impl<Graph: AdjacencyList + Clone + AdjacencyTest + 'static> RuleArticulationPoint<Graph> {
    pub fn new(_n: NumNodes) -> Self {
        Self {
            _graph: Default::default(),
        }
    }
}

impl<Graph: AdjacencyList + Clone + AdjacencyTest + 'static> ReductionRule<Graph>
    for RuleArticulationPoint<Graph>
{
    const NAME: &str = "RuleArticulationPoint";

    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        if redundant.are_all_unset() {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        let mut aps = graph.compute_articulation_points();
        if aps.are_all_unset() {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        {
            let mut aps_list = aps
                .iter_set_bits()
                .map(|u| (graph.number_of_nodes() - graph.degree_of(u), u))
                .collect_vec();

            aps_list.sort_unstable();

            for (_, u) in aps_list {
                if graph.neighbors_of(u).any(|v| aps.get_bit(v)) {
                    aps.clear_bit(u);
                }
            }
        }

        let mut solver = HighsDominatingSetSolver::new(graph.number_of_nodes());

        debug_assert!(solution.iter().all(|u| covered.get_bit(u)));

        let mut changed = false;
        for u in aps.iter_set_bits() {
            if redundant.get_bit(u) {
                if Self::process_small_ccs_at_red_ap(
                    graph,
                    solution,
                    covered,
                    redundant,
                    &mut solver,
                    u,
                ) {
                    changed = true;
                    continue;
                }
            } else if Self::process_small_ccs_at_ap(
                graph,
                solution,
                covered,
                redundant,
                &mut solver,
                u,
            ) {
                changed = true;
                continue;
            }
        }

        debug_assert!(solution.iter().all(|u| covered.get_bit(u)));

        (changed, None::<Box<dyn Postprocessor<Graph>>>)
    }
}

impl<Graph: AdjacencyList + Clone + AdjacencyTest + 'static> RuleArticulationPoint<Graph> {
    #[allow(unreachable_code, unused)]
    fn process_small_ccs_at_ap(
        graph: &mut Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &BitSet,
        solver: &mut HighsDominatingSetSolver,
        art_point: u32,
    ) -> bool {
        let mut restore_u_covered_to = covered.set_bit(art_point);

        let mut changed = false;
        for v in graph.neighbors_of(art_point) {
            let mut search = DFS::new(graph, v);
            search.exclude_node(art_point);

            let mut cc = search.take(1 + MAX_CC_SIZE as usize).collect_vec();
            if cc.len() > MAX_CC_SIZE as usize {
                continue;
            }

            cc.push(art_point);

            let eps = 0.3 / cc.len() as f64;

            let problem =
                solver.build_problem_of_subgraph(graph, covered, redundant, cc.as_slice(), |w| {
                    if w == art_point {
                        1.0 - eps - eps
                    } else if graph.has_edge(w, art_point) {
                        1.0 - eps
                    } else {
                        1.0
                    }
                });

            if let SolverResult::Optimal(solved) = problem.solve_exact(Some(SOLVER_TIMEOUT)) {
                changed |= !solved.is_empty();
                solution.add_nodes(solved.iter().cloned());
                covered.set_bits(cc.into_iter());

                if solution.is_in_domset(art_point) {
                    covered.set_bits(graph.neighbors_of(art_point));
                    restore_u_covered_to = true;
                } else {
                    restore_u_covered_to |= graph
                        .neighbors_of(art_point)
                        .any(|v: u32| solution.is_in_domset(v));
                }

                break;
            }
        }

        if !restore_u_covered_to {
            covered.clear_bit(art_point);
        }

        changed
    }

    fn process_small_ccs_at_red_ap(
        graph: &mut Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &BitSet,
        solver: &mut HighsDominatingSetSolver,
        art_point: u32,
    ) -> bool {
        // we only support uncovered redundant (those nodes are dealt with by the reducer)
        if covered.set_bit(art_point) {
            return false;
        }

        let mut restore_u_covered_to = false;
        let mut changed = false;
        for v in graph.neighbors_of(art_point) {
            let mut search = DFS::new(graph, v);
            search.exclude_node(art_point);

            let mut cc = search.take(1 + MAX_CC_SIZE as usize).collect_vec();
            if cc.len() > MAX_CC_SIZE as usize {
                continue;
            }

            cc.push(art_point);

            let eps = 0.3 / graph.degree_of(art_point) as f64;
            let problem =
                solver.build_problem_of_subgraph(graph, covered, redundant, cc.as_slice(), |w| {
                    if graph.has_edge(w, art_point) {
                        1.0 - eps
                    } else {
                        1.0
                    }
                });

            if let SolverResult::Optimal(solved) = problem.solve_exact(Some(SOLVER_TIMEOUT))
                && !solved.is_empty()
            {
                debug!("Solved CC at {art_point} with nodes {cc:?}");

                changed = true;
                solution.add_nodes(solved.iter().cloned());

                for &w in &solved {
                    covered.set_bits(graph.closed_neighbors_of(w));
                }
                debug_assert!(cc.iter().all(|&w| covered.get_bit(w)));
                debug_assert!(!solution.is_in_domset(art_point));

                restore_u_covered_to |= graph
                    .neighbors_of(art_point)
                    .any(|v: u32| solution.is_in_domset(v));

                break;
            }
        }

        if !restore_u_covered_to {
            covered.clear_bit(art_point);
        }

        changed
    }
}
