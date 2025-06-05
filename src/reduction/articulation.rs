use std::{marker::PhantomData, time::Duration};

use itertools::Itertools;
#[allow(unused_imports)]
use log::info;

use crate::{graph::*, reduction::small_exact::small_subgraph_exact};

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

        let mut solver_buffer = vec![Node::MAX; graph.len()];

        let mut changed = false;
        for u in aps.iter_set_bits() {
            if redundant.get_bit(u) {
                if covered.get_bit(u) {
                    continue;
                }

                if Self::process_small_ccs_at_red_ap(
                    graph,
                    solution,
                    covered,
                    redundant,
                    &mut solver_buffer,
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
                &mut solver_buffer,
                u,
            ) {
                changed = true;
                continue;
            }
        }

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
        solver_buffer: &mut [Node],
        u: u32,
    ) -> bool {
        // rule is currently unsound
        return false;
        let mut restore_u_covered_to = covered.set_bit(u);

        let precious = [u];

        let mut changed = false;
        for v in graph.neighbors_of(u) {
            let mut search = DFS::new(graph, v);
            search.exclude_node(u);

            let mut cc = search.take(1 + MAX_CC_SIZE as usize).collect_vec();
            if cc.len() > MAX_CC_SIZE as usize {
                continue;
            }

            cc.push(u);

            let neighbors = graph
                .neighbors_of(u)
                .filter(|v| cc.contains(v))
                .map(|v| (v, covered.get_bit(v)))
                .collect_vec();

            // mark neighbors with 1-eps, and ap with 1-2eps

            if let Some(solved) = small_subgraph_exact(
                graph,
                covered,
                redundant,
                &cc,
                &precious,
                solver_buffer,
                SOLVER_TIMEOUT,
            ) {
                solution.add_nodes(solved.iter().cloned());
                restore_u_covered_to |= solution.is_in_domset(u);

                covered.set_bits(cc.into_iter());
                changed = true;
                break;
            }
        }

        if !restore_u_covered_to {
            covered.clear_bit(u);
        }

        changed
    }

    fn process_small_ccs_at_red_ap(
        graph: &mut Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &BitSet,
        solver_buffer: &mut [Node],
        u: u32,
    ) -> bool {
        if covered.get_bit(u) {
            return false;
        }

        let mut changed = false;
        for v in graph.neighbors_of(u) {
            let mut search = DFS::new(graph, v);
            search.exclude_node(u);

            let mut cc = search.take(1 + MAX_CC_SIZE as usize).collect_vec();
            if cc.len() > MAX_CC_SIZE as usize {
                continue;
            }

            cc.push(u);
            covered.set_bit(u);

            let precious = cc
                .iter()
                .copied()
                .filter(|&w| graph.has_edge(w, u))
                .collect_vec();

            if let Some(solved) = small_subgraph_exact(
                graph,
                covered,
                redundant,
                &cc,
                precious.as_slice(),
                solver_buffer,
                SOLVER_TIMEOUT,
            ) {
                solution.add_nodes(solved.iter().cloned());
                covered.set_bits(cc.into_iter());

                if !precious.into_iter().any(|w| solved.iter().contains(&w)) {
                    covered.clear_bit(u);
                }
                changed = true;

                break;
            }
        }

        changed
    }
}
