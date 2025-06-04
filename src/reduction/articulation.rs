use std::{marker::PhantomData, time::Duration};

use itertools::Itertools;
#[allow(unused_imports)]
use log::info;

use crate::{graph::*, reduction::small_exact::small_subgraph_exact};

use super::*;

const SOLVER_TIMEOUT: Duration = Duration::from_secs(1);

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

        //if false {
        //    // THIS IS A HEURISTIC!
        //    let mut changed = false;
        //    for u in aps.iter_set_bits() {
        //        let deg = graph.degree_of(u);
        //        if deg > 150 && !solution.is_in_domset(u) {
        //            changed = true;
        //            info!("Heuristically fix articulation point {u} of degree {deg}");
        //            solution.add_node(u);
        //            covered.set_bits(graph.closed_neighbors_of(u));
        //        }
        //    }
        //}

        aps &= redundant;
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

        const MAX_CC_SIZE: Node = 200;

        let mut changed = false;
        for u in aps.iter_set_bits() {
            if covered.get_bit(u) {
                continue;
            }

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
                    solver_buffer.as_mut_slice(),
                    SOLVER_TIMEOUT,
                )
                // && precious.into_iter().any(|w| solved.iter().contains(&w))
                {
                    solution.add_nodes(solved.iter().cloned());
                    covered.set_bits(cc.into_iter());

                    if !precious.into_iter().any(|w| solved.iter().contains(&w)) {
                        covered.clear_bit(u);
                    }

                    changed = true;
                    break;
                }
            }
        }

        (changed, None::<Box<dyn Postprocessor<Graph>>>)
    }
}
