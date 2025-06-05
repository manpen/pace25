use crate::graph::AdjacencyList;

use super::*;

pub struct RuleIsolatedReduction;

impl<Graph: AdjacencyList + 'static> ReductionRule<Graph> for RuleIsolatedReduction {
    const NAME: &str = "RuleIsolated";

    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        if redundant.cardinality() == 0 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        let mut changed = false;

        for u in graph.vertices() {
            if solution.is_in_domset(u) || redundant.get_bit(u) || covered.get_bit(u) {
                continue;
            }

            if graph.neighbors_of(u).all(|v| redundant.get_bit(v)) {
                solution.add_node(u);
                covered.set_bits(graph.closed_neighbors_of(u));
                changed = true;
            }
        }

        (changed, None::<Box<dyn Postprocessor<Graph>>>)
    }
}
