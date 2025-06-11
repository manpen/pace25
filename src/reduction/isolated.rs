use crate::graph::AdjacencyList;

use super::*;

pub struct RuleIsolatedReduction;

impl<Graph: AdjacencyList + 'static> ReductionRule<Graph> for RuleIsolatedReduction {
    const NAME: &str = "RuleIsolated";

    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        if never_select.cardinality() == 0 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        let mut changed = false;

        for u in graph.vertices() {
            if domset.is_in_domset(u) || never_select.get_bit(u) || covered.get_bit(u) {
                continue;
            }

            if graph.neighbors_of(u).all(|v| never_select.get_bit(v)) {
                domset.fix_node(u);
                covered.set_bits(graph.closed_neighbors_of(u));
                changed = true;
            }
        }

        (changed, None::<Box<dyn Postprocessor<Graph>>>)
    }
}
