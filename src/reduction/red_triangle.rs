use itertools::Itertools;

use crate::graph::*;

use super::*;

pub struct RuleRedTriangle {
    buffer: Vec<Edge>,
}

impl RuleRedTriangle {
    pub fn new(_n: NumNodes) -> Self {
        Self {
            buffer: Vec::with_capacity(8),
        }
    }
}

impl<Graph: AdjacencyList + AdjacencyTest + GraphEdgeEditing + 'static> ReductionRule<Graph>
    for RuleRedTriangle
{
    const NAME: &str = "RedTriangle";

    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        _domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        if never_select.cardinality() == 0 || covered.cardinality() == 0 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        let mut changed = false;

        for u in never_select.iter_set_bits() {
            if let Some((a, b)) = graph.neighbors_of(u).collect_tuple()
                && graph.has_edge(a, b)
            {
                self.buffer.extend([a, b].into_iter().flat_map(|x| {
                    graph
                        .neighbors_of(x)
                        .filter(|&w| w != a && w != b && covered.get_bit(w))
                        .map(move |w| Edge(x, w))
                }));

                changed |= !self.buffer.is_empty();
                graph.remove_edges(self.buffer.drain(..));
            }
        }

        (changed, None::<Box<dyn Postprocessor<Graph>>>)
    }
}
