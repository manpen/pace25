use itertools::Itertools;
use num::Integer;

use crate::graph::*;

use super::*;

pub struct RuleRedundantCover {
    alt_node: BitSet,
}

impl RuleRedundantCover {
    pub fn new(n: NumNodes) -> Self {
        Self {
            alt_node: BitSet::new(n),
        }
    }
}

impl<
    Graph: std::fmt::Debug
        + AdjacencyList
        + AdjacencyTest
        + GraphEdgeEditing
        + UnsafeGraphEditing
        + GraphEdgeOrder
        + 'static,
> ReductionRule<Graph> for RuleRedundantCover
{
    const NAME: &str = "RedundantCover";

    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        _solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        if redundant.cardinality() == 0 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        self.alt_node.clear_all();
        for red in redundant.iter_set_bits() {
            if covered.get_bit(red) {
                continue;
            }

            let (a, b) = match graph.neighbors_of(red).collect_tuple() {
                Some(x) => x,
                None => continue,
            };

            if !graph.has_edge(a, b) {
                continue;
            }

            self.alt_node.set_bit(a);
            self.alt_node.set_bit(b);
        }

        if self.alt_node.cardinality() == 0 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        let mut num_removed = 0;
        for u in graph.vertices_range() {
            if self.alt_node.get_bit(u) {
                num_removed += unsafe {
                    graph.remove_half_edges_at_if(u, |v| {
                        !self.alt_node.get_bit(v) && covered.get_bit(v)
                    })
                };
            } else if covered.get_bit(u) {
                num_removed +=
                    unsafe { graph.remove_half_edges_at_if(u, |v| self.alt_node.get_bit(v)) };
            }
        }

        assert!(num_removed.is_even());
        unsafe {
            graph.set_number_of_edges(graph.number_of_edges() - num_removed / 2);
        }

        (num_removed > 0, None::<Box<dyn Postprocessor<Graph>>>)
    }
}
