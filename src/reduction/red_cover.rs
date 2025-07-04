use itertools::Itertools;
use num::Integer;
use stream_bitset::prelude::{BitmaskStreamConsumer as _, ToBitmaskStream};

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
        _domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        if never_select.cardinality() == 0 || covered.cardinality() == 0 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        debug_assert_eq!(self.alt_node.cardinality(), 0);
        for red in (never_select.bitmask_stream() - &*covered).iter_set_bits() {
            if let Some((a, b)) = graph.neighbors_of(red).collect_tuple()
                && graph.has_edge(a, b)
            {
                self.alt_node.set_bit(a);
                self.alt_node.set_bit(b);
            }
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

        self.alt_node.clear_all();

        (num_removed > 0, None::<Box<dyn Postprocessor<Graph>>>)
    }
}
