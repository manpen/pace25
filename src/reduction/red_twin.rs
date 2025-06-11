use fxhash::{FxBuildHasher, FxHashSet};
use itertools::Itertools;

use crate::graph::*;

use super::*;

pub struct RuleRedTwin {
    buffer: FxHashSet<Edge>,
}

impl RuleRedTwin {
    pub fn new(_n: NumNodes) -> Self {
        Self {
            buffer: FxHashSet::with_capacity_and_hasher(1000, FxBuildHasher::default()),
        }
    }
}

impl<Graph: AdjacencyList + AdjacencyTest + GraphEdgeEditing + 'static> ReductionRule<Graph>
    for RuleRedTwin
{
    const NAME: &str = "RedTwin";

    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        _domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        if never_select.cardinality() == 0 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        let covered_before = covered.cardinality();
        for u in never_select.iter_set_bits() {
            if let Some((a, b)) = graph.neighbors_of(u).collect_tuple() {
                let norm = Edge(a, b).normalized();
                if !self.buffer.insert(norm) {
                    covered.set_bit(u);
                }
            }
        }
        self.buffer.clear();

        (
            covered.cardinality() != covered_before,
            None::<Box<dyn Postprocessor<Graph>>>,
        )
    }
}
