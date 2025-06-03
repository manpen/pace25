use std::marker::PhantomData;

use super::*;
use crate::{graph::*, utils::DominatingSet};

pub struct SubsetRuleTwoReduction<G> {
    _graph: PhantomData<G>,
}

const NOT_SET: Node = Node::MAX;

impl<Graph: AdjacencyList + GraphEdgeEditing + 'static> ReductionRule<Graph>
    for SubsetRuleTwoReduction<Graph>
{
    const NAME: &str = "SubsetRuleTwoReduction";

    fn apply_rule(
        graph: &mut Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        let n = graph.len();
        assert!(NOT_SET as usize >= n);

        (false, None::<Box<dyn Postprocessor<Graph>>>)
    }
}
