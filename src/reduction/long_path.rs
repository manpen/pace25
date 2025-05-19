use super::*;
use crate::graph::*;
use itertools::Itertools;
use std::marker::PhantomData;

#[must_use] // this rule has a post-processing step and may cause invalid results if not applied
pub struct LongPathReduction<G> {
    removed_paths: Vec<Vec<Node>>,
    total_nodes_deleted: NumNodes,
    _graph: PhantomData<G>,
}

/// This reduction rule shortens paths of lengths at least 5 by removing groups of three nodes
/// until as long as at least 2 nodes remain. It is a two-staged rule that requires post-processing
/// once a domset for the modified graph was computed.
impl<G: AdjacencyList + GraphEdgeEditing + 'static> ReductionRule<G> for LongPathReduction<G> {
    const NAME: &str = "LongPathReduction";

    fn apply_rule(
        graph: &mut G,
        _solution: &mut DominatingSet,
        covered: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<G>>>) {
        let long_paths = graph.path_iter_with_atleast_path_nodes(5).collect_vec();
        if long_paths.is_empty() {
            return (false, None);
        }

        let mut total_nodes_deleted = 0;
        for path in &long_paths {
            // we can remove groups of three as long as at least four nodes remain
            let nodes_to_remove = ((path.len() - 4) / 3) * 3;

            if path[2..2 + nodes_to_remove]
                .iter()
                .any(|&u| !covered.get_bit(u))
            {
                // TODO: Fix me
                continue;
            }

            for &u in &path[2..2 + nodes_to_remove] {
                graph.remove_edges_at_node(u);
                covered.set_bit(u);
            }

            total_nodes_deleted += nodes_to_remove as NumNodes;
            graph.add_edge(path[1], path[2 + nodes_to_remove], EdgeColor::Black);
        }

        (
            true,
            Some(Box::new(Self {
                removed_paths: long_paths,
                total_nodes_deleted,
                _graph: Default::default(),
            })),
        )
    }
}

impl<G: AdjacencyList + GraphEdgeEditing> Postprocessor<G> for LongPathReduction<G> {
    fn post_process(&mut self, _graph: &mut G, solution: &mut DominatingSet, covered: &mut BitSet) {
        for path in self.removed_paths.drain(..) {
            for (i, &u) in path.iter().enumerate() {
                if covered.set_bit(u) {
                    continue;
                }

                let add = path[(i + 1).min(path.len() - 1)];
                solution.add_node(add);
                covered.set_bit(add);
                if i + 2 < path.len() {
                    covered.set_bit(path[i + 2]);
                }
            }
        }
    }
}

impl<G> LongPathReduction<G> {
    pub fn total_nodes_deleted(&self) -> NumNodes {
        self.total_nodes_deleted
    }
}
