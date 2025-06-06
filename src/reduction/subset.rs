use crate::{
    exact::DominatingSet,
    graph::*,
    reduction::{Postprocessor, ReductionRule},
};

/// # Subset-Rule
/// A node is subset-dominated by another node, if its neighborhood is a subset of the others neighborhood.
/// Here, we only consider the restricted neighborhoods in which nodes that are always covered by fixed nodes
/// are left out of consideration.
pub struct RuleSubsetReduction {
    non_perm_degree: Vec<NumNodes>,
    candidates: Vec<Node>,
    offsets: Vec<usize>,
}

impl RuleSubsetReduction {
    pub fn new(n: NumNodes) -> Self {
        Self {
            non_perm_degree: vec![0; n as usize],
            candidates: Vec::new(),
            offsets: Vec::new(),
        }
    }
}

impl<Graph: AdjacencyList + NeighborsSlice> ReductionRule<Graph> for RuleSubsetReduction {
    const NAME: &str = "RuleSubset";

    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        _solution: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> (bool, Option<Box<dyn super::Postprocessor<Graph>>>) {
        // never_select is monotone and only ever adds nodes
        let prev_size = never_select.cardinality();

        // Compute permanently covered nodes and degrees
        for u in graph.vertices() {
            self.non_perm_degree[u as usize] = graph.degree_of(u) + 1;
        }

        for u in covered.iter_set_bits() {
            for v in graph.closed_neighbors_of(u) {
                self.non_perm_degree[v as usize] -= 1;
            }
        }

        // Sort adjacency lists to allow binary searching later on
        for u in graph.vertices_range() {
            graph.as_neighbors_slice_mut(u).sort_unstable();
        }

        for u in graph.vertices() {
            if never_select.get_bit(u) {
                continue;
            }

            // If every neighbor (including u) is permanently covered, skip this node
            // Otherwise pick node with maximum number of non-permanently covered neighbors
            if let Some(node) = graph
                .closed_neighbors_of(u)
                .filter(|&v| !covered.get_bit(v))
                .min_by_key(|&v| self.non_perm_degree[v as usize])
            {
                // Only nodes that are not alredy marked as never_select are qualified to be
                // candidates (as they either do not need to be marked twice or can not be used as
                // a witness for marking u)
                for v in graph
                    .closed_neighbors_of(node)
                    .filter(|&v| !never_select.get_bit(v) && v != u)
                {
                    self.candidates.push(v);
                    self.offsets.push(0);
                }
            } else {
                continue;
            }

            // Only consider candidates that are adjacent to all non-permanently covered neighbors in N(u)
            for v in graph.neighbors_of(u) {
                if covered.get_bit(v) {
                    continue;
                }

                for i in (0..self.candidates.len()).rev() {
                    let candidate = self.candidates[i];
                    if let Ok(index) =
                        graph.as_neighbors_slice(candidate)[self.offsets[i]..].binary_search(&v)
                    {
                        // Since edge-lists are sorted, v is increasing and we can use offsets[i] to
                        // allow for faster binary searches in later iterations
                        self.offsets[i] += index;
                    } else {
                        self.candidates.swap_remove(i);
                        self.offsets.swap_remove(i);
                    }
                }
            }

            // Check u
            if !covered.get_bit(u) {
                for i in (0..self.candidates.len()).rev() {
                    let candidate = self.candidates[i];
                    if graph.as_neighbors_slice(candidate)[self.offsets[i]..]
                        .binary_search(&u)
                        .is_err()
                    {
                        self.candidates.swap_remove(i);
                        self.offsets.swap_remove(i);
                    }
                }
            }

            // Consume candidates to find a dominating node of u
            for candidate in self.candidates.drain(..) {
                // Since possibly min_neighbor = Some(u), we always have
                // non_perm_degree[candidate as usize] <= non_perm_degree[u as usize]
                //
                // Inequality thus implies that the right side is bigger
                if self.non_perm_degree[candidate as usize] == self.non_perm_degree[u as usize]
                    && u > candidate
                {
                    never_select.set_bit(candidate);
                } else {
                    never_select.set_bit(u);
                }
            }

            self.offsets.clear();
        }

        (
            prev_size != never_select.cardinality(),
            None::<Box<dyn Postprocessor<Graph>>>,
        )
    }
}
