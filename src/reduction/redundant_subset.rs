use std::collections::HashSet;

use itertools::Itertools;

use crate::{
    exact::DominatingSet,
    graph::*,
    reduction::{Postprocessor, ReductionRule},
};

pub struct RuleRedundantSubsetReduction {
    redundant_degree: Vec<NumNodes>,
    candidates: Vec<Node>,
    offsets: Vec<usize>,
}

impl RuleRedundantSubsetReduction {
    pub fn new(n: NumNodes) -> Self {
        Self {
            redundant_degree: vec![0; n as usize],
            candidates: Vec::new(),
            offsets: Vec::new(),
        }
    }
}

impl<Graph: AdjacencyList + NeighborsSlice> ReductionRule<Graph> for RuleRedundantSubsetReduction {
    const NAME: &str = "RuleRedundantSubset";

    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        _solution: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> (bool, Option<Box<dyn super::Postprocessor<Graph>>>) {
        // covered is monotone and only ever adds nodes
        let prev_size = covered.cardinality();

        // Compute number of redundant neighbors for each node
        for u in graph.vertices() {
            self.redundant_degree[u as usize] = 0;
        }

        for u in never_select.iter_set_bits() {
            for v in graph.closed_neighbors_of(u) {
                self.redundant_degree[v as usize] += 1;
            }
        }

        {
            let mut red_twin: HashSet<Edge> =
                HashSet::with_capacity(never_select.cardinality() as usize);
            for u in never_select.iter_set_bits() {
                if let Some((a, b)) = graph.neighbors_of(u).collect_tuple() {
                    let norm = Edge(a, b).normalized();
                    if !red_twin.insert(norm) {
                        covered.set_bit(u);
                    }
                }
            }
        }


        // Sort adjacency lists to allow binary searching later on
        for u in graph.vertices_range() {
            graph.as_neighbors_slice_mut(u).sort_unstable();
        }

        for u in never_select.iter_set_bits() {
            // If node was already processed in a previous iteration, skip it
            if covered.get_bit(u) {
                continue;
            }

            if let Some(node) = graph
                .neighbors_of(u)
                .min_by_key(|&v| self.redundant_degree[v as usize])
            {
                for v in graph
                    .neighbors_of(node)
                    .filter(|&v| never_select.get_bit(v) && v != u && !covered.get_bit(v))
                {
                    self.candidates.push(v);
                    self.offsets.push(0);
                }
            } else {
                continue;
            }

            for v in graph.neighbors_of(u) {
                debug_assert!(!never_select.get_bit(v));

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

            // N(candidate) is a superset of N(u) for every candidate in self.candidates
            covered.set_bits(self.candidates.drain(..));
            self.offsets.clear();
        }

        (
            prev_size != covered.cardinality(),
            None::<Box<dyn Postprocessor<Graph>>>,
        )
    }
}
