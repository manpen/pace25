use super::*;

impl<'a, G> Kernelization<'a, G>
where
    G: FullfledgedGraph,
{
    pub(super) fn rule_twins(&mut self) -> bool {
        let mut neighbors: Vec<_> = self.graph.neighbors_as_bitset().collect();
        neighbors.iter_mut().enumerate().for_each(|(i, bs)| {
            bs.set_bit(i as Node);
        });

        let merges_before = self.sequence.len();

        loop {
            let twins: Vec<_> = self
                .graph
                .distance_two_pairs()
                .filter(|&(u, v)| self.are_twins(&mut neighbors, u, v))
                .collect();

            if twins.is_empty() {
                break;
            }

            for (u, v) in twins {
                if !self.are_twins(&mut neighbors, u, v) {
                    continue;
                }

                let red_before = self.graph.red_degree_of(u).max(self.graph.red_degree_of(v));
                self.sequence.merge_node_into(u, v);
                self.graph.merge_node_into(u, v);
                assert!(red_before >= self.graph.red_degree_of(v));

                let mut nu = std::mem::take(&mut neighbors[u as usize]);
                neighbors[v as usize].or(&nu);

                nu.unset_all();
                neighbors[u as usize] = nu;

                neighbors.iter_mut().for_each(|n| {
                    n.unset_bit(u);
                });
            }
        }

        merges_before != self.sequence.len()
    }

    fn are_twins(&self, neighbors: &mut [BitSet], u: Node, v: Node) -> bool {
        if self.is_protected(u) || self.is_protected(v) {
            return false;
        }

        self.are_twins_impl(neighbors, u, v) || self.are_twins_impl(neighbors, v, u)
    }

    fn are_twins_impl(&self, neighbors: &mut [BitSet], smaller: Node, larger: Node) -> bool {
        debug_assert_ne!(smaller, larger);

        let was_set_before = neighbors[smaller as usize].set_bit(larger);

        let are_twins = (|| {
            let mut lneighbor = neighbors[larger as usize].clone();
            lneighbor.set_bit(smaller);

            if !neighbors[smaller as usize].is_subset_of(&lneighbor) {
                return false;
            }

            lneighbor.and_not(&neighbors[smaller as usize]);

            lneighbor.is_subset_of(&self.graph.red_neighbors_of_as_bitset(larger))
                && self
                    .graph
                    .red_neighbors_of_as_bitset(smaller)
                    .is_subset_of(&self.graph.red_neighbors_of_as_bitset(larger))
        })();

        if !was_set_before {
            neighbors[smaller as usize].unset_bit(larger);
        }

        are_twins
    }
}
