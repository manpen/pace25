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

                self.sequence.merge_node_into(u, v);
                self.graph.merge_node_into(u, v);

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
        debug_assert_ne!(u, v);

        if self.is_protected(u) || self.is_protected(v) {
            return false;
        }

        // sutable degree 1 nodes were taken care by [`prune_leaves`]
        if self.graph.degree_of(u) < 2 || self.graph.degree_of(u) != self.graph.degree_of(v) {
            return false;
        }

        if self.graph.red_degree_of(u) != 0 && self.graph.red_degree_of(v) != 0 {
            let mut ru = self.graph.red_neighbors_of_as_bitset(u);
            let mut rv = self.graph.red_neighbors_of_as_bitset(v);

            ru.unset_bit(v);
            rv.unset_bit(u);

            if !ru.is_subset_of(&rv) && !rv.is_subset_of(&ru) {
                return false;
            }
        }

        let was_set_before = neighbors[u as usize].set_bit(v);
        neighbors[v as usize].set_bit(u);

        let are_twins = neighbors[u as usize] == neighbors[v as usize];

        if !was_set_before {
            neighbors[u as usize].unset_bit(v);
            neighbors[v as usize].unset_bit(u);
        }

        are_twins
    }
}
