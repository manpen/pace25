use super::*;

impl<'a, G> Kernelization<'a, G>
where
    G: FullfledgedGraph,
{
    pub(super) fn rule_leaves(&mut self) -> bool {
        repeat_while!({
            let merges_before = self.sequence.len();
            for host in self.graph.vertices_range() {
                if self.is_protected(host) || self.graph.degree_of(host) < 2 {
                    continue;
                }

                let mut neighbors: Vec<_> = self
                    .graph
                    .neighbors_of(host)
                    .iter()
                    .filter(|&&v| {
                        host != v && self.graph.degree_of(v) == 1 && !self.is_protected(v)
                    })
                    .copied()
                    .collect();

                if neighbors.len() < 2 {
                    continue;
                }

                let survivor = neighbors.pop().unwrap();

                if neighbors.iter().any(|&v| self.graph.red_degree_of(v) > 0) {
                    self.graph.try_add_edge(survivor, host, EdgeColor::Red);
                }

                for removed in neighbors {
                    self.sequence.merge_node_into(removed, survivor);
                    self.graph.remove_edges_at_node(removed);
                }
            }
            merges_before != self.sequence.len()
        })
    }
}
