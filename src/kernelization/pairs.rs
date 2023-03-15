use super::*;

impl<'a, G> Kernelization<'a, G>
where
    G: FullfledgedGraph,
{
    pub(super) fn rule_pairs(&mut self) -> bool {
        repeat_while!({
            let merges_before = self.sequence.len();
            for host in self.graph.vertices_range() {
                if self.is_protected(host) || self.graph.degree_of(host) != 1 {
                    continue;
                }

                let neighbor = self.graph.neighbors_of(host)[0];

                if neighbor < host
                    || self.is_protected(neighbor)
                    || self.graph.degree_of(neighbor) != 1
                {
                    continue;
                }

                self.sequence.merge_node_into(host, neighbor);
                self.graph.remove_edges_at_node(host);
            }
            merges_before != self.sequence.len()
        })
    }
}
