use log::info;

use super::*;

impl<'a, G> Kernelization<'a, G>
where
    G: FullfledgedGraph,
{
    #[allow(dead_code)]
    pub(super) fn rule_tree(&mut self) -> bool {
        repeat_while!({
            let merges_before = self.sequence.len();
            info!("rule_tree {merges_before}");

            if self.graph.number_of_edges() == 0 {
                return false;
            }

            // is whole graph a tree?
            {
                let root = self.graph.vertices_with_neighbors().next().unwrap();
                if self.is_tree(root) {
                    self.solve_tree(root, false, root);
                }
            }

            for ColoredEdge(u, v, c) in self.graph.compute_colored_bridges() {
                if (c.is_red() && self.slack < 3)
                    || self.graph.degree_of(u) <= 2 && self.graph.degree_of(v) <= 2
                {
                    continue;
                };

                self.graph.remove_edge(u, v);

                for x in [u, v] {
                    if !self.is_tree(x) {
                        continue;
                    }

                    self.solve_tree(x, true, x);
                }

                self.graph.add_edge(u, v, EdgeColor::Black);
            }

            if merges_before < self.sequence.len() {
                println!(
                    "Pruned: {}; Slack: {}",
                    self.sequence.len() - merges_before,
                    self.slack
                );
            }

            merges_before != self.sequence.len()
        })
    }

    fn is_tree(&self, root: Node) -> bool {
        let mut visited = BitSet::new(self.graph.number_of_nodes());

        for (parent, node) in self.graph.bfs_with_predecessor(root) {
            if !self
                .graph
                .neighbors_of(node)
                .iter()
                .all(|&v| v == parent || !visited.set_bit(v))
            {
                return false;
            }
        }

        true
    }

    fn solve_tree(&mut self, root: Node, protect_root: bool, parent: Node) {
        let mut prev = None;

        for red_edges in [true, false] {
            let mut children = if red_edges {
                if self.graph.red_degree_of(root) == 0 {
                    continue;
                }
                self.graph.red_neighbors_of_as_bitset(root)
            } else {
                self.graph.black_neighbors_of_as_bitset(root)
            };

            children.unset_bit(parent);

            for child in children.iter() {
                self.solve_tree(child, false, root);
                assert_eq!(self.graph.degree_of(child), 1);

                if self.slack < 2 {
                    self.slack = self.slack.max(self.graph.red_degree_of(root));
                }

                if let Some(prev) = prev {
                    self.graph.merge_node_into(prev, child);
                    self.sequence.merge_node_into(prev, child);
                }

                prev = Some(child);
            }
        }

        if protect_root {
            return;
        }

        if let Some(child) = prev {
            self.graph.merge_node_into(child, root);
            self.sequence.merge_node_into(child, root);
        }
    }
}
