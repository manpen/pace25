use log::{info, trace};

use super::*;

impl<'a, G> Kernelization<'a, G>
where
    G: FullfledgedGraph,
{
    #[allow(dead_code)]
    pub(super) fn rule_two_paths(&mut self) -> bool {
        if self.slack < 2 {
            return false;
        }

        repeat_while!({
            let merges_before = self.sequence.len();
            info!("rule_two_path {merges_before}");

            for host in self.graph.vertices_range() {
                if self.is_protected(host) || self.graph.degree_of(host) < 2 {
                    continue;
                }

                let mut two_paths: Vec<_> = self
                    .graph
                    .neighbors_of(host)
                    .filter_map(|v| {
                        let w = self.graph.continue_path(host, v)?;
                        (!self.is_protected(w)
                            && self.graph.has_black_edge(host, v)
                            && self.graph.degree_of(w) == 1)
                            .then_some((v, w))
                    })
                    .collect();

                loop {
                    if two_paths.len() < 2 {
                        break;
                    }

                    let (u, v) = two_paths.pop().unwrap();
                    let (a, b) = *two_paths.last().unwrap();

                    self.graph.merge_node_into(u, a);
                    self.graph.merge_node_into(v, b);
                    self.sequence.merge_node_into(u, a);
                    self.sequence.merge_node_into(v, b);
                }
            }

            if merges_before != self.sequence.len() {
                trace!("Two-Path Merges: {}", self.sequence.len() - merges_before);
            }

            merges_before != self.sequence.len()
        })
    }
}
