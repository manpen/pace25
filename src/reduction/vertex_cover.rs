use itertools::Itertools;

use crate::graph::*;

use super::*;

#[derive(Default)]
pub struct RuleVertexCover {
    edges: Vec<Edge>,
    old_to_new: Vec<Node>,
    new_to_old: Vec<Node>,
    marker: Vec<Node>,
    neighbors: Vec<Node>,
}

impl RuleVertexCover {
    pub fn new(n: NumNodes) -> Self {
        Self {
            old_to_new: vec![NOT_SET; n as usize],
            marker: vec![NOT_SET; n as usize],
            ..Default::default()
        }
    }
}

const NOT_SET: Node = Node::MAX;

impl<Graph: AdjacencyList + AdjacencyTest + 'static> ReductionRule<Graph> for RuleVertexCover {
    const NAME: &str = "RuleVertexCover";

    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        if redundant.cardinality() < 3 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        self.edges.clear();

        // most graphs do not contain the gadgets, so let's first collect all edges
        // (i.e. usually an empty vec!)
        redundant
            .iter_set_bits()
            .filter_map(|u| {
                if graph.degree_of(u) != 2 || covered.get_bit(u) {
                    return None;
                }

                let (a, b) = graph.neighbors_of(u).collect_tuple()?;

                // there should not be any red-red edges!
                assert!(!redundant.get_bit(a) || !redundant.get_bit(b));

                Some(Edge(a, b).normalized())
            })
            .collect_into(&mut self.edges);

        self.edges.sort_unstable();
        self.edges.dedup();

        if self.edges.len() < 3 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        self.old_to_new = vec![NOT_SET; graph.len()];
        self.new_to_old.clear();

        for &Edge(u, v) in &self.edges {
            if self.old_to_new[u as usize] == NOT_SET {
                self.old_to_new[u as usize] = self.new_to_old.len() as Node;
                self.new_to_old.push(u);
            }

            if self.old_to_new[v as usize] == NOT_SET {
                self.old_to_new[v as usize] = self.new_to_old.len() as Node;
                self.new_to_old.push(v);
            }
        }

        let mut vc_graph = AdjArray::from_edges(
            self.new_to_old.len() as Node,
            self.edges
                .iter()
                .map(|&Edge(u, v)| Edge(self.old_to_new[u as usize], self.old_to_new[v as usize])),
        );
        let mut changed = false;

        self.marker = vec![NOT_SET; vc_graph.len()];

        // search for cliques
        'reject: for u in vc_graph.vertices_range() {
            // we need a clique of ... at least three nodes
            let deg = vc_graph.degree_of(u);
            if deg < 2 {
                continue;
            }

            // where u is the smallest node
            if vc_graph
                .neighbors_of(u)
                .any(|v| vc_graph.degree_of(v) > deg)
            {
                continue;
            }

            for v in vc_graph.closed_neighbors_of(u) {
                self.marker[v as usize] = u;
            }

            let mut org_has_int_edge = false;
            let ou = self.new_to_old[u as usize];
            for ov in graph.neighbors_of(ou) {
                // case 1: there is a mapped neighbor (then it's in the clique)
                let nv = self.old_to_new[ov as usize];
                if nv != NOT_SET {
                    // if it's in our clique: then we have the edge, we need
                    if self.marker[nv as usize] == u {
                        org_has_int_edge = true;
                        continue;
                    } else {
                        // otherwise we do not have a clique (which me happen, since we check that only after this loop!s)
                        continue 'reject;
                    }
                }

                // case 2: the neighbor is covered (then we can ignore it)
                if covered.get_bit(ov) {
                    continue;
                }

                // case 3: it's redundant ... then at least one other neighbor needs to be in the clique (other than u)
                if redundant.get_bit(ov)
                    && graph.neighbors_of(ov).any(|w| {
                        w != ou
                            && self.old_to_new[w as usize] != NOT_SET
                            && self.marker[self.old_to_new[w as usize] as usize] == u
                    })
                {
                    continue;
                }

                continue 'reject;
            }

            if !org_has_int_edge {
                continue;
            }

            if vc_graph.neighbors_of(u).any(|v| {
                solution.is_in_domset(v)
                    || vc_graph
                        .neighbors_of(v)
                        .filter(|w| self.marker[*w as usize] == u)
                        .count()
                        != deg as usize
            }) {
                continue;
            }

            vc_graph
                .closed_neighbors_of(u)
                .collect_into(&mut self.neighbors);

            for v in vc_graph.neighbors_of(u) {
                let v = self.new_to_old[v as usize];
                solution.add_node(v);
                covered.set_bits(graph.closed_neighbors_of(v));
            }

            for u in self.neighbors.drain(..) {
                vc_graph.remove_edges_at_node(u);
            }

            changed = true;
        }

        (changed, None::<Box<dyn Postprocessor<Graph>>>)
    }
}
