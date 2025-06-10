use itertools::Itertools;

use crate::{graph::*, utils::NodeMarker};

use super::*;

#[derive(Default)]
pub struct RuleVertexCover {
    edges: Vec<Edge>,
    old_to_new: NodeMarker,
    new_to_old: Vec<Node>,
    marker: NodeMarker,
    neighbors: Vec<Node>,
}

impl RuleVertexCover {
    pub fn new(n: NumNodes) -> Self {
        Self {
            old_to_new: NodeMarker::new(n, NOT_SET),
            marker: NodeMarker::new(n, NOT_SET),
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
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        if never_select.cardinality() < 3 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        self.edges.clear();

        // most graphs do not contain the gadgets, so let's first collect all edges
        // (i.e. usually an empty vec!)
        never_select
            .iter_set_bits()
            .filter_map(|u| {
                if graph.degree_of(u) != 2 || covered.get_bit(u) {
                    return None;
                }

                let (a, b) = graph.neighbors_of(u).collect_tuple()?;

                // there should not be any red-red edges!
                assert!(!never_select.get_bit(a) || !never_select.get_bit(b));

                Some(Edge(a, b).normalized())
            })
            .collect_into(&mut self.edges);

        self.edges.sort_unstable();
        self.edges.dedup();

        if self.edges.len() < 3 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        self.old_to_new.reset();
        self.new_to_old.clear();

        for &Edge(u, v) in &self.edges {
            if !self.old_to_new.is_marked(u) {
                self.old_to_new.mark_with(u, self.new_to_old.len() as Node);
                self.new_to_old.push(u);
            }

            if !self.old_to_new.is_marked(v) {
                self.old_to_new.mark_with(v, self.new_to_old.len() as Node);
                self.new_to_old.push(v);
            }
        }

        let mut vc_graph = AdjArray::from_edges(
            self.new_to_old.len() as Node,
            self.edges
                .iter()
                .map(|&Edge(u, v)| Edge(self.old_to_new.get_mark(u), self.old_to_new.get_mark(v))),
        );
        let mut changed = false;

        self.marker.reset_up_to(vc_graph.len() as NumNodes);

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

            self.marker
                .mark_all_with(vc_graph.closed_neighbors_of(u), u);

            let mut org_has_int_edge = false;
            let ou = self.new_to_old[u as usize];
            for ov in graph.neighbors_of(ou) {
                // case 1: there is a mapped neighbor (then it's in the clique)
                let nv = self.old_to_new.get_mark(ov);
                if nv != NOT_SET {
                    // if it's in our clique: then we have the edge, we need
                    if self.marker.is_marked_with(nv, u) {
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
                if never_select.get_bit(ov)
                    && graph.neighbors_of(ov).any(|w| {
                        w != ou
                            && self.old_to_new.is_marked(w)
                            && self.marker.is_marked_with(self.old_to_new.get_mark(w), u)
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
                domset.is_in_domset(v)
                    || vc_graph
                        .neighbors_of(v)
                        .filter(|&w| self.marker.is_marked_with(w, u))
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
                domset.fix_node(v);
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
