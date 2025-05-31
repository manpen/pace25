use std::marker::PhantomData;

use itertools::Itertools;

use crate::graph::*;

use super::*;

pub struct RuleVertexCover<G> {
    _graph: PhantomData<G>,
}

const NOT_SET: Node = Node::MAX;

impl<Graph: AdjacencyList + AdjacencyTest + 'static> ReductionRule<Graph>
    for RuleVertexCover<Graph>
{
    const NAME: &str = "RuleVertexCover";

    fn apply_rule(
        graph: &mut Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        if redundant.cardinality() < 3 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        // most graphs do not contain the gadgets, so let's first collect all edges
        // (i.e. usually an empty vec!)
        let mut edges = redundant
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
            .collect_vec();

        edges.sort_unstable();
        edges.dedup();

        if edges.len() < 3 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        let mut old_to_new = vec![NOT_SET; graph.len()];
        let mut new_to_old = Vec::with_capacity(128);

        for &Edge(u, v) in &edges {
            if old_to_new[u as usize] == NOT_SET {
                old_to_new[u as usize] = new_to_old.len() as Node;
                new_to_old.push(u);
            }

            if old_to_new[v as usize] == NOT_SET {
                old_to_new[v as usize] = new_to_old.len() as Node;
                new_to_old.push(v);
            }
        }

        let mut vc_graph = AdjArray::from_edges(
            new_to_old.len() as Node,
            edges
                .into_iter()
                .map(|Edge(u, v)| Edge(old_to_new[u as usize], old_to_new[v as usize])),
        );
        let mut changed = false;

        let mut marker = old_to_new;
        marker.fill(NOT_SET);

        // search for cliques
        for u in vc_graph.vertices_range() {
            let deg = vc_graph.degree_of(u);
            if deg < 2 || 2 * deg != graph.degree_of(new_to_old[u as usize]) {
                continue;
            }

            for v in vc_graph.closed_neighbors_of(u) {
                marker[v as usize] = u;
            }

            if vc_graph.neighbors_of(u).any(|v| {
                solution.is_in_domset(v)
                    || vc_graph
                        .neighbors_of(v)
                        .filter(|w| marker[*w as usize] == u)
                        .count()
                        != deg as usize
            }) {
                continue;
            }

            let neighbors = vc_graph.closed_neighbors_of(u).collect_vec();

            for v in vc_graph.neighbors_of(u) {
                let v = new_to_old[v as usize];
                solution.add_node(v);
                covered.set_bits(graph.closed_neighbors_of(v));
            }

            for u in neighbors {
                vc_graph.remove_edges_at_node(u);
            }

            changed = true;
        }

        (changed, None::<Box<dyn Postprocessor<Graph>>>)
    }
}
