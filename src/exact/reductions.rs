use super::*;
use crate::graph::*;
use std::fmt::Debug;

pub fn prune_leaves<
    G: Clone
        + AdjacencyList
        + GraphEdgeOrder
        + ColoredAdjacencyList
        + ColoredAdjacencyTest
        + GraphEdgeEditing
        + Debug,
>(
    graph: &mut G,
    contract_seq: &mut ContractionSequence,
) {
    while let Some((u, v)) = graph
        .distance_two_pairs()
        .find(|&(u, v)| graph.degree_of(u) == 1 && graph.degree_of(v) == 1)
    {
        graph.merge_node_into(u, v);
        contract_seq.merge_node_into(u, v);
    }
}

pub fn prune_twins<
    G: Clone
        + AdjacencyList
        + GraphEdgeOrder
        + ColoredAdjacencyList
        + ColoredAdjacencyTest
        + GraphEdgeEditing
        + Debug,
>(
    graph: &mut G,
    contract_seq: &mut ContractionSequence,
) {
    while let Some((u, v)) = graph.distance_two_pairs().find(|&(u, v)| {
        if graph.red_degree_of(u) != 0 && graph.red_degree_of(v) != 0 {
            return false;
        }

        if graph.black_degree_of(u) == 0 || graph.black_degree_of(u) != graph.black_degree_of(v) {
            return false;
        }

        let mut s1 = graph.neighbors_of_as_bitset(u);
        let mut s2 = graph.neighbors_of_as_bitset(v);

        s1.set_bit(u);
        s1.set_bit(v);
        s2.set_bit(u);
        s2.set_bit(v);

        s1 == s2
    }) {
        contract_seq.merge_node_into(u, v);
        graph.remove_edges_at_node(u);
    }
}
