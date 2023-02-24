use itertools::Itertools;

use super::*;
use crate::prelude::*;
use std::fmt::Debug;

pub fn initial_pruning<
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
    prune_leaves(graph, contract_seq);
    prune_twins(graph, contract_seq);
}

pub fn prune_tiny_graph<
    G: Clone
        + AdjacencyList
        + GraphEdgeOrder
        + ColoredAdjacencyList
        + ColoredAdjacencyTest
        + GraphEdgeEditing
        + Debug,
>(
    graph: &mut G,
    slack: NumNodes,
    contract_seq: &mut ContractionSequence,
) {
    if graph.number_of_edges() == 0
        || graph.number_of_edges() > (slack as NumEdges) * (slack.saturating_sub(1) as NumEdges) / 2
    {
        return;
    }

    if graph
        .degrees()
        .filter(|&d| d > 0)
        .take(2 + slack as usize)
        .count()
        <= 1 + slack as usize
    {
        let mut remaining_nodes: Vec<_> = graph
            .vertices()
            .filter(|&u| graph.degree_of(u) > 0)
            .collect();

        for &x in &remaining_nodes {
            graph.remove_edges_at_node(x);
        }

        let survivor = remaining_nodes.pop().unwrap();
        for removed in remaining_nodes {
            contract_seq.merge_node_into(removed, survivor);
        }
    }
}

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
    for host in graph.vertices_range() {
        if graph.degree_of(host) < 2 {
            continue;
        }

        let mut neighbors = graph
            .neighbors_of(host)
            .iter()
            .filter(|&&v| host != v && graph.degree_of(v) == 1)
            .copied()
            .collect_vec();

        if neighbors.len() < 2 {
            continue;
        }

        let survivor = neighbors.pop().unwrap();

        if neighbors.iter().any(|&v| graph.red_degree_of(v) > 0) {
            graph.try_add_edge(survivor, host, EdgeColor::Red);
        }

        for removed in neighbors {
            contract_seq.merge_node_into(removed, survivor);
            graph.remove_edges_at_node(removed);
        }
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
    let mut neighbors: Vec<_> = graph.neighbors_as_bitset().collect();
    neighbors.iter_mut().enumerate().for_each(|(i, bs)| {
        bs.set_bit(i as Node);
    });

    let are_twins = |graph: &G, neighbors: &mut [BitSet], u: Node, v: Node| {
        debug_assert_ne!(u, v);

        // sutable degree 1 nodes were taken care by [`prune_leaves`]
        if graph.degree_of(u) < 2 || graph.degree_of(u) != graph.degree_of(v) {
            return false;
        }

        if graph.red_degree_of(u) != 0 && graph.red_degree_of(v) != 0 {
            let ru = graph.red_neighbors_of_as_bitset(u);
            let rv = graph.red_neighbors_of_as_bitset(v);

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
    };

    loop {
        let twins: Vec<_> = graph
            .distance_two_pairs()
            .filter(|&(u, v)| are_twins(graph, &mut neighbors, u, v))
            .collect();

        if twins.is_empty() {
            break;
        }

        for (u, v) in twins {
            if !are_twins(graph, &mut neighbors, u, v) {
                continue;
            }

            contract_seq.merge_node_into(u, v);
            graph.merge_node_into(u, v);

            let mut nu = std::mem::take(&mut neighbors[u as usize]);
            neighbors[v as usize].or(&nu);

            nu.unset_all();
            neighbors[u as usize] = nu;

            neighbors.iter_mut().for_each(|n| {
                n.unset_bit(u);
            });
        }
    }
}

pub fn prune_red_path<
    G: Clone
        + AdjacencyList
        + GraphEdgeOrder
        + ColoredAdjacencyList
        + ColoredAdjacencyTest
        + GraphEdgeEditing
        + ColorFilter
        + Debug,
>(
    graph: &mut G,
    slack: NumNodes,
    contract_seq: &mut ContractionSequence,
) {
    if slack < 2 {
        return;
    }

    let mut path = Vec::new();
    'next_node: for u in graph.vertices_range() {
        if graph.degree_of(u) != 1 || graph.red_degree_of(u) > 0 {
            continue;
        }

        path.clear();
        path.push(u);

        let mut parent = u;
        let mut front = graph.neighbors_of(u)[0];

        loop {
            if graph.degree_of(front) != 2 {
                continue 'next_node;
            }

            path.push(front);

            if graph.red_degree_of(front) > 0 {
                break;
            }

            (parent, front) = (
                front,
                graph.neighbors_of(front)[(graph.neighbors_of(front)[0] == parent) as usize],
            );
        }

        if path.len() < 3 {
            continue;
        }

        for (&removed, &survivor) in path.iter().tuple_windows() {
            contract_seq.merge_node_into(removed, survivor);
            graph.merge_node_into(removed, survivor);
        }
    }
}
