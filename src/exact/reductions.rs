use itertools::Itertools;
use log::trace;

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
    trace!(
        "Start initial pruning n={} nnz={} m={}",
        graph.number_of_nodes(),
        graph.number_of_nodes_with_neighbors(),
        graph.number_of_edges()
    );

    loop {
        let m = graph.number_of_edges();
        let k = contract_seq.merges().len();

        prune_leaves(graph, contract_seq);
        prune_twins(graph, contract_seq);

        if graph.number_of_edges() == m && contract_seq.merges().len() == k {
            break;
        }
    }

    trace!(
        "Done initial pruning n={} nnz={} m={} dd={:?}",
        graph.number_of_nodes(),
        graph.number_of_nodes_with_neighbors(),
        graph.number_of_edges(),
        graph.degree_distribution()
    );
}

pub fn default_pruning<
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
    trace!(
        "Start default pruning n={} nnz={} m={}",
        graph.number_of_nodes(),
        graph.number_of_nodes_with_neighbors(),
        graph.number_of_edges()
    );

    loop {
        let m = graph.number_of_edges();
        let k = contract_seq.merges().len();
        prune_tiny_graph(graph, slack, contract_seq);
        prune_leaves(graph, contract_seq);
        prune_leaves_at_red_node(graph, contract_seq);
        prune_twins(graph, contract_seq);
        prune_red_path(graph, slack, contract_seq);
        if graph.number_of_edges() == m && contract_seq.merges().len() == k {
            break;
        }
    }

    trace!(
        "Done default pruning n={} nnz={} m={}",
        graph.number_of_nodes(),
        graph.number_of_nodes_with_neighbors(),
        graph.number_of_edges()
    );
}

pub fn prune_outer_paths<
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
    if slack < 1 {
        return;
    }

    let mut contracts_before = usize::MAX;

    while contracts_before != contract_seq.merges().len() {
        contracts_before = contract_seq.merges().len();
        for middle in graph.vertices_range() {
            if graph.degree_of(middle) != 2 {
                continue;
            }

            let (leaf, inner) = {
                let mut iter = graph.neighbors_of(middle);
                let mut a = iter.next().unwrap();
                let mut b = iter.next().unwrap();

                if graph.degree_of(a) != 1 {
                    std::mem::swap(&mut a, &mut b);
                }

                (a, b)
            };

            if !(graph.degree_of(leaf) == 1 && graph.degree_of(inner) == 2) {
                continue;
            }

            if graph.red_degree_of(inner) != 0 && !graph.has_red_edge(middle, inner) {
                continue;
            }

            contract_seq.merge_node_into(leaf, middle);
            graph.merge_node_into(leaf, middle);
        }
    }
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

pub fn prune_leaves_at_red_node<
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
    loop {
        let merges_before = contract_seq.merges().len();
        for u in graph.vertices_range() {
            if graph.degree_of(u) != 1 {
                continue;
            }

            let v = graph.neighbors_of(u).next().unwrap();
            if graph.black_degree_of(v) > graph.black_degree_of(u) {
                continue;
            }

            graph.remove_edges_at_node(u);
            contract_seq.merge_node_into(u, v);
        }

        if contract_seq.merges().len() == merges_before {
            break;
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
            .filter(|&v| host != v && graph.degree_of(v) == 1)
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
            let mut ru = graph.red_neighbors_of_as_bitset(u);
            let mut rv = graph.red_neighbors_of_as_bitset(v);

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
        let mut front = graph.neighbors_of(u).next().unwrap();

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
                graph.neighbors_of(front).find(|&w| w != parent).unwrap(),
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

pub fn prune_red_bridges<
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
    mut solver: impl FnMut(&mut G, Node, NumNodes) -> Option<ContractionSequence>,
) {
    if slack < 2 {
        return;
    }

    let red_bridges = graph
        .compute_colored_bridges()
        .into_iter()
        .filter(|e| e.2.is_red())
        .collect_vec();

    if red_bridges.is_empty() {
        return;
    }

    for &ColoredEdge(u, v, _) in &red_bridges {
        let mut local_graph = graph.clone();
        local_graph.remove_edge(u, v);

        let part = local_graph.partition_into_connected_components(true);

        for (mut cc, mapper) in part.split_into_subgraphs(graph) {
            if cc.number_of_nodes() > 20 || cc.red_degrees().max().unwrap() >= slack {
                continue;
            }

            if let Some(seq) = solver(
                &mut cc,
                mapper.new_id_of(u).or(mapper.new_id_of(v)).unwrap(),
                slack,
            ) {
                for (removed, survivor) in seq
                    .merges()
                    .iter()
                    .map(|&(x, y)| (mapper.old_id_of(x).unwrap(), mapper.old_id_of(y).unwrap()))
                {
                    graph.merge_node_into(removed, survivor);
                    contract_seq.merge_node_into(removed, survivor);
                }

                let last_survivor = contract_seq.merges().last().unwrap().1;

                let over_the_bridge = if part.class_of_node(last_survivor) == part.class_of_node(u)
                {
                    v
                } else {
                    u
                };

                assert_ne!(
                    part.class_of_node(over_the_bridge),
                    part.class_of_node(last_survivor),
                    "{u} {v} {graph:?}"
                );

                graph.try_add_edge(over_the_bridge, last_survivor, EdgeColor::Red);
                return prune_red_bridges(graph, slack, contract_seq, solver);
            }
        }
    }
}
