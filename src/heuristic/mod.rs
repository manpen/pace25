use itertools::Itertools;

use crate::{
    graph::*,
    utils::{dominating_set::DominatingSet, radix::NodeHeap},
};

pub mod reverse_greedy_search;
pub mod subsets;

/// # The basic Greedy-Approximation
///
/// 1. Fixes
///     - Singletons
///     - Nodes that are the only neighbors of some other node
///     - Nodes part of a triangle where the other two nodes are only incident to the triangle
/// 2. Greedily adds nodes neighbored to the highest number of uncovered nodes to DomSet until
///    covered
/// 3. Remove nodes that now redundant, ie. do not cover any nodes that not other DomSet-Node
///    covers
///
/// Returns the solution
pub fn greedy_approximation(
    graph: &(impl AdjacencyList + AdjacencyTest + SelfLoop),
) -> DominatingSet {
    let mut solution = DominatingSet::new(graph.number_of_nodes());

    // Computed fixed nodes (ie. nodes that are guaranteed to be in an optimal solution)
    for u in graph.vertices() {
        match graph.degree_of(u) {
            1 => {
                solution.fix_node(u);
            }
            2 => {
                let (nb1, nb2) = graph.neighbors_of(u).collect_tuple().unwrap();
                let nb = if nb1 == u { nb2 } else { nb1 };

                if !solution.is_fixed_node(u) && !solution.is_fixed_node(nb) {
                    solution.fix_node(nb);
                }
            }
            3 => {
                let (nb1, nb2, nb3) = graph.neighbors_of(u).collect_tuple().unwrap();
                let (nb1, nb2) = if nb1 == u {
                    (nb2, nb3)
                } else if nb2 == u {
                    (nb1, nb3)
                } else {
                    (nb1, nb2)
                };

                if solution.is_fixed_node(u)
                    || solution.is_fixed_node(nb1)
                    || solution.is_fixed_node(nb2)
                {
                    continue;
                }

                if graph.degree_of(nb1) == 3 && graph.has_edge(nb1, nb2) {
                    solution.fix_node(nb2);
                } else if graph.degree_of(nb2) == 3 && graph.has_edge(nb2, nb1) {
                    solution.fix_node(nb1);
                }
            }
            _ => {}
        };
    }

    // Compute how many neighbors in the DomSet every node has
    let mut num_covered = vec![0usize; graph.number_of_nodes() as usize];
    let mut total_covered = 0;
    for u in solution.iter() {
        for v in graph.neighbors_of(u) {
            num_covered[v as usize] += 1;
            if num_covered[v as usize] == 1 {
                total_covered += 1;
            }
        }
    }

    // Compute scores for non-fixed nodes
    let mut heap = NodeHeap::new(graph.number_of_nodes() as usize, 0);
    for u in graph.vertices() {
        if solution.is_fixed_node(u) {
            continue;
        }

        let mut node_score = graph.max_degree();
        for v in graph.neighbors_of(u) {
            node_score -= (num_covered[v as usize] == 0) as NumNodes;
        }

        if node_score == graph.max_degree() {
            continue;
        }

        heap.push(node_score, u);
    }

    // Compute rest of DomSet via GreedyAlgorithm
    let max_degree = graph.max_degree();
    let degrees: Vec<NumNodes> = graph
        .vertices()
        .map(|u| (max_degree - graph.degree_of(u)))
        .collect();

    while total_covered < graph.number_of_nodes() {
        let (_, node) = heap.pop_with_tiebreaker(&degrees).unwrap();
        solution.add_node(node);

        for u in graph.neighbors_of(node) {
            num_covered[u as usize] += 1;
            if num_covered[u as usize] == 1 {
                total_covered += 1;
                for v in graph.neighbors_of(u) {
                    if v != node {
                        let current_score = heap.remove(v);
                        heap.push(current_score + 1, v);
                    }
                }
            }
        }
    }

    // Remove redundant nodes from DomSet
    let mut index = solution.num_of_fixed_nodes();
    while index < solution.len() {
        let node = solution.ith_node(index);
        if graph
            .neighbors_of(node)
            .all(|u| num_covered[u as usize] > 1)
        {
            for u in graph.neighbors_of(node) {
                num_covered[u as usize] -= 1;
            }
            solution.remove_node(node);
            continue;
        }

        index += 1;
    }

    solution
}
