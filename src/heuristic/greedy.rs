use crate::{
    graph::*,
    kernelization::{KernelizationRule, rule1::Rule1},
    utils::{dominating_set::DominatingSet, radix::NodeHeap},
};

/// # The basic Greedy-Approximation
///
/// 1. Fixes nodes in complete subgraphs where at most one node has neighbors outside the subgraph
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

    // Apply Rule1
    Rule1::apply_rule(graph, &mut solution);

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
    while total_covered < graph.number_of_nodes() {
        let (_, node) = heap.pop().unwrap();
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
