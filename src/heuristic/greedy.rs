use crate::{
    graph::*,
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
    solution: &mut DominatingSet,
    covered_nodes: &BitSet,
    never_select: &BitSet,
) {
    // FIXME: this should be 0 going forward
    let prev_dom_nodes = solution.len();

    // Compute how many neighbors in the DomSet every node has
    let mut num_covered = vec![0usize; graph.number_of_nodes() as usize];
    covered_nodes
        .iter_set_bits()
        .for_each(|u| num_covered[u as usize] = 1);
    let mut total_covered = covered_nodes.cardinality();
    for u in solution.iter() {
        num_covered[u as usize] += 1;
        if num_covered[u as usize] == 1 {
            total_covered += 1;
        }
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
        if solution.is_in_domset(u) || never_select.get_bit(u) {
            continue;
        }

        // Neighborhood no longer closed
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
                    if v != node && !never_select.get_bit(v) {
                        let current_score = heap.remove(v);
                        heap.push(current_score + 1, v);
                    }
                }
            }
        }
    }

    // Remove redundant nodes from DomSet
    let mut index = prev_dom_nodes;
    while index < solution.len() {
        let node = solution.ith_node(index);
        if graph
            .neighbors_of(node)
            .all(|u| num_covered[u as usize] > 1)
        {
            for u in graph.closed_neighbors_of(node) {
                num_covered[u as usize] -= 1;
            }
            solution.remove_node(node);
            continue;
        }
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    use super::*;

    #[test]
    /// Randomly generate G(n,p) graphs and check that the algorithm produces a feasible solution
    fn full_graph() {
        let mut rng = Pcg64Mcg::seed_from_u64(123456);
        for _ in 0..1000 {
            let graph = AdjArray::random_black_gnp(&mut rng, 100, 0.03);
            let graph = CsrGraph::from_edges(graph.number_of_nodes(), graph.edges(true));
            let mut domset = DominatingSet::new(graph.number_of_nodes());
            super::greedy_approximation(
                &graph,
                &mut domset,
                &graph.vertex_bitset_unset(),
                &graph.vertex_bitset_unset(),
            );
            assert!(domset.is_valid(&graph));
        }
    }

    #[test]
    /// Randomly generate G(n,p) graphs and check that the algorithm produces a feasible solution
    fn graph_with_partial_solution() {
        let mut rng = Pcg64Mcg::seed_from_u64(123456);
        for i in 0..1000 {
            let org_graph = AdjArray::random_black_gnp(&mut rng, 100, 0.03);
            let org_graph =
                CsrGraph::from_edges(org_graph.number_of_nodes(), org_graph.edges(true));
            let graph = org_graph.clone(); // TODO: this should be mut
            let mut domset = DominatingSet::new(graph.number_of_nodes());

            for _ in 0..i % 10 {
                let fixed_node = rng.gen_range(graph.vertices_range());
                if domset.is_fixed_node(fixed_node) {
                    continue;
                }

                domset.fix_node(fixed_node);
                // TODO: We want to delete the node from the graph, but currently the interface does not allow it
            }

            super::greedy_approximation(
                &graph,
                &mut domset,
                &graph.vertex_bitset_unset(),
                &graph.vertex_bitset_unset(),
            );
            assert!(domset.is_valid(&org_graph));
        }
    }
}
