use smallvec::SmallVec;

use crate::{graph::*, utils::DominatingSet};

use super::KernelizationRule;

/// Rule1
///
/// For node u, partition its neighborhood into 3 distinct sets:
/// - N1(u) = set of all neighbors v that have a neighbor x not incident to u               (=Type1-Neighbors)
/// - N2(u) = set of all neighbors v not in N1(u) that are incident to a node in N1(u)      (=Type2-Neighbors)
/// - N3(u) = set of all remaining neighbors (ie. only incident to u and N2(u))             (=Type3-Neighbors)
///
/// It can be shown that if |N3(u)| > 0, u is part of an optimal dominating set and all nodes in
/// N2(u) and N3(u) must not be considered in further computations.
///
/// This algorithm correctly computes (and fixes in the provided DominatingSet) all nodes u with
/// |N3(u)| > 0. It also returns a BitSet indicating which nodes are in the Type(2 or 3)-Neighborhood
/// of u and can thus be removed. Note that this will *not* return all such Type(2 or 3)-Neighbors,
/// only some.
///
/// Rough steps of the algorithm are:
/// (1) We first compute a superset of possible candidates (u, v) where u is a possible Type3-Neighbor
/// for v. We only consider nodes for v which have the maximum degree among the neighborhood of u.
/// (2) We compute a mapping f: V -> V that maps Type3-Nodes to their respective dominating node.
/// Here, we break ties in the opposite direction and prefer dominating nodes with smaller degrees.
/// (3) We iterate over each candidate-pair (u,v) and confirm whether u is truly a Type3-Neighbor
/// for u. If true, we mark u as redundant and fix v as a dominating node.
pub struct Rule1;

const NOT_SET: Node = Node::MAX;

impl<Graph: AdjacencyList + SelfLoop> KernelizationRule<&Graph> for Rule1 {
    fn apply_rule(graph: &Graph, sol: &mut DominatingSet) -> BitSet {
        let n = graph.len();
        assert!(NOT_SET as usize >= n);

        // Inverse mappings of step (1) and (2)
        let mut inv_mappings: Vec<SmallVec<[Node; 4]>> = vec![Default::default(); n];

        // Parent[u] = v if (u,v) is a possible candidate
        let mut parent: Vec<Node> = vec![NOT_SET; n];

        // Used for confirming whether neighborhoods of nodes are subsets of other neighborhoods.
        let mut marked: Vec<Node> = vec![NOT_SET; n];

        // List of all possible candidate-pairs
        let mut potential_type3_node: Vec<(Node, Node)> = Vec::new();

        // BitSet indicating that a node is a Type(2 or 3)-Candidate (not confirmed yet)
        let mut type2_nodes: BitSet = graph.vertex_bitset_unset();

        // Helper-BitSet to ensure we only process each node once later
        let mut processed: BitSet = graph.vertex_bitset_unset();

        // BitSet indicating redundant nodes -> returned at the end
        let mut redundant = graph.vertex_bitset_unset();

        // (1) Compute first mapping and fix possible singletons
        for u in graph.vertices() {
            let max_neighbor = graph
                .neighbors_of(u)
                .map(|u| (graph.degree_of(u), u))
                .max()
                .map(|(_, u)| u)
                .unwrap();
            if max_neighbor != u {
                inv_mappings[max_neighbor as usize].push(u);
            } else if graph.degree_of(u) == 1 && !sol.is_fixed_node(u) {
                sol.fix_node(u);
            }
        }

        // (1) Compute list of candidate-pairs based on mapping
        for u in graph.vertices() {
            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            for v in graph.neighbors_of(u) {
                marked[v as usize] = u;
            }

            // Check whether N[v] is a subset of N[u]
            for v in inv_mappings[u as usize].drain(..) {
                if graph.neighbors_of(v).all(|x| marked[x as usize] == u) {
                    parent[v as usize] = u;
                    potential_type3_node.push((v, u));

                    type2_nodes.set_bit(v);
                }
            }
        }

        // We drained inv_mappings earlier completely, so we can now reuse it
        debug_assert!(inv_mappings.iter().all(|vec| vec.is_empty()));

        // (2) Compute second mapping from list of candidate-pairs
        for &(u, _) in &potential_type3_node {
            for v in graph.neighbors_of(u) {
                // Only process each node once
                if u == v || processed.get_bit(v) {
                    continue;
                }

                // Mark closed neighborhood N[v] of v (SelfLoop marker)
                for x in graph.neighbors_of(v) {
                    marked[x as usize] = v;
                }

                // Find minimum dominating node of neighbors in neighborhood of v
                if let Some((_, min_node)) = graph
                    .neighbors_of(v)
                    .filter_map(|x| {
                        let pt = parent[x as usize];
                        (pt != NOT_SET && pt != v && marked[pt as usize] == v)
                            .then(|| (graph.degree_of(pt), pt))
                    })
                    .min()
                {
                    // We drained inv_mappings earlier completely, so we can now reuse it
                    inv_mappings[min_node as usize].push(v);
                }

                processed.set_bit(v);
            }
        }

        // (3) Mark candidates as possible Type2-Nodes if their neighborhoods are subsets
        for u in graph.vertices() {
            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            for v in graph.neighbors_of(u) {
                marked[v as usize] = u;
            }

            for v in inv_mappings[u as usize].drain(..) {
                if graph.neighbors_of(v).all(|x| marked[x as usize] == u) {
                    type2_nodes.set_bit(v);
                }
            }
        }

        // (3) If all neighbors of a candidate are marked as Type2-Neighbors (except the dominating node),
        // the candidate must be a Type3-Neighbor and we can fix a node and mark all remaining Type2-Neighbors as redundant.
        for (u, v) in potential_type3_node {
            if sol.is_fixed_node(v) {
                // By assumption, u is not a Type1-Neighbor
                redundant.set_bit(u);
                continue;
            }

            if graph
                .neighbors_of(u)
                .all(|x| type2_nodes.get_bit(x) || x == v)
            {
                // By assumption, u is not a Type1-Neighbor
                sol.fix_node(v);
                redundant.set_bit(u);
            }
        }

        redundant
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    // The standard Rule implementation with runtime O(n * D * D) where D is the maximum degree in the graph.
    fn naive_rule1_impl(graph: &(impl AdjacencyList + SelfLoop), sol: &mut DominatingSet) {
        let mut marked = graph.vertex_bitset_unset();
        let mut type2_nodes = graph.vertex_bitset_unset();
        for u in graph.vertices() {
            if graph.degree_of(u) == 1 {
                sol.fix_node(u);
                continue;
            }

            marked.clear_all();
            type2_nodes.clear_all();
            for v in graph.neighbors_of(u) {
                marked.set_bit(v);
            }

            for v in graph.neighbors_of(u) {
                if u == v {
                    continue;
                }

                if graph.neighbors_of(v).all(|x| marked.get_bit(x)) {
                    if graph.degree_of(v) == graph.degree_of(u) && u < v {
                        continue;
                    }

                    type2_nodes.set_bit(v);
                }
            }

            let mut type3 = false;
            for v in graph.neighbors_of(u) {
                if u == v {
                    continue;
                }

                type3 = type3
                    || graph
                        .neighbors_of(v)
                        .all(|x| x == u || type2_nodes.get_bit(x));
            }

            if type3 {
                sol.fix_node(u);
            }
        }
    }

    fn get_random_graph(rng: &mut impl Rng, n: NumNodes, m: NumEdges) -> CsrGraph {
        let mut set = BitSet::new(n * n);
        let mut edges: Vec<Edge> = Vec::with_capacity(m as usize);
        while edges.len() < m as usize {
            let u = rng.gen_range(0..n);
            let v = rng.gen_range(0..n);
            if Edge(u, v).is_loop() {
                continue;
            }

            if !set.set_bit(u * v) {
                edges.push(Edge(u, v));
            }
        }

        CsrGraph::from_edges(n, edges)
    }

    #[test]
    fn compare_rule1_implementations() {
        let rng = &mut rand::thread_rng();

        for _ in 0..100 {
            let n = rng.gen_range(5..50);
            let m = rng.gen_range(1..(n * (n - 1) / 4)) as NumEdges;

            let graph = get_random_graph(rng, n, m);

            let mut sol1 = DominatingSet::new(n);
            let mut sol2 = sol1.clone();

            Rule1::apply_rule(&graph, &mut sol1);
            naive_rule1_impl(&graph, &mut sol2);

            assert!(sol1.equals(&sol2));
        }
    }
}
