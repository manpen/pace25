use std::marker::PhantomData;

use super::*;
use crate::{graph::*, utils::DominatingSet};

use log::info;
use smallvec::SmallVec;

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
pub struct RuleOneReduction<G> {
    _graph: PhantomData<G>,
}

const NOT_SET: Node = Node::MAX;

impl<Graph: AdjacencyList + GraphEdgeEditing + 'static> ReductionRule<Graph>
    for RuleOneReduction<Graph>
{
    const NAME: &str = "RuleOne";

    fn apply_rule(
        graph: &mut Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        _redundant: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        let n = graph.len();
        assert!(NOT_SET as usize >= n);

        // Inverse mappings of step (1) and (2)
        let mut inv_mappings: Vec<SmallVec<[Node; 4]>> = vec![Default::default(); n];

        // Parent[u] = v if (u,v) is a possible candidate
        let mut parent: Vec<Node> = vec![NOT_SET; n];

        // Used for confirming whether neighborhoods of nodes are subsets of other neighborhoods.
        let mut marked: Vec<Node> = vec![NOT_SET; n];

        // BitSet indicating that a node is a Type(2 or 3)-Candidate (not confirmed yet)
        let mut type2_nodes: BitSet = graph.vertex_bitset_unset();

        // Helper-BitSet to ensure we only process each node once later
        let mut processed: BitSet = graph.vertex_bitset_unset();

        // (1) Compute first mapping and fix possible singletons
        for u in covered.iter_cleared_bits() {
            if graph.degree_of(u) == 0 {
                continue;
            }

            let max_neighbor = graph
                .closed_neighbors_of(u)
                .map(|u| (graph.degree_of(u), u))
                .max()
                .map(|(_, u)| u)
                .unwrap();

            if max_neighbor != u {
                inv_mappings[max_neighbor as usize].push(u);
            }
        }

        // (1) Compute list of candidate-pairs based on mapping
        for u in graph.vertices() {
            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            for v in graph.closed_neighbors_of(u) {
                marked[v as usize] = u;
            }

            // Check whether N[v] is a subset of N[u]
            for v in inv_mappings[u as usize].drain(..) {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| marked[x as usize] == u)
                {
                    parent[v as usize] = u;
                    type2_nodes.set_bit(v);
                }
            }
        }

        // We drained inv_mappings earlier completely, so we can now reuse it
        debug_assert!(inv_mappings.iter().all(|vec| vec.is_empty()));

        // (2) Compute second mapping from list of candidate-pairs
        for u in type2_nodes.iter_set_bits() {
            for v in graph.closed_neighbors_of(u) {
                // Only process each node once
                if processed.set_bit(v) {
                    continue;
                }

                // Mark closed neighborhood N[v] of v (SelfLoop marker)
                for x in graph.closed_neighbors_of(v) {
                    marked[x as usize] = v;
                }

                // Find minimum dominating node of neighbors in neighborhood of v
                if let Some((_, min_node)) = graph
                    .closed_neighbors_of(v)
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
            }
        }

        parent = vec![NOT_SET; n];
        let mut removable_nodes = graph.vertex_bitset_unset();

        let mut selected = Vec::with_capacity(128);
        // (3) Mark candidates as possible Type2-Nodes if their neighborhoods are subsets
        for u in graph.vertices() {
            if inv_mappings[u as usize].is_empty() {
                continue;
            }

            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            for v in graph.closed_neighbors_of(u) {
                marked[v as usize] = u;
            }

            for &v in &inv_mappings[u as usize] {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| marked[x as usize] == u)
                {
                    parent[v as usize] = u;
                }
            }

            for v in inv_mappings[u as usize].drain(..) {
                if covered.get_bit(v) {
                    continue;
                }
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| parent[x as usize] == u || x == u)
                {
                    domset.fix_node(u);
                    selected.push(u);
                    covered.set_bits(graph.closed_neighbors_of(u));
                    break;
                }
            }
        }

        processed.clear_all();
        for &u in &selected {
            processed.set_bits(graph.closed_neighbors_of(u));
        }

        for u in graph.vertices() {
            if processed.get_bit(u)
                && !domset.is_in_domset(u)
                && graph
                    .neighbors_of(u)
                    .filter(|x| !processed.get_bit(*x))
                    .count()
                    <= 1
            {
                removable_nodes.set_bit(u);
            }
        }

        let mut modified = !selected.is_empty() || removable_nodes.cardinality() > 0;

        assert!(removable_nodes.cardinality() >= selected.len() as NumNodes);
        info!(
            "RuleOne removeable nodes: {}",
            removable_nodes.cardinality()
        );
        debug_assert!(removable_nodes.iter_set_bits().all(|u| covered.get_bit(u)));

        for u in removable_nodes.iter_set_bits() {
            graph.remove_edges_at_node(u);
        }

        let mut neighbors_to_remove = Vec::new();
        processed -= &removable_nodes;

        for u in processed.iter_set_bits() {
            // TODO: We want to have a drain_neighbors function in graph!
            neighbors_to_remove.extend(graph.neighbors_of(u).filter(|&v| processed.get_bit(v)));
            if neighbors_to_remove.is_empty() {
                continue;
            }

            for &v in &neighbors_to_remove {
                graph.remove_edge(u, v);
            }
            modified = true;
            neighbors_to_remove.clear();
        }

        for u in selected {
            graph.remove_edges_at_node(u);
        }

        covered.update_cleared_bits(|u| {
            let is_singleton = graph.degree_of(u) == 0;
            if is_singleton {
                domset.fix_node(u);
            }
            is_singleton
        });

        (modified, None::<Box<dyn Postprocessor<Graph>>>)
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    // The standard Rule implementation with runtime O(n * D * D) where D is the maximum degree in the graph.
    fn naive_rule1_impl(
        graph: &(impl AdjacencyList + SelfLoop),
        sol: &mut DominatingSet,
    ) -> BitSet {
        let mut marked = BitSet::new(graph.number_of_nodes());
        let mut type2_nodes = BitSet::new(graph.number_of_nodes());

        let mut redundant = BitSet::new(graph.number_of_nodes());
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
                redundant.set_bits(type2_nodes.iter_set_bits());
            }
        }

        redundant
    }

    fn get_random_graph(rng: &mut impl Rng, n: NumNodes, m: NumEdges) -> (AdjArray, CsrGraph) {
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

        (
            AdjArray::from_edges(n, &edges),
            CsrGraph::from_edges(n, &edges),
        )
    }

    #[test]
    fn compare_rule1_implementations() {
        let rng = &mut rand::thread_rng();

        for _ in 0..1000 {
            let n = rng.gen_range(5..50);
            let m = rng.gen_range(1..(n * (n - 1) / 4)) as NumEdges;

            let (mut adj_graph, csr_graph) = get_random_graph(rng, n, m);

            let mut sol1 = DominatingSet::new(n);
            let mut sol2 = DominatingSet::new(n);

            {
                let mut covered = adj_graph.vertex_bitset_unset();

                // Rule One does not fix singleton nodes anymore
                for u in adj_graph.vertices() {
                    if adj_graph.degree_of(u) == 0 {
                        sol1.fix_node(u);
                        covered.set_bit(u);
                    }
                }

                let mut red = adj_graph.vertex_bitset_unset();
                let _ =
                    RuleOneReduction::apply_rule(&mut adj_graph, &mut sol1, &mut covered, &mut red);
            }
            naive_rule1_impl(&csr_graph, &mut sol2);

            assert!(
                sol2.iter().all(|u| sol1.is_in_domset(u)),
                "Test: {sol1:?}\nRef:  {sol2:?}"
            );
        }
    }
}
