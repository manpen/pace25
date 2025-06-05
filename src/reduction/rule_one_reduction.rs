use super::*;
use crate::{
    graph::*,
    utils::{DominatingSet, NodeMarker},
};

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
pub struct RuleOneReduction {
    /// Inverse mappings of step (1) and (2)
    inv_mappings: Vec<SmallVec<[Node; 4]>>,
    /// Used for confirming whether neighborhoods of nodes are subsets of other neighborhoods.
    marked: NodeMarker,
    /// Parent[u] = v if (u,v) is a possible candidate
    parent: NodeMarker,
    /// BitSet indicating that a node is a Type(2 or 3)-Candidate (not confirmed yet)
    type2_nodes: BitSet,
    /// Helper-BitSet to ensure we only process each node once later
    processed: BitSet,
    /// Number of uncovered nodes in closed neighborhood
    non_perm_degree: Vec<NumNodes>,
    /// List of nodes with at least one Type3-Neighbor
    selected: Vec<Node>,
}

impl RuleOneReduction {
    pub fn new(n: NumNodes) -> Self {
        Self {
            inv_mappings: vec![Default::default(); n as usize],
            marked: NodeMarker::new(n, NOT_SET),
            parent: NodeMarker::new(n, NOT_SET),
            type2_nodes: BitSet::new(n),
            processed: BitSet::new(n),
            non_perm_degree: vec![NOT_SET; n as usize],
            selected: Vec::with_capacity(n as usize),
        }
    }
}

const NOT_SET: Node = Node::MAX;

impl<Graph: AdjacencyList + GraphEdgeEditing + 'static> ReductionRule<Graph> for RuleOneReduction {
    const NAME: &str = "RuleOne";

    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        let n = graph.len();
        assert!(NOT_SET as usize >= n);

        // Reset variables
        self.marked.reset();
        self.parent.reset();
        self.type2_nodes.clear_all();
        self.processed.clear_all();
        self.selected.clear();

        // Compute permanently covered nodes and degrees
        for u in 0..graph.number_of_nodes() {
            self.non_perm_degree[u as usize] = graph.degree_of(u) + 1;
        }

        for u in covered.iter_set_bits() {
            for v in graph.closed_neighbors_of(u) {
                self.non_perm_degree[v as usize] -= 1;
            }
        }

        // (1) Compute first mapping and fix possible singletons
        for u in graph.vertices() {
            if graph.degree_of(u) == 0 {
                continue;
            }

            let max_neighbor = graph
                .closed_neighbors_of(u)
                .map(|u| (self.non_perm_degree[u as usize], u))
                .max()
                .map(|(_, u)| u)
                .unwrap();

            if max_neighbor != u {
                self.inv_mappings[max_neighbor as usize].push(u);
            }
        }

        // (1) Compute list of candidate-pairs based on mapping
        for u in graph.vertices() {
            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            self.marked.mark_all_with(graph.closed_neighbors_of(u), u);

            // Check whether N[v] is a subset of N[u]
            for v in self.inv_mappings[u as usize].drain(..) {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| self.marked.is_marked_with(x, u) || covered.get_bit(x))
                {
                    self.parent.mark_with(v, u);
                    self.type2_nodes.set_bit(v);
                }
            }
        }

        // We drained inv_mappings earlier completely, so we can now reuse it
        debug_assert!(self.inv_mappings.iter().all(|vec| vec.is_empty()));

        // (2) Compute second mapping from list of candidate-pairs
        for u in self.type2_nodes.iter_set_bits() {
            for v in graph.closed_neighbors_of(u) {
                // Only process each node once
                if self.processed.set_bit(v) {
                    continue;
                }

                // Mark closed neighborhood N[v] of v (SelfLoop marker)
                self.marked.mark_all_with(graph.closed_neighbors_of(v), v);

                // Find minimum dominating node of neighbors in neighborhood of v
                if let Some((_, min_node)) = graph
                    .closed_neighbors_of(v)
                    .filter_map(|x| {
                        let pt = self.parent.get_mark(x);
                        (pt != NOT_SET && pt != v && self.marked.is_marked_with(pt, v))
                            .then(|| (self.non_perm_degree[pt as usize], pt))
                    })
                    .min()
                    && !redundant.get_bit(min_node)
                {
                    // We drained inv_mappings earlier completely, so we can now reuse it
                    self.inv_mappings[min_node as usize].push(v);
                }
            }
        }

        self.parent.reset();

        // (3) Mark candidates as possible Type2-Nodes if their neighborhoods are subsets
        for u in graph.vertices() {
            if self.inv_mappings[u as usize].is_empty() {
                continue;
            }

            // Mark closed neighborhood N[u] of u (SelfLoop marker)
            self.marked.mark_all_with(graph.closed_neighbors_of(u), u);

            for &v in &self.inv_mappings[u as usize] {
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| self.marked.is_marked_with(x, u) || covered.get_bit(x))
                {
                    self.parent.mark_with(v, u);
                }
            }

            for v in self.inv_mappings[u as usize].drain(..) {
                if covered.get_bit(v) {
                    continue;
                }
                if graph
                    .closed_neighbors_of(v)
                    .all(|x| self.parent.is_marked_with(x, u) || x == u)
                {
                    assert!(!redundant.get_bit(u));
                    domset.fix_node(u);
                    self.selected.push(u);
                    covered.set_bits(graph.closed_neighbors_of(u));
                    break;
                }
            }
        }

        (
            !self.selected.is_empty(),
            None::<Box<dyn Postprocessor<Graph>>>,
        )
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
                let mut rule1 = RuleOneReduction::new(n);
                let _ = rule1.apply_rule(&mut adj_graph, &mut sol1, &mut covered, &mut red);
            }
            naive_rule1_impl(&csr_graph, &mut sol2);

            assert!(
                sol2.iter().all(|u| sol1.is_in_domset(u)),
                "Test: {sol1:?}\nRef:  {sol2:?}"
            );
        }
    }
}
