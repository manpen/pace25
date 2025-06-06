use smallvec::SmallVec;

use super::*;
use crate::{
    graph::*,
    utils::{DominatingSet, Marker, NodeMarker},
};

type PairMarker = Marker<(Node, Node), Node>;

pub struct SubsetRuleTwoReduction {
    /// Used for marking neighborhoods of a node to do subset-checks
    node_marker: NodeMarker,
    /// Marker used for Type2-Neighborhood-Check of reference-pair
    pair_marker: PairMarker,
    /// Marker used for Type3-Neighborhood-Check of reference-pair
    pair_marker2: PairMarker,
    /// List of all possible reduction-candidates (u, v, t) where u < v with edges {u,t},{t,v}
    candidates: Vec<(Node, Node, Node)>,
    /// Helper list to keep track of all covered neighbors in a single iteration
    /// AS WELL AS used for computing whether a type3-neighborhood can be covered by a single node
    nbs: Vec<Node>,
    /// type2[u] stores a list of all nodes v that are subset-dominated by u and are thus not
    /// catched in the Rule2-scheme in candidates
    type2: Vec<SmallVec<[Node; 4]>>,
    /// List of all reductions that might be applicable
    #[allow(clippy::type_complexity)]
    possible_reductions: Vec<((Node, Node), SmallVec<[Node; 4]>)>,
    /// Offsets used for faster binary search
    offsets: Vec<usize>,
}

impl SubsetRuleTwoReduction {
    pub fn new(n: NumNodes) -> Self {
        Self {
            node_marker: NodeMarker::new(n, NOT_SET),
            pair_marker: PairMarker::new(n, (NOT_SET, NOT_SET)),
            pair_marker2: PairMarker::new(n, (NOT_SET, NOT_SET)),
            candidates: Vec::new(),
            nbs: Vec::new(),
            type2: vec![Default::default(); n as usize],
            possible_reductions: Vec::new(),
            offsets: Vec::new(),
        }
    }
}

const NOT_SET: Node = Node::MAX;

impl<
    Graph: AdjacencyList + GraphEdgeEditing + AdjacencyTest + NeighborsSlice + ExtractCsrRepr + 'static,
> ReductionRule<Graph> for SubsetRuleTwoReduction
{
    const NAME: &str = "SubsetRuleTwoReduction";

    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        let n = graph.len();
        assert!(NOT_SET as usize >= n);

        self.node_marker.reset();
        self.nbs.clear();

        for u in 0..n {
            self.type2[u].clear();
        }

        'outer: for u in graph.vertices() {
            // Nodes need to have at least degree 2 to be an applicable candidate for Rule2
            if graph.degree_of(u) < 2 || never_select.get_bit(u) {
                continue;
            }

            // Mark closed neighborhood of u
            self.node_marker
                .mark_all_with(graph.closed_neighbors_of(u), u);

            for v in graph.neighbors_of(u) {
                // Let N be the neighborhood of v that is *not* covered by u.
                // Then, we want to push {u,ref}-v to candidates if ref is the *only* uncovered
                // node in N. If there are multiple uncovered nodes in N, we do nothing.
                // If every node in N is covered, we want to push {u,x}-v to candidates for every x
                // in N.
                //
                //
                // ref_node is the uniquely identified uncovered node in N. If multiple or none
                // exist, ref_node = NOT_SET.
                //
                // If multiple exist, we also make sure that covered_neighbors is empty, otherwise
                // covered_neighbors = N.
                let mut ref_node = NOT_SET;
                debug_assert!(self.nbs.is_empty());

                let mut nbs = graph
                    .neighbors_of(v)
                    .filter(|&x| !self.node_marker.is_marked_with(x, u));

                // See if N is empty, ie. every neighbor of v is marked by u
                let first_neighbor = nbs.next();
                if first_neighbor.is_none() {
                    // u and v are twins: break ties by index and covered-state (this rule prefers
                    // uncovered neighbors (with smaller index) as fixed nodes)
                    if graph.degree_of(u) == graph.degree_of(v)
                        && v > u
                        && (!covered.get_bit(v) || covered.get_bit(u))
                    {
                        // We do not need to push u to type2[v] here as it is guaranteed that we
                        // will reach this expression in the v-iteration of 'outer and u will be
                        // pushed to type2[v] then
                        never_select.set_bit(u);
                        continue 'outer;
                    }

                    never_select.set_bit(v);
                    self.type2[u as usize].push(v);
                    continue;
                }

                for x in std::iter::once(first_neighbor.unwrap()).chain(nbs) {
                    if !covered.get_bit(x) {
                        // If we find at least one uncovered node in N, we no longer care for
                        // covered ones regardless of if there is one or multiple such nodes in N.
                        self.nbs.clear();

                        // First uncovered node found
                        if ref_node == NOT_SET {
                            ref_node = x;
                            continue;
                        }

                        // Second uncovered node found
                        ref_node = NOT_SET;
                        break;
                    }

                    // No uncovered node was yet discovered, so save x for later
                    if ref_node == NOT_SET {
                        self.nbs.push(x);
                    }
                }

                // Exactly one uncovered node in N was found
                //
                // In this case, covered_neighbors is guaranteed to be empty for the next iteration
                if ref_node != NOT_SET {
                    self.candidates.push((ref_node.min(u), ref_node.max(u), v));
                    continue;
                }

                // All uncovered neighbors of v are marked by u, ie. u subset-dominates v
                if !self.nbs.is_empty() {
                    // TBD: possibly check if non_perm_degree[u] == non_perm_degree[v] and we
                    // should break ties in favor of v instead?
                    never_select.set_bit(v);
                    self.type2[u as usize].push(v);
                }

                // |covered_neighbors| > 0 iff there was no uncovered node in N found
                //
                // Also makes sure that covered_neighbors is empty in the next iteration
                for x in self.nbs.drain(..) {
                    self.candidates.push((x.min(u), x.max(u), v));
                }
            }
        }

        // Early return if no candidates were found (redundant-markers still could have been applied)
        if self.candidates.is_empty() {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        // All candidates with the same reference-pair are now in succession
        self.candidates.sort_unstable();
        self.candidates.dedup();

        // Current considered reference-pair and its type(2,3)-nodes
        let mut curr_ref_pair = (NOT_SET, NOT_SET);
        let mut curr_type3: SmallVec<[Node; 4]> = Default::default();

        // Merge all type3-nodes of a given reference pair
        for (ref_u, ref_v, t3) in self.candidates.drain(..) {
            if (ref_u, ref_v) == curr_ref_pair {
                curr_type3.push(t3);
            } else {
                if curr_ref_pair != (NOT_SET, NOT_SET) {
                    self.possible_reductions
                        .push((curr_ref_pair, curr_type3.clone()));
                }

                curr_ref_pair = (ref_u, ref_v);
                curr_type3.clear();
                curr_type3.push(t3);
            }
        }

        // Last entry of candidates was not yet processed;
        // one entry in candidates is guaranteed, so curr_ref_pair is no longer (NOT_SET, NOT_SET)
        self.possible_reductions
            .push((curr_ref_pair, curr_type3.clone()));

        // TBD: possible shuffle of reductions in case of conflicts

        self.pair_marker.reset();
        self.pair_marker2.reset();

        // Sort adjacency lists to allow for binary search
        for u in graph.vertices_range() {
            graph.as_neighbors_slice_mut(u).sort_unstable();
        }

        self.nbs.clear();

        let mut modified = false;
        for ((ref_u, ref_v), mut cand) in self.possible_reductions.drain(..) {
            // Nodes marked as redundant can not act as witnesses here
            if never_select.get_bit(ref_u) || never_select.get_bit(ref_v) {
                continue;
            }

            let marker = (ref_u, ref_v);

            // Mark N[ref_u,ref_v]
            self.pair_marker
                .mark_all_with(graph.closed_neighbors_of(ref_u), marker);

            for u in graph.closed_neighbors_of(ref_v) {
                // There are possibly nodes u that are type3-nodes to (ref_u,ref_v) but were not
                // catched by our above scheme as they are neighbored to type2-nodes of both ref_u
                // and ref_v
                if self.pair_marker.is_marked_with(u, marker) {
                    // TBD: find faster method
                    cand.push(u);
                } else {
                    self.pair_marker.mark_with(u, marker);
                }
            }

            // Type2-Nodes were also not catched for this specific pair and need to be checked in
            // their entirety if they are subject to this specific reference-pair
            cand.extend_from_slice(&self.type2[ref_u as usize]);
            cand.extend_from_slice(&self.type2[ref_v as usize]);

            // Remove duplicates
            cand.sort_unstable();
            cand.dedup();

            // Can both reference nodes be fixed or only one?
            let (mut ref_u_fixable, mut ref_v_fixable) = (false, false);

            // Mark Type(2,3)-nodes with marker in marker2
            for &u in &cand {
                if graph
                    .neighbors_of(u)
                    .all(|v| self.pair_marker.is_marked_with(v, marker) || covered.get_bit(v))
                {
                    self.pair_marker2.mark_with(u, marker);
                }
            }

            // Store all candidates for later as this procedure will remove type2-nodes
            let prev_cand = cand.clone();
            for i in (0..cand.len()).rev() {
                let u = cand[i];
                // A type3-node cannot be already covered and all neighbors must either
                // - be a type2-node
                // - be ref_u
                // - be ref_v
                //
                // If not a type3-node, remove it from candidates
                if covered.get_bit(u)
                    || graph.closed_neighbors_of(u).any(|v| {
                        !self.pair_marker2.is_marked_with(v, marker) && v != ref_u && v != ref_v
                    })
                {
                    cand.swap_remove(i);
                }
            }

            // We need at least 2 type3-nodes for Rule2 to be applicable
            if cand.len() <= 1 {
                continue;
            }

            debug_assert!(self.nbs.is_empty());
            debug_assert!(self.offsets.is_empty());

            // Check if there is any Type(2,3)-node that covers all type3-nodes
            //
            // Since cand is the list of all type3-nodes, every such node must be a closed-neighbor
            // of any node in cand (and not ref_u, ref_v)
            let min_node = cand
                .iter()
                .map(|&u| (graph.degree_of(u), u))
                .min()
                .unwrap()
                .1;
            for u in graph.closed_neighbors_of(min_node) {
                // Only nodes with degree at least |cand| could possibly cover cand in its
                // entirety (at least one edge to ref_u,ref_v must exist)
                if graph.degree_of(u) >= cand.len() as NumNodes && u != ref_u && u != ref_v {
                    self.nbs.push(u);
                    self.offsets.push(0);
                }
            }

            cand.sort_unstable();

            for &u in &cand {
                for i in (0..self.nbs.len()).rev() {
                    let nb = self.nbs[i];
                    if nb == u {
                        continue;
                    }

                    if let Ok(index) =
                        graph.as_neighbors_slice(nb)[self.offsets[i]..].binary_search(&u)
                    {
                        // Since edge-lists are sorted, v is increasing and we can use offsets[i] to
                        // allow for faster binary searches in later iterations
                        self.offsets[i] += index;
                    } else {
                        self.nbs.swap_remove(i);
                        self.offsets.swap_remove(i);
                    }
                }
            }

            // If check_nbs is empty, so is offsets
            if !self.nbs.is_empty() {
                self.nbs.clear();
                self.offsets.clear();
                continue;
            }

            // Mark all type(2,3)-nodes as redundant
            never_select.set_bits(
                prev_cand
                    .into_iter()
                    .filter(|&u| self.pair_marker2.is_marked_with(u, marker)),
            );

            // Compute whether ref_u and ref_v are fixable, ie. have a type3-neighbor that only
            // they are connected to (among ref_u, ref_v)
            for t3 in cand {
                let res = graph.has_neighbors(t3, [ref_u, ref_v]);
                if [true, false] == res {
                    ref_u_fixable = true;
                } else if [false, true] == res {
                    ref_v_fixable = true;
                }

                if ref_u_fixable && ref_v_fixable {
                    break;
                }
            }

            // Fix ref_u if possible
            if ref_u_fixable {
                domset.fix_node(ref_u);
                covered.set_bits(graph.closed_neighbors_of(ref_u));
                modified = true;
            }

            // Fix ref_v if possible
            if ref_v_fixable {
                domset.fix_node(ref_v);
                covered.set_bits(graph.closed_neighbors_of(ref_v));
                modified = true;
            }
        }

        (modified, None::<Box<dyn Postprocessor<Graph>>>)
    }
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    use super::*;

    fn base_graph(n: NumNodes) -> (AdjArray, NumNodes) {
        let mut graph = AdjArray::new(5 + n);
        for u in 0..n {
            graph.add_edge(0, u + 5);
            graph.add_edge(1, u + 5);
        }

        // Add Path with length 3 to prevent reductions of 0/1
        graph.add_edge(0, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph.add_edge(4, 1);

        (graph, 5)
    }

    #[test]
    fn trivial_cases() {
        let rng = &mut Pcg64Mcg::seed_from_u64(2025);

        // 2-Paths only
        {
            for n in 2..100 {
                let (mut graph, off) = base_graph(n);

                let mut domset = DominatingSet::new(off + n);
                let mut covered = graph.vertex_bitset_unset();
                let mut redundant = graph.vertex_bitset_unset();

                let mut rule = SubsetRuleTwoReduction::new(off + n);

                let (modified, _) =
                    rule.apply_rule(&mut graph, &mut domset, &mut covered, &mut redundant);

                assert!(!modified);
                assert_eq!(domset.len(), 0);
                assert_eq!(covered.cardinality(), 0);
                assert_eq!(redundant.cardinality(), n);
                assert!((0..n).all(|u| redundant.get_bit(u + off)));
            }
        }

        // Redundant only with in-between edges
        {
            for n in 4..100 {
                let (mut graph, off) = base_graph(n);

                // Insert random edges without creating a node that covers everything
                for _ in 0..(n * n - n / 4) {
                    let u = rng.gen_range(0..n);
                    let v = rng.gen_range(0..n);

                    if u == v
                        || graph.degree_of(off + u) >= n - 1
                        || graph.degree_of(off + v) >= n - 1
                    {
                        continue;
                    }

                    graph.try_add_edge(off + u, off + v);
                }

                let mut domset = DominatingSet::new(off + n);
                let mut covered = graph.vertex_bitset_unset();
                let mut redundant = graph.vertex_bitset_unset();

                let mut rule = SubsetRuleTwoReduction::new(off + n);

                let (modified, _) =
                    rule.apply_rule(&mut graph, &mut domset, &mut covered, &mut redundant);

                assert!(!modified);
                assert_eq!(domset.len(), 0);
                assert_eq!(covered.cardinality(), 0);
                assert_eq!(redundant.cardinality(), n);
                assert!((0..n).all(|u| redundant.get_bit(u + off)));
            }
        }

        // One fixed node
        {
            for n in 3..100 {
                let (mut graph, off) = base_graph(n);

                // Insert random edges without creating a node that covers everything
                for _ in 0..(n * n - n / 4) {
                    let u = rng.gen_range(0..n);
                    let v = rng.gen_range(0..n);

                    if u == v
                        || graph.degree_of(off + u) >= n - 1
                        || graph.degree_of(off + v) >= n - 1
                    {
                        continue;
                    }

                    graph.try_add_edge(off + u, off + v);
                }

                if graph.degree_of(off) == 2 {
                    graph.add_edge(off, off + 1);
                }
                graph.remove_edge(1, off);

                let mut domset = DominatingSet::new(off + n);
                let mut covered = graph.vertex_bitset_unset();
                let mut redundant = graph.vertex_bitset_unset();

                let mut rule = SubsetRuleTwoReduction::new(off + n);

                let (modified, _) =
                    rule.apply_rule(&mut graph, &mut domset, &mut covered, &mut redundant);

                assert!(modified);
                assert_eq!(domset.len(), 1);
                assert!(domset.is_in_domset(0));
                assert_eq!(covered.cardinality(), graph.degree_of(0) + 1);
                assert_eq!(redundant.cardinality(), n);
                assert!((0..n).all(|u| redundant.get_bit(u + off)));
            }
        }

        // Two fixed nodes
        {
            for n in 4..100 {
                let (mut graph, off) = base_graph(n);

                // Insert random edges without creating a node that covers everything
                for _ in 0..(n * n - n / 4) {
                    let u = rng.gen_range(0..n);
                    let v = rng.gen_range(0..n);

                    if u == v
                        || graph.degree_of(off + u) >= n - 1
                        || graph.degree_of(off + v) >= n - 1
                    {
                        continue;
                    }

                    graph.try_add_edge(off + u, off + v);
                }

                graph.try_remove_edge(off, off + 1);

                if graph.degree_of(off) == 2 {
                    graph.add_edge(off, off + 3);
                }
                graph.remove_edge(1, off);

                if graph.degree_of(off + 1) == 2 {
                    graph.add_edge(off + 1, off + 2);
                }
                graph.remove_edge(0, off + 1);

                let mut domset = DominatingSet::new(off + n);
                let mut covered = graph.vertex_bitset_unset();
                let mut redundant = graph.vertex_bitset_unset();

                let mut rule = SubsetRuleTwoReduction::new(off + n);

                let (modified, _) =
                    rule.apply_rule(&mut graph, &mut domset, &mut covered, &mut redundant);

                assert!(modified);
                assert_eq!(domset.len(), 2);
                assert!(domset.is_in_domset(0) && domset.is_in_domset(1));
                assert_eq!(covered.cardinality(), n + off - 1);
                assert_eq!(redundant.cardinality(), n);
                assert!((0..n).all(|u| redundant.get_bit(u + off)));
            }
        }

        // No Reductions - 1
        {
            for n in 3..100 {
                let mut graph = AdjArray::new(2 + n + n);
                for u in 0..n {
                    graph.add_edge(0, 2 + u);
                    graph.add_edge(1, 2 + u);
                    graph.add_edge(2 + u, 2 + u + n);
                    graph.add_edge(2 + u + n, (u + 1) % n + 2 + n);
                }

                let mut domset = DominatingSet::new(2 + n + n);
                let mut covered = graph.vertex_bitset_unset();
                let mut redundant = graph.vertex_bitset_unset();

                let mut rule = SubsetRuleTwoReduction::new(2 + n + n);

                let (modified, _) =
                    rule.apply_rule(&mut graph, &mut domset, &mut covered, &mut redundant);

                assert!(!modified);
                assert_eq!(domset.len(), 0);
                assert_eq!(covered.cardinality(), 0);
                assert_eq!(redundant.cardinality(), 0);
            }
        }

        // No Reductions - 2
        {
            for n in 3..100 {
                let (mut graph, off) = base_graph(n);
                for u in 1..n {
                    graph.add_edge(off, off + u);
                }

                let mut domset = DominatingSet::new(off + n);
                let mut covered = graph.vertex_bitset_unset();
                let mut redundant = graph.vertex_bitset_unset();

                let mut rule = SubsetRuleTwoReduction::new(off + n);

                let (modified, _) =
                    rule.apply_rule(&mut graph, &mut domset, &mut covered, &mut redundant);

                assert!(!modified);
                assert_eq!(domset.len(), 0);
                assert_eq!(covered.cardinality(), 0);
                assert_eq!(redundant.cardinality(), n - 1);
            }
        }
    }
}
