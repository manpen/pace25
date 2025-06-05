use std::marker::PhantomData;

use itertools::Itertools;
use smallvec::SmallVec;

use super::*;
use crate::{graph::*, utils::DominatingSet};

pub struct SubsetRuleTwoReduction<G> {
    _graph: PhantomData<G>,
}

const NOT_SET: Node = Node::MAX;

impl<Graph: AdjacencyList + GraphEdgeEditing + AdjacencyTest + std::fmt::Debug + 'static>
    ReductionRule<Graph> for SubsetRuleTwoReduction<Graph>
{
    const NAME: &str = "SubsetRuleTwoReduction";

    fn apply_rule(
        graph: &mut Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        let n = graph.len();
        assert!(NOT_SET as usize >= n);

        let mut marked = vec![(NOT_SET, NOT_SET); n];
        let mut marked2 = vec![(NOT_SET, NOT_SET); n];
        let mut marked3 = vec![NOT_SET; n];

        let mut candidates: Vec<(Node, Node, Node)> = Vec::new();
        let mut covered_neighbors: Vec<Node> = Vec::new();

        let mut type2: Vec<SmallVec<[Node; 4]>> = vec![Default::default(); n];

        redundant.clear_all();

        'outer: for u in graph.vertices() {
            if graph.degree_of(u) < 2 || redundant.get_bit(u) {
                continue;
            }

            for v in graph.closed_neighbors_of(u) {
                marked3[v as usize] = u;
            }

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
                debug_assert!(covered_neighbors.is_empty());

                let mut nbs = graph.neighbors_of(v).filter(|&x| marked3[x as usize] != u);

                // See if N is empty, ie. every neighbor of v is marked by u
                let first_neighbor = nbs.next();
                if first_neighbor.is_none() {
                    // u and v are twins: break ties by index and covered-state (this rule prefers
                    // uncovered neighbors (with smaller index) as fixed nodes)
                    if graph.degree_of(u) == graph.degree_of(v)
                        && v > u
                        && (!covered.get_bit(v) || covered.get_bit(u))
                    {
                        redundant.set_bit(u);
                        continue 'outer;
                    }

                    redundant.set_bit(v);
                    type2[u as usize].push(v);
                    continue;
                }

                for x in std::iter::once(first_neighbor.unwrap()).chain(nbs) {
                    if !covered.get_bit(x) {
                        // If we find at least one uncovered node in N, we no longer care for
                        // covered ones regardless of if there is one or multiple such nodes in N.
                        covered_neighbors.clear();

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
                        covered_neighbors.push(x);
                    }
                }

                // Exactly one uncovered node in N was found
                //
                // In this case, covered_neighbors is guaranteed to be empty for the next iteration
                if ref_node != NOT_SET {
                    candidates.push((ref_node.min(u), ref_node.max(u), v));
                    continue;
                }

                // All uncovered neighbors of v are marked by u, ie. u subset-dominates v
                if !covered_neighbors.is_empty() {
                    // TBD: possibly check if non_perm_degree[u] == non_perm_degree[v] and we
                    // should break ties in favor of v instead?
                    redundant.set_bit(v);
                    type2[u as usize].push(v);
                }

                // |covered_neighbors| > 0 iff there was no uncovered node in N found
                //
                // Also makes sure that covered_neighbors is empty in the next iteration
                for x in covered_neighbors.drain(..) {
                    candidates.push((x.min(u), x.max(u), v));
                }
            }
        }

        if candidates.is_empty() {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        candidates.sort_unstable();
        candidates.dedup();

        // Current considered reference-pair and its type(2,3)-nodes
        let mut curr_ref_pair = (NOT_SET, NOT_SET);
        let mut curr_type3: SmallVec<[Node; 4]> = Default::default();

        // List of all reductions that are not trivially false
        let mut possible_reductions = Vec::new();

        // Merge all type3-nodes of a given reference pair
        for (ref_u, ref_v, t3) in candidates {
            if (ref_u, ref_v) == curr_ref_pair {
                curr_type3.push(t3);
            } else {
                if curr_ref_pair != (NOT_SET, NOT_SET) {
                    possible_reductions.push((curr_ref_pair, curr_type3.clone()));
                }

                curr_ref_pair = (ref_u, ref_v);
                curr_type3.clear();
                curr_type3.push(t3);
            }
        }

        possible_reductions.push((curr_ref_pair, curr_type3.clone()));

        // TBD: possible shuffle of reductions in case of conflicts

        let mut modified = false;
        for ((ref_u, ref_v), mut cand) in possible_reductions {
            // Nodes marked as redundant can not act as witnesses here
            if redundant.get_bit(ref_u) || redundant.get_bit(ref_v) {
                continue;
            }

            // Every node is witness to at most one node-pair and thus node-pairs can get uniquely
            // identified by one of their candidates
            //
            // |cand| > 1 is guaranteed
            let marker = (ref_u, ref_v);

            for u in graph.closed_neighbors_of(ref_u) {
                marked[u as usize] = marker;
            }
            for u in graph.closed_neighbors_of(ref_v) {
                if marked[u as usize] == marker {
                    // TBD: find faster method
                    cand.push(u);
                } else {
                    marked[u as usize] = marker;
                }
            }

            cand.extend_from_slice(&type2[ref_u as usize]);
            cand.extend_from_slice(&type2[ref_v as usize]);

            cand.sort_unstable();
            cand.dedup();

            // Can both reference nodes be fixed or only one?
            let (mut ref_u_fixable, mut ref_v_fixable) = (false, false);

            for &u in &cand {
                if graph
                    .neighbors_of(u)
                    .all(|v| marked[v as usize] == marker || covered.get_bit(v))
                {
                    marked2[u as usize] = marker;
                }
            }

            let prev_cand = cand.clone();
            for i in (0..cand.len()).rev() {
                let u = cand[i];
                if covered.get_bit(u)
                    || graph.closed_neighbors_of(u).any(|v| {
                        marked2[v as usize] != marker
                            && v != ref_u
                            && v != ref_v
                            && !redundant.get_bit(v)
                    })
                {
                    cand.swap_remove(i);
                }
            }

            if cand.len() <= 1 {
                continue;
            }

            let mut check_nbs = graph
                .closed_neighbors_of(cand[0])
                .filter(|&u| {
                    graph.degree_of(u) + 1 >= cand.len() as NumNodes && u != ref_u && u != ref_v
                })
                .collect_vec();

            // SLOW: consider sorting as in SubSetRule
            for &u in &cand[1..] {
                if covered.get_bit(u) {
                    continue;
                }
                for i in (0..check_nbs.len()).rev() {
                    if check_nbs[i] != u && !graph.has_edge(u, check_nbs[i]) {
                        check_nbs.swap_remove(i);
                    }
                }
            }

            if !check_nbs.is_empty() {
                continue;
            }

            // Nodes are guaranteed to be redundant
            redundant.set_bits(
                prev_cand
                    .into_iter()
                    .filter(|&u| marked2[u as usize] == marker),
            );

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

            if ref_u_fixable {
                domset.fix_node(ref_u);
                covered.set_bits(graph.closed_neighbors_of(ref_u));
                modified = true;
            }

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

                let (modified, _) = SubsetRuleTwoReduction::apply_rule(
                    &mut graph,
                    &mut domset,
                    &mut covered,
                    &mut redundant,
                );

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

                let (modified, _) = SubsetRuleTwoReduction::apply_rule(
                    &mut graph,
                    &mut domset,
                    &mut covered,
                    &mut redundant,
                );

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

                let (modified, _) = SubsetRuleTwoReduction::apply_rule(
                    &mut graph,
                    &mut domset,
                    &mut covered,
                    &mut redundant,
                );

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

                let (modified, _) = SubsetRuleTwoReduction::apply_rule(
                    &mut graph,
                    &mut domset,
                    &mut covered,
                    &mut redundant,
                );

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

                let (modified, _) = SubsetRuleTwoReduction::apply_rule(
                    &mut graph,
                    &mut domset,
                    &mut covered,
                    &mut redundant,
                );

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

                let (modified, _) = SubsetRuleTwoReduction::apply_rule(
                    &mut graph,
                    &mut domset,
                    &mut covered,
                    &mut redundant,
                );

                assert!(!modified);
                assert_eq!(domset.len(), 0);
                assert_eq!(covered.cardinality(), 0);
                assert_eq!(redundant.cardinality(), n - 1);
            }
        }
    }
}
