use std::marker::PhantomData;

use super::*;
use crate::{graph::*, utils::DominatingSet};

use smallvec::SmallVec;

pub struct TwoHopRuleTwoReduction<G> {
    _graph: PhantomData<G>,
}

const NOT_SET: Node = Node::MAX;

impl<Graph: AdjacencyList + GraphEdgeEditing + AdjacencyTest + 'static> ReductionRule<Graph>
    for TwoHopRuleTwoReduction<Graph>
{
    const NAME: &str = "TwoHopRuleTwo";

    fn apply_rule(
        graph: &mut Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        let n = graph.len();
        assert!(NOT_SET as usize >= n);

        let mut neighbors = graph.vertex_bitset_unset();

        let mut candidates: Vec<(Node, Node, Node)> = Vec::new();

        for u in graph
            .vertices()
            .filter(|&u| graph.degree_of(u) > 1 && redundant.get_bit(u))
        {
            neighbors.clear_all();
            neighbors.set_bits(graph.closed_neighbors_of(u));

            for v in graph.neighbors_of(u) {
                let mut nbs = graph.neighbors_of(v).filter(|&x| !neighbors.get_bit(x));

                let nb1 = nbs.next();
                let nb2 = nbs.next();

                if let (Some(cand), None) = (nb1, nb2) {
                    candidates.push((u.min(cand), u.max(cand), v));
                }
            }
        }

        candidates.sort_unstable();

        let mut curr_pair = (Node::MAX, Node::MAX);
        let mut curr_type3: SmallVec<[Node; 4]> = Default::default();

        let mut possible_reductions = Vec::new();

        for (ref_u, ref_v, t3) in candidates {
            if (ref_u, ref_v) == curr_pair {
                curr_type3.push(t3);
            } else {
                if curr_type3.len() > 1 {
                    possible_reductions.push((curr_pair, curr_type3.clone()));
                }

                curr_pair = (ref_u, ref_v);
                curr_type3.clear();
                curr_type3.push(t3);
            }
        }

        if curr_type3.len() > 1 {
            possible_reductions.push((curr_pair, curr_type3.clone()));
        }

        let mut modified = false;

        for ((ref_u, ref_v), type3) in possible_reductions {
            // Check if any node in type3 covers every other node in type3
            // Since every node in type3 is guaranteed to be only connected to ref_u,ref_v,type3,
            // this can easily done via a degree-check
            if redundant.get_bit(ref_u)
                || redundant.get_bit(ref_v)
                || type3
                    .iter()
                    .any(|&u| graph.degree_of(u) == (type3.len() as NumNodes - 3))
            {
                continue;
            }

            modified = true;

            // Can both reference nodes be fixed or only one?
            let (mut ref_u_fixable, mut ref_v_fixable) = (false, false);
            for &t3 in &type3 {
                match graph.has_neighbors(t3, [ref_u, ref_v]) {
                    [true, false] => {
                        ref_u_fixable = true;
                    }
                    [false, true] => {
                        ref_v_fixable = true;
                    }
                    _ => {}
                };

                if ref_u_fixable && ref_v_fixable {
                    break;
                }
            }

            if ref_u_fixable {
                domset.fix_node(ref_u);
                covered.set_bits(graph.closed_neighbors_of(ref_u));
            }

            if ref_v_fixable {
                domset.fix_node(ref_v);
                covered.set_bits(graph.closed_neighbors_of(ref_v));
            }

            // Only one node was fixed and edges to the other must be removed
            if ref_u_fixable != ref_v_fixable {
                let not_fixed = if ref_u_fixable { ref_v } else { ref_u };

                for &t3 in &type3 {
                    graph.remove_edge(t3, not_fixed);
                }

                if graph.degree_of(not_fixed) == 0 {
                    domset.fix_node(not_fixed);
                    covered.set_bit(not_fixed);
                }
            }

            // If at least one node was fixed, all nodes in type3 can be removed,
            // otherwise, remove all but 3 nodes
            if !ref_u_fixable && !ref_v_fixable {
                redundant.set_bits(type3.iter().copied());
                covered.set_bits(type3.iter().copied().skip(3));

                // TBD: verify that this is correct
                type3.into_iter().skip(3).for_each(|u| {
                    graph.remove_edges_at_node(u);
                });
            }
        }

        (modified, None::<Box<dyn Postprocessor<Graph>>>)
    }
}
