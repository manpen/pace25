use super::*;
use crate::graph::{
    AdjacencyList, GraphEdgeEditing, GraphEdgeOrder, NumEdges, NumNodes, UnsafeGraphEditing,
};
use std::{
    cmp::Reverse,
    collections::HashMap,
    marker::PhantomData,
    time::{Duration, Instant},
};

use itertools::Itertools;
use log::info;

pub struct Reducer<G> {
    post_processors: Vec<Box<dyn Postprocessor<G>>>,
    time_per_rule: HashMap<&'static str, Duration>,
    _graph: PhantomData<G>,
}

impl<G> Default for Reducer<G> {
    fn default() -> Self {
        Self {
            post_processors: Default::default(),
            time_per_rule: Default::default(),
            _graph: Default::default(),
        }
    }
}

impl<G: GraphEdgeOrder + AdjacencyList + GraphEdgeEditing + UnsafeGraphEditing> Reducer<G> {
    pub fn new() -> Self {
        Default::default()
    }

    /// Apply the rule `R` once and print out some statistics.
    /// Returns `true` iff the rule reported that it applied some change
    pub fn apply_rule<R: ReductionRule<G>>(
        &mut self,
        rule: &mut R,
        graph: &mut G,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> bool {
        let before_nodes = graph.vertices_with_neighbors().count();
        let before_edges = graph.number_of_edges();
        let before_in_domset = domset.len();
        let before_covered = covered.cardinality();
        let before_redundant = never_select.cardinality() as i32;

        debug_assert!(domset.iter().all(|u| graph.degree_of(u) == 0));

        let start_rule = Instant::now();
        let (mut changed, post) = rule.apply_rule(graph, domset, covered, never_select);
        assert!(changed || post.is_none());

        let start_clean = Instant::now();
        let duration_rule = start_clean.duration_since(start_rule);
        let unneccesary_edges = if changed || R::REMOVAL {
            let removed_edges = self.remove_unnecessary_edges(graph, domset, covered, never_select);
            changed |= removed_edges > 0;
            removed_edges
        } else {
            0
        };
        let duration_clean = start_clean.elapsed();

        debug_assert!(domset.iter().all(|u| graph.degree_of(u) == 0));

        let delta_nodes = before_nodes - graph.vertices_with_neighbors().count();
        let delta_edges = before_edges - graph.number_of_edges();
        let delta_in_domset = domset.len() - before_in_domset;
        let delta_covered = covered.cardinality() - before_covered;
        let delta_redundant = never_select.cardinality() as i32 - before_redundant;

        if changed {
            info!(
                "{:25} n -= {delta_nodes:6}, m -= {delta_edges:6}, |D| += {delta_in_domset:7}, |covered| += {delta_covered:7}, |redundant| += {delta_redundant:7}, |edges| -= {unneccesary_edges:6}, time: {:5}ms + {:3}ms",
                R::NAME,
                duration_rule.as_millis(),
                duration_clean.as_millis()
            );
        }

        if let Some(p) = post {
            self.post_processors.push(p);
        }

        self.time_per_rule
            .entry(R::NAME)
            .and_modify(|x| *x += duration_rule)
            .or_insert(duration_rule);

        self.time_per_rule
            .entry("RemoveUnnecessaryEdges")
            .and_modify(|x| *x += duration_clean)
            .or_insert(duration_clean);

        changed
    }

    /// Apply the rule `R` until it reports that no more change is possible.
    /// Returns the number of applications of the rule
    pub fn apply_rule_exhaustively<R: ReductionRule<G>>(
        &mut self,
        rule: &mut R,
        graph: &mut G,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> NumNodes {
        let mut iters = 1;

        while self.apply_rule(rule, graph, domset, covered, never_select) {
            iters += 1;
        }

        iters
    }

    pub fn remove_unnecessary_edges(
        &self,
        graph: &mut G,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> NumEdges {
        let mut delete_node = covered.clone();
        delete_node &= never_select;

        let mut half_edges_removed = 0;

        for u in graph.vertices_range() {
            if delete_node.get_bit(u) {
                half_edges_removed += unsafe { graph.remove_half_edges_at(u) };
                continue;
            }

            half_edges_removed += match (covered.get_bit(u), never_select.get_bit(u)) {
                (true, false) => unsafe {
                    graph.remove_half_edges_at_if(u, |v| {
                        covered.get_bit(v) || delete_node.get_bit(v)
                    })
                },
                (false, true) => unsafe {
                    graph.remove_half_edges_at_if(u, |v| {
                        never_select.get_bit(v) || delete_node.get_bit(v)
                    })
                },
                (false, false) => unsafe {
                    graph.remove_half_edges_at_if(u, |v| delete_node.get_bit(v))
                },
                (true, true) => unreachable!("This node is contained in delete_node"),
            };
        }

        assert!(half_edges_removed % 2 == 0);
        unsafe {
            graph.set_number_of_edges(graph.number_of_edges() - half_edges_removed / 2);
        }

        // Delete edges between nodes (u,v) where u is covered and v is the *only* uncovered neighbor of u
        //
        // Rest of deletions are done in post-processing
        for u in covered.iter_set_bits().filter(|&u| !domset.is_in_domset(u)) {
            let mut nbs = graph.neighbors_of(u).filter(|x| !covered.get_bit(*x));

            let nb1 = nbs.next();
            let nb2 = nbs.next();

            // Iterator no longer needed; potentially save time by not consuming fully
            std::mem::drop(nbs);

            if let (Some(v), None) = (nb1, nb2) {
                graph.remove_edge(u, v);
                half_edges_removed += 2;
            }
        }

        covered.update_cleared_bits(|u| {
            let is_singleton = graph.degree_of(u) == 0;
            if is_singleton {
                // If redundant[u] = 1, then u was dominated by another node v that was removed in
                // this iteration along with every other neighbor of u because u was the only
                // uncovered neighbor of those nodes.
                //
                // It would be equally optimal to put v into the dominating set instead, but at
                // this point, it does not matter.
                domset.fix_node(u);
            }
            is_singleton
        });

        half_edges_removed / 2
    }

    pub fn report_summary(&self) {
        let mut items: Vec<(&str, Duration)> = self
            .time_per_rule
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect_vec();

        items.sort_by_key(|(k, v)| (Reverse(*v), *k));

        info!("Preprocessing completed");
        for (rule, time) in items {
            info!("  |- Rule {rule:30} took {:7}ms", time.as_millis());
        }
    }

    pub fn post_process(
        &mut self,
        graph: &mut G,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) {
        while let Some(mut p) = self.post_processors.pop() {
            p.post_process(graph, solution, covered, never_select);
        }
    }
}
