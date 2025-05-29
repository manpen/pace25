use super::*;
use crate::graph::{AdjacencyList, GraphEdgeOrder, NumNodes, UnsafeGraphEditing};

use log::info;
use std::marker::PhantomData;

pub struct Reducer<G> {
    post_processors: Vec<Box<dyn Postprocessor<G>>>,
    _graph: PhantomData<G>,
}

impl<G> Default for Reducer<G> {
    fn default() -> Self {
        Self {
            post_processors: Default::default(),
            _graph: Default::default(),
        }
    }
}

impl<G: GraphEdgeOrder + AdjacencyList + UnsafeGraphEditing> Reducer<G> {
    pub fn new() -> Self {
        Default::default()
    }

    /// Apply the rule `R` once and print out some statistics.
    /// Returns `true` iff the rule reported that it applied some change
    pub fn apply_rule<R: ReductionRule<G>>(
        &mut self,
        graph: &mut G,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> bool {
        let before_nodes = graph.vertices_with_neighbors().count();
        let before_edges = graph.number_of_edges();
        let before_in_domset = solution.len();
        let before_covered = covered.cardinality();
        let before_redundant = redundant.cardinality() as i32;

        debug_assert!(solution.iter().all(|u| graph.degree_of(u) == 0));

        let (mut changed, post) = R::apply_rule(graph, solution, covered, redundant);
        assert!(changed || post.is_none());
        changed |= self.remove_unnecessary_edges(graph, covered, redundant);
        debug_assert!(solution.iter().all(|u| graph.degree_of(u) == 0));
        debug_assert!(solution.iter().all(|u| !redundant.get_bit(u)));

        let delta_nodes = before_nodes - graph.vertices_with_neighbors().count();
        let delta_edges = before_edges - graph.number_of_edges();
        let delta_in_domset = solution.len() - before_in_domset;
        let delta_covered = covered.cardinality() - before_covered;
        let delta_redundant = redundant.cardinality() as i32 - before_redundant;

        info!(
            "{} n -= {delta_nodes}, m -= {delta_edges}, |D| += {delta_in_domset}, |covered| += {delta_covered}, |redundant| += {delta_redundant}, changed={changed}",
            R::NAME
        );

        if let Some(p) = post {
            self.post_processors.push(p);
        }

        changed
    }

    /// Apply the rule `R` until it reports that no more change is possible.
    /// Returns the number of applications of the rule
    pub fn apply_rule_exhaustively<R: ReductionRule<G>>(
        &mut self,
        graph: &mut G,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> NumNodes {
        let mut iters = 1;

        while self.apply_rule::<R>(graph, solution, covered, redundant) {
            iters += 1;
        }

        info!("{} applied exhaustively {iters} times", R::NAME);
        iters
    }

    pub fn remove_unnecessary_edges(
        &self,
        graph: &mut G,
        covered: &BitSet,
        redundant: &BitSet,
    ) -> bool {
        let mut delete_node = covered.clone();
        delete_node &= redundant;

        let mut half_edges_removed = 0;

        for u in graph.vertices_range() {
            if delete_node.get_bit(u) {
                half_edges_removed += unsafe { graph.remove_half_edges_at(u) };
                continue;
            }

            half_edges_removed += match (covered.get_bit(u), redundant.get_bit(u)) {
                (true, false) => unsafe {
                    graph.remove_half_edges_at_if(u, |v| {
                        covered.get_bit(v) || delete_node.get_bit(v)
                    })
                },
                (false, true) => unsafe {
                    graph.remove_half_edges_at_if(u, |v| {
                        redundant.get_bit(v) || delete_node.get_bit(v)
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

        info!(
            "Removed {} cov-cov or red-red edges",
            half_edges_removed / 2
        );

        half_edges_removed > 0
    }

    pub fn post_process(
        &mut self,
        graph: &mut G,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) {
        while let Some(mut p) = self.post_processors.pop() {
            p.post_process(graph, solution, covered, redundant);
        }
    }
}
