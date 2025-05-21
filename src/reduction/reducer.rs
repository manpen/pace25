use std::marker::PhantomData;

use log::info;

use crate::graph::{AdjacencyList, GraphEdgeOrder, NumNodes};

use super::*;

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

impl<G: GraphEdgeOrder + AdjacencyList> Reducer<G> {
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

        let (changed, post) = R::apply_rule(graph, solution, covered, redundant);
        assert!(changed || post.is_none());
        debug_assert!(solution.iter().all(|u| graph.degree_of(u) == 0));

        let delta_nodes = before_nodes - graph.vertices_with_neighbors().count();
        let delta_edges = before_edges - graph.number_of_edges();
        let delta_in_domset = solution.len() - before_in_domset;
        let delta_covered = covered.cardinality() - before_covered;
        let delta_redundant = redundant.cardinality() as i32 - before_redundant;

        info!(
            "{} n -= {delta_nodes}, m -= {delta_edges}, |D| += {delta_in_domset}, |covered| += {delta_covered}, |redundant| += {delta_redundant}",
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

    pub fn post_process(
        &mut self,
        graph: &mut G,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
    ) {
        while let Some(mut p) = self.post_processors.pop() {
            p.post_process(graph, solution, covered);
        }
    }
}
