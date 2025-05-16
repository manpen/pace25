use std::marker::PhantomData;

use log::info;

use crate::graph::{AdjacencyList, GraphEdgeOrder};

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

    pub fn apply_rule<R: ReductionRule<G>>(
        &mut self,
        graph: &mut G,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
    ) -> bool {
        let before_nodes = graph.vertices_with_neighbors().count();
        let before_edges = graph.number_of_edges();
        let before_in_domset = solution.len();
        let before_covered = covered.cardinality();

        let (changed, post) = R::apply_rule(graph, solution, covered);
        assert!(changed || post.is_none());

        let delta_nodes = before_nodes - graph.vertices_with_neighbors().count();
        let delta_edges = before_edges - graph.number_of_edges();
        let delta_in_domset = solution.len() - before_in_domset;
        let delta_covered = covered.cardinality() - before_covered;

        info!(
            "{} n -= {delta_nodes}, m -= {delta_edges}, |D| += {delta_in_domset}, |covered| += {delta_covered}",
            R::NAME
        );

        if let Some(p) = post {
            self.post_processors.push(p);
        }

        changed
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
