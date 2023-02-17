use crate::{
    graph::{
        twin_width_sat_encoding::TwinWidthSatEncoding, AdjacencyList, ColoredAdjacencyList,
        ColoredAdjacencyTest, GraphEdgeEditing, GraphEdgeOrder,
    },
    heuristic::monte_carlo_search_tree::timeout_monte_carlo_search_tree_solver_preprocessed,
};
use std::fmt::Debug;

use super::contraction_sequence::ContractionSequence;

pub struct TwoStageSatSolver<G> {
    graph: G,
    preprocessing_sequence: ContractionSequence,
    time: std::time::Duration,
}

impl<
        G: Clone
            + AdjacencyList
            + GraphEdgeOrder
            + ColoredAdjacencyList
            + ColoredAdjacencyTest
            + GraphEdgeEditing
            + Debug,
    > TwoStageSatSolver<G>
{
    pub fn new(graph: &G, first_stage_timeout: std::time::Duration) -> TwoStageSatSolver<G> {
        let preprocessing_sequence = ContractionSequence::new(graph.number_of_nodes());

        let clone = graph.clone();

        //prune_leaves(&mut clone, &mut preprocessing_sequence);
        //prune_twins(&mut clone, &mut preprocessing_sequence);

        TwoStageSatSolver {
            graph: clone,
            preprocessing_sequence,
            time: first_stage_timeout,
        }
    }

    pub fn solve(mut self) -> (u32, ContractionSequence) {
        let (sol, heu_seq, _) =
            timeout_monte_carlo_search_tree_solver_preprocessed(&self.graph, self.time);
        let mut sat_encoding = TwinWidthSatEncoding::new(&self.graph);

        // Try to improve upon the existing solution
        if sol != 0
            && let Some((tww,seq)) = sat_encoding.solve_varisat(sol-1) {
            self.preprocessing_sequence.append(&seq);
            (tww,self.preprocessing_sequence)
        }
        else {
            // Could not improve heuristic solver
            self.preprocessing_sequence.append(&heu_seq);
            (sol,self.preprocessing_sequence)
        }
    }
}
