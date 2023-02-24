use crate::prelude::{contraction_sequence::ContractionSequence, AdjacencyList, GraphEdgeOrder, ColoredAdjacencyList, ColoredAdjacencyTest, GraphEdgeEditing};
use std::fmt::Debug;
pub struct ContractionRefiner<G> {
    sequence: ContractionSequence,
    graph: G,
    tww: u32
}


impl<G: Clone
+ AdjacencyList
+ GraphEdgeOrder
+ ColoredAdjacencyList
+ ColoredAdjacencyTest
+ Debug
+ GraphEdgeEditing> ContractionRefiner<G> {
    pub fn new(graph: &G, seq: ContractionSequence, tww: u32) -> ContractionRefiner<G> {
        ContractionRefiner { sequence: seq, graph: graph.clone(), tww}
    }

    pub fn solve(&self) -> (u32,ContractionSequence) {
        let mut current_index = 0;
        loop {
            
        }
    }
}