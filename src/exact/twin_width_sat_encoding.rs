use crate::graph::{AdjacencyList, GraphEdgeOrder, GraphEdgeEditing};

pub struct TwinWidthSatEncoding<G> {
    graph: G
}

impl<G: Clone
        +AdjacencyList
        + GraphEdgeOrder
        + GraphEdgeEditing> TwinWidthSatEncoding<G> {
    pub fn encode(&self) -> Vec<Vec<i32>> {
        Vec::new()
    }
}