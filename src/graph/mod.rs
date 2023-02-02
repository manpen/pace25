//pub mod adj_list;
pub mod edge;
pub mod pace_reader;

type Node = u32;
type NumNodes = Node;
type NumEdges = u64;
type Edge = (u32, u32);
type ColoredEdge = (u32, u32, EdgeColor);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EdgeColor {
    Black,
    Red,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EdgeType {
    None,
    Black,
    Red,
}
