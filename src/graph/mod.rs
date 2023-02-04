pub mod adj_list;
pub mod color_filter;
pub mod edge;
pub mod traversal;

pub type Node = u32;
pub type NumNodes = Node;
pub type NumEdges = u64;

use std::{borrow::Borrow, ops::Range};

pub use adj_list::*;
pub use edge::*;
pub use traversal::*;

/// Provides getters pertaining to the size of a graph
pub trait GraphNodeOrder {
    type VertexIter<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    /// Returns the number of nodes of the graph
    fn number_of_nodes(&self) -> Node;

    /// Return the number of nodes as usize
    fn len(&self) -> usize {
        self.number_of_nodes() as usize
    }

    /// Returns an iterator over V.
    fn vertices(&self) -> Self::VertexIter<'_>;

    /// Returns a range of vertices possibly including deleted vertices
    /// In contrast to self.vertices(), the range returned by self.vertices_ranges() does
    /// not borrow self and hence may be used where additional mutable references of self are needed
    ///
    /// # Warning
    /// This method may iterate over deleted vertices (if supported by an implementation). It is the
    /// responsibility of the caller to identify and treat them accordingly.
    fn vertices_range(&self) -> Range<Node> {
        0..self.number_of_nodes()
    }

    /// Returns true if the graph has no nodes (and thus no edges)
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait GraphEdgeOrder {
    /// Returns the number of edges of the graph
    fn number_of_edges(&self) -> NumEdges;
}

#[macro_export]
macro_rules! node_iterator {
    ($iter : ident, $single : ident, $type : ty) => {
        fn $iter(&self) -> impl Iterator<Item = $type> + '_ {
            self.vertices().map(|u| self.$single(u))
        }
    };
}

pub trait AdjacencyList: GraphNodeOrder + Sized {
    /// Returns a slice of neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn neighbors_of(&self, u: Node) -> &[Node];

    /// Returns the number of neighbors of from [`u`]
    fn degree_of(&self, u: Node) -> NumNodes {
        self.neighbors_of(u).len() as NumNodes
    }

    node_iterator!(degrees, degree_of, NumNodes);
    node_iterator!(neighbors, neighbors_of, &[Node]);
}

pub trait ColoredAdjacencyList: AdjacencyList {
    /// Returns a slice of black neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn black_neighbors_of(&self, u: Node) -> &[Node];

    /// Returns a slice of red neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn red_neighbors_of(&self, u: Node) -> &[Node];

    /// Returns the number of black neighbors of from [`u`]
    fn black_degree_of(&self, u: Node) -> NumNodes {
        self.black_neighbors_of(u).len() as NumNodes
    }

    /// Returns the number of red neighbors of from [`u`]
    fn red_degree_of(&self, u: Node) -> NumNodes {
        self.red_neighbors_of(u).len() as NumNodes
    }

    node_iterator!(black_degrees, black_degree_of, NumNodes);
    node_iterator!(red_degrees, red_degree_of, NumNodes);
    node_iterator!(black_neighbors, black_neighbors_of, &[Node]);
    node_iterator!(red_neighbors, red_neighbors_of, &[Node]);
}

/// Provides efficient tests whether an edge exists
pub trait AdjacencyTest {
    /// Returns *true* exactly if the graph contains the directed edge (u, v)
    fn has_edge(&self, u: Node, v: Node) -> bool;
}

pub trait ColoredAdjacencyTest: AdjacencyTest {
    fn has_black_edge(&self, u: Node, v: Node) -> bool {
        self.type_of_edge(u, v).is_black()
    }

    fn has_red_edge(&self, u: Node, v: Node) -> bool {
        self.type_of_edge(u, v).is_red()
    }

    fn type_of_edge(&self, u: Node, v: Node) -> EdgeKind;
}

pub trait GraphNew {
    /// Creates an empty graph with n singleton nodes
    fn new(n: NumNodes) -> Self;
}

/// Provides functions to insert/delete edges
pub trait GraphEdgeEditing: GraphNew {
    /// Adds the directed edge *(u,v)* to the graph. I.e., the edge FROM u TO v.
    /// ** Panics if the edge is already contained or possibly if u, v >= n **
    fn add_edge(&mut self, u: Node, v: Node, color: EdgeColor) {
        assert!(self.try_add_edge(u, v, color).is_none())
    }

    /// Adds the directed edge *(u,v)* to the graph. I.e., the edge FROM u TO v.
    /// Returns *true* exactly if the edge was not present previously.
    /// ** Can panic if u, v >= n, depending on implementation **
    fn try_add_edge(&mut self, u: Node, v: Node, color: EdgeColor) -> EdgeKind;

    fn add_edges(&mut self, edges: impl IntoIterator<Item = impl Into<Edge>>, color: EdgeColor) {
        for Edge(u, v) in edges.into_iter().map(|d| d.into()) {
            self.add_edge(u, v, color);
        }
    }

    fn add_colored_edges<I>(&mut self, edges: impl IntoIterator<Item = impl Borrow<ColoredEdge>>) {
        for ColoredEdge(u, v, color) in edges.into_iter().map(|d| *d.borrow()) {
            self.add_edge(u, v, color);
        }
    }

    /// Removes the directed edge *(u,v)* from the graph. I.e., the edge FROM u TO v.
    /// ** Panics if the edge is not present or u, v >= n **
    fn remove_edge(&mut self, u: Node, v: Node) {
        assert!(self.try_remove_edge(u, v).is_some())
    }

    /// Removes the directed edge *(u,v)* from the graph. I.e., the edge FROM u TO v.
    /// If the edge was removed, returns *true* and *false* otherwise.
    /// ** Panics if u, v >= n **
    fn try_remove_edge(&mut self, u: Node, v: Node) -> EdgeKind;

    /// Removes all edges into and out of node u
    fn remove_edges_at_node(&mut self, u: Node);

    /// Removes all edges into and out of node `u` and connects every in-neighbor with every out-neighbor.
    fn merge_node_into(&mut self, removed: Node, survivor: Node);
}
