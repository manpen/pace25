pub mod adj_array;
pub mod bitset;
pub mod color_filter;
pub mod connectivity;
pub mod cut_vertex;
pub mod distance_two_pairs;
pub mod edge;
pub mod gnp;
pub mod graph_digest;
pub mod node_mapper;
pub mod partition;
pub mod traversal;
pub mod twin_width_sat_encoding;

pub use adj_array::*;
pub use bitset::*;
pub use color_filter::*;
pub use connectivity::*;
pub use cut_vertex::*;
pub use distance_two_pairs::*;
pub use edge::*;
pub use gnp::*;
pub use graph_digest::*;
pub use node_mapper::*;
pub use partition::*;
pub use traversal::*;

use itertools::Itertools;
use std::{borrow::Borrow, ops::Range};

pub type Node = u32;
pub type NumNodes = Node;
pub type NumEdges = u64;

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

#[macro_export]
macro_rules! node_bitset_of {
    ($bitset : ident, $slice : ident) => {
        fn $bitset(&self, node: Node) -> BitSet {
            BitSet::new_all_unset_but::<NumNodes, _, _>(
                self.number_of_nodes(),
                self.$slice(node).iter(),
            )
        }
    };
}

pub trait AdjacencyList: GraphNodeOrder + Sized {
    /// Returns a slice of neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn neighbors_of(&self, u: Node) -> &[Node];

    /// Returns the number of neighbors of from `u`
    fn degree_of(&self, u: Node) -> NumNodes {
        self.neighbors_of(u).len() as NumNodes
    }

    node_iterator!(degrees, degree_of, NumNodes);
    node_iterator!(neighbors, neighbors_of, &[Node]);
    node_bitset_of!(neighbors_of_as_bitset, neighbors_of);
    node_iterator!(neighbors_as_bitset, neighbors_of_as_bitset, BitSet);

    fn closed_two_neighborhood_of(&self, u: Node) -> BitSet {
        let mut ns = BitSet::new(self.number_of_nodes());
        ns.set_bit(u);
        for &v in self.neighbors_of(u) {
            ns.set_bit(v);
            ns.set_bits(self.neighbors_of(v).iter().copied());
        }
        ns
    }

    fn edges_of(&self, u: Node, only_normalized: bool) -> impl Iterator<Item = Edge> + '_ {
        self.neighbors_of(u)
            .iter()
            .map(move |&v| Edge(u, v))
            .filter(move |e| !only_normalized || e.is_normalized())
    }

    fn ordered_edges_of(&self, u: Node, only_normalized: bool) -> impl Iterator<Item = Edge> {
        let mut edges = self.edges_of(u, only_normalized).collect_vec();
        edges.sort();
        edges.into_iter()
    }

    fn edges(&self, only_normalized: bool) -> impl Iterator<Item = Edge> + '_ {
        self.vertices_range()
            .flat_map(move |u| self.edges_of(u, only_normalized))
    }

    fn ordered_edges(&self, only_normalized: bool) -> impl Iterator<Item = Edge> + '_ {
        self.vertices_range()
            .flat_map(move |u| self.ordered_edges_of(u, only_normalized))
    }
}

pub trait ColoredAdjacencyList: AdjacencyList {
    /// Returns a slice of black neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn black_neighbors_of(&self, u: Node) -> &[Node];
    node_bitset_of!(black_neighbors_of_as_bitset, black_neighbors_of);

    /// Returns a slice of red neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn red_neighbors_of(&self, u: Node) -> &[Node];
    node_bitset_of!(red_neighbors_of_as_bitset, red_neighbors_of);

    /// Returns the number of black neighbors of from `u`
    fn black_degree_of(&self, u: Node) -> NumNodes {
        self.black_neighbors_of(u).len() as NumNodes
    }

    /// Returns the number of red neighbors of from `u`
    fn red_degree_of(&self, u: Node) -> NumNodes {
        self.red_neighbors_of(u).len() as NumNodes
    }

    /// Returns an iterator of the colored edges incident to `u`
    fn colored_edges_of(
        &self,
        u: Node,
        only_normalized: bool,
    ) -> impl Iterator<Item = ColoredEdge> + '_ {
        self.black_neighbors_of(u)
            .iter()
            .map(move |&v| ColoredEdge(u, v, EdgeColor::Black))
            .chain(
                self.red_neighbors_of(u)
                    .iter()
                    .map(move |&v| ColoredEdge(u, v, EdgeColor::Red)),
            )
            .filter(move |e| !only_normalized || e.is_normalized())
    }

    /// Returns an iterator of the colored edges incident to `u` sorted by neighbor index.
    /// This involves allocation!
    fn ordered_colored_edges_of(
        &self,
        u: Node,
        only_normalized: bool,
    ) -> impl Iterator<Item = ColoredEdge> {
        let mut edges = self.colored_edges_of(u, only_normalized).collect_vec();
        edges.sort();
        edges.into_iter()
    }

    fn colored_edges(&self, only_normalized: bool) -> impl Iterator<Item = ColoredEdge> + '_ {
        self.vertices_range()
            .flat_map(move |u| self.colored_edges_of(u, only_normalized))
    }

    fn ordered_colored_edges(
        &self,
        only_normalized: bool,
    ) -> impl Iterator<Item = ColoredEdge> + '_ {
        self.vertices_range()
            .flat_map(move |u| self.ordered_colored_edges_of(u, only_normalized))
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

    fn add_colored_edges(&mut self, edges: impl IntoIterator<Item = impl Borrow<ColoredEdge>>) {
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

    /// Dry-run of [`GraphEdgeEditing::merge_node_into`] that only returns the red-degree after a merge were carried out
    fn red_degree_after_merge(&self, removed: Node, survivor: Node) -> NumNodes {
        self.red_neighbors_after_merge(removed, survivor, false)
            .cardinality() as NumNodes
    }

    /// Dry-run of [`GraphEdgeEditing::merge_node_into`] that only returns the red-neighbors after a merge
    fn red_neighbors_after_merge(&self, removed: Node, survivor: Node, only_new: bool) -> BitSet;
}
