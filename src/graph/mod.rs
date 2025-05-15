pub mod adj_array;
pub mod bipartite;
pub mod bridges;
pub mod complement;
pub mod connectivity;
pub mod csr;
pub mod cut_vertex;
pub mod distance_two_pairs;
pub mod edge;
pub mod gnp;
pub mod graph_digest;
pub mod matrix;
pub mod modules;
pub mod node_mapper;
pub mod partition;
pub mod relabel;
pub mod subgraph;
pub mod traversal;

pub use adj_array::*;
pub use bipartite::*;
pub use bridges::*;
pub use complement::*;
pub use connectivity::*;
pub use csr::*;
pub use cut_vertex::*;
pub use distance_two_pairs::*;
pub use edge::*;
pub use gnp::*;
pub use graph_digest::*;
pub use matrix::*;
pub use modules::Modules;
pub use node_mapper::*;
pub use partition::*;
pub use subgraph::*;
pub use traversal::*;

use itertools::Itertools;
use std::{borrow::Borrow, ops::Range};
use stream_bitset::prelude::*;
mod graph_tests;
pub mod sliced_buffer;

pub type Node = u32;
pub type NumNodes = Node;
pub type NumEdges = u32;
pub type BitSet = BitSetImpl<Node>;
pub type EdgeBitSet = BitSetImpl<NumEdges>;

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

    /// Returns empty bitset with one entry per node
    fn vertex_bitset_unset(&self) -> BitSet {
        BitSet::new(self.number_of_nodes())
    }

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

    /// Returns empty bitset with one entry per node
    fn edge_bitset_unset(&self) -> BitSet {
        BitSet::new(self.number_of_edges())
    }
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
            BitSet::new_with_bits_set(self.number_of_nodes(), self.$slice(node))
        }
    };
}

pub trait AdjacencyList: GraphNodeOrder + Sized {
    type NeighborIter<'a>: Iterator<Item = Node> + 'a
    where
        Self: 'a;

    /// Returns a slice of neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn neighbors_of(&self, u: Node) -> Self::NeighborIter<'_>;

    /// If v has degree two (i.e. neighbors [u, w]), this function continues
    /// the walk `u`, `v`, `w` and returns `Some(w)`. Otherwise it returns `None`.
    fn continue_path(&self, u: Node, v: Node) -> Option<Node> {
        (self.degree_of(v) == 2).then(|| self.neighbors_of(v).find(|&w| w != u).unwrap())
    }

    /// Returns the number of neighbors of from `u`
    fn degree_of(&self, u: Node) -> NumNodes;

    // Returns an iterator to all vertices with non-zero degree
    fn vertices_with_neighbors(&self) -> impl Iterator<Item = Node> + '_ {
        self.degrees()
            .enumerate()
            .filter_map(|(u, d)| (d > 0).then_some(u as Node))
    }

    // Returns the number of nodes with non-zero degree
    fn number_of_nodes_with_neighbors(&self) -> NumNodes {
        self.vertices_with_neighbors().count() as NumNodes
    }

    // Returns a distribution sorted by degree
    fn degree_distribution(&self) -> Vec<(NumNodes, NumNodes)> {
        let mut distr = self
            .degrees()
            .counts()
            .into_iter()
            .map(|(d, n)| (d, n as NumNodes))
            .collect_vec();
        distr.sort_by_key(|(d, _)| *d);
        distr
    }

    fn max_degree(&self) -> NumNodes {
        self.degrees().max().unwrap_or(0)
    }

    node_iterator!(degrees, degree_of, NumNodes);
    node_iterator!(neighbors, neighbors_of, Self::NeighborIter<'_>);
    node_bitset_of!(neighbors_of_as_bitset, neighbors_of);
    node_iterator!(neighbors_as_bitset, neighbors_of_as_bitset, BitSet);

    type NeighborsStream<'a>: BitmaskStream + 'a
    where
        Self: 'a;
    fn neighbors_of_as_stream(&self, u: Node) -> Self::NeighborsStream<'_>;

    fn closed_two_neighborhood_of(&self, u: Node) -> BitSet {
        let mut ns = self.vertex_bitset_unset();
        ns.set_bit(u);
        for v in self.neighbors_of(u) {
            ns.set_bit(v);
            ns.set_bits(self.neighbors_of(v));
        }
        ns
    }

    fn closed_three_neighborhood_of(&self, u: Node) -> BitSet {
        let mut ns = self.vertex_bitset_unset();
        ns.set_bit(u);
        for v in self.closed_two_neighborhood_of(u).iter_set_bits() {
            ns.set_bit(v);
            ns.set_bits(self.neighbors_of(v));
        }
        ns
    }

    fn edges_of(&self, u: Node, only_normalized: bool) -> impl Iterator<Item = Edge> + '_ {
        self.neighbors_of(u)
            .map(move |v| Edge(u, v))
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
    type BlackNeighborIter<'a>: Iterator<Item = Node> + 'a
    where
        Self: 'a;

    type RedNeighborIter<'a>: Iterator<Item = Node> + 'a
    where
        Self: 'a;

    type BlackNeighborsStream<'a>: BitmaskStream + 'a
    where
        Self: 'a;
    type RedNeighborsStream<'a>: BitmaskStream + 'a
    where
        Self: 'a;

    /// Returns a slice of black neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn black_neighbors_of(&self, u: Node) -> Self::BlackNeighborIter<'_>;
    node_bitset_of!(black_neighbors_of_as_bitset, black_neighbors_of);
    fn black_neighbors_of_as_stream(&self, u: Node) -> Self::BlackNeighborsStream<'_>;

    /// Returns a slice of red neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn red_neighbors_of(&self, u: Node) -> Self::RedNeighborIter<'_>;
    node_bitset_of!(red_neighbors_of_as_bitset, red_neighbors_of);
    fn red_neighbors_of_as_stream(&self, u: Node) -> Self::RedNeighborsStream<'_>;

    /// Returns the number of black neighbors of from `u`
    fn black_degree_of(&self, u: Node) -> NumNodes;

    /// Returns the number of red neighbors of from `u`
    fn red_degree_of(&self, u: Node) -> NumNodes;

    fn max_red_degree(&self) -> NumNodes {
        self.red_degrees().max().unwrap_or(0)
    }

    fn max_black_degree(&self) -> NumNodes {
        self.black_degrees().max().unwrap_or(0)
    }

    /// Returns an iterator of the colored edges incident to `u`
    fn colored_edges_of(
        &self,
        u: Node,
        only_normalized: bool,
    ) -> impl Iterator<Item = ColoredEdge> + '_ {
        self.black_neighbors_of(u)
            .map(move |v| ColoredEdge(u, v, EdgeColor::Black))
            .chain(
                self.red_neighbors_of(u)
                    .map(move |v| ColoredEdge(u, v, EdgeColor::Red)),
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
    node_iterator!(
        black_neighbors,
        black_neighbors_of,
        Self::BlackNeighborIter<'_>
    );
    node_iterator!(red_neighbors, red_neighbors_of, Self::RedNeighborIter<'_>);
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

    fn remove_edges(&mut self, edges: impl IntoIterator<Item = impl Borrow<Edge>>) {
        for Edge(u, v) in edges.into_iter().map(|d| *d.borrow()) {
            self.remove_edge(u, v);
        }
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

/// A trait that allows accessing and modification of neighbors by index
pub trait IndexedAdjacencyList: AdjacencyList {
    /// Returns the ith neighbor of a node
    ///
    /// Possibly panics if i >= degree_of(u)
    fn ith_neighbor(&self, u: Node, i: NumNodes) -> Node;

    /// If ith_neighbor(u, i) = v, returns j such that ith_neighbor(v, j) = u.
    ///
    /// Possibly panics if i >= degree_of(u)
    fn ith_cross_position(&self, u: Node, i: NumNodes) -> NumNodes;

    /// Swaps neighbos at positions nb1_pos and nb2_pos of u
    ///
    /// Possibly panics if nb1_pos >= degree_of(u) || nb2_pos >= degree_of(u)
    fn swap_neighbors(&mut self, u: Node, nb1_pos: NumNodes, nb2_pos: NumNodes);
}

/// A super trait for creating a graph from scratch from a set of edges and a number of nodes
pub trait GraphFromReader {
    /// Create a graph from a number of nodes and an iterator over Edges
    fn from_edges(n: NumNodes, edges: impl IntoIterator<Item = impl Into<Edge>>) -> Self;
}

impl<G: GraphNew + GraphEdgeEditing> GraphFromReader for G {
    fn from_edges(n: NumNodes, edges: impl IntoIterator<Item = impl Into<Edge>>) -> Self {
        let mut graph = Self::new(n);
        graph.add_edges(edges, EdgeColor::Black);
        graph
    }
}

/// Trait for extracting a Csr-Representation from the graph.
pub trait ExtractCsrRepr {
    /// Extract a CSR representation of the graph, ie. a compacted list of all edges sorted by
    /// source and a list of offsets indicating where the neighbors of u begin to appear in this
    /// edge list.
    fn extract_csr_repr(&self) -> CsrEdges;
}

/// A marker trait indicating that *u* is considered a neighbor of *u*
pub trait SelfLoop {}

pub trait FullfledgedGraph:
    Clone
    + AdjacencyList
    + GraphEdgeOrder
    + ColoredAdjacencyList
    + ColoredAdjacencyTest
    + GraphEdgeEditing
    + Complement
    + GraphDigest
    + std::fmt::Debug
{
}

impl<G> FullfledgedGraph for G where
    G: Clone
        + AdjacencyList
        + GraphEdgeOrder
        + ColoredAdjacencyList
        + ColoredAdjacencyTest
        + GraphEdgeEditing
        + Complement
        + GraphDigest
        + std::fmt::Debug
{
}

/// A static graph possibly does not require the ability of edge-modification but instead allows
/// for edge-reordering
pub trait StaticGraph:
    Clone + IndexedAdjacencyList + GraphEdgeOrder + AdjacencyTest + GraphFromReader + ExtractCsrRepr
{
}

impl<
    G: Clone
        + IndexedAdjacencyList
        + GraphEdgeOrder
        + AdjacencyTest
        + GraphFromReader
        + ExtractCsrRepr,
> StaticGraph for G
{
}
