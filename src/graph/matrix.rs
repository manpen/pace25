use super::*;
use std::ops::Range;

#[derive(Clone)]
pub struct AdjMatrix {
    adj: Vec<BitSet>, // TODO: we should use a single allocation for all of them
    number_of_edges: NumEdges,
}

impl AdjMatrix {
    fn adj_of(&self, u: Node) -> &BitSet {
        &self.adj[2 * u as usize]
    }

    fn red_adj_of(&self, u: Node) -> &BitSet {
        &self.adj[1 + 2 * u as usize]
    }

    fn adj_of_mut(&mut self, u: Node) -> &mut BitSet {
        &mut self.adj[2 * u as usize]
    }

    fn red_adj_of_mut(&mut self, u: Node) -> &mut BitSet {
        &mut self.adj[1 + 2 * u as usize]
    }
}

impl GraphNodeOrder for AdjMatrix {
    type VertexIter<'a> = impl Iterator<Item = Node> + 'a;

    fn number_of_nodes(&self) -> NumNodes {
        self.adj.len() as NumNodes / 2
    }

    fn vertices_range(&self) -> Range<Node> {
        0..self.number_of_nodes()
    }

    fn vertices(&self) -> Self::VertexIter<'_> {
        self.vertices_range()
    }
}

impl GraphEdgeOrder for AdjMatrix {
    fn number_of_edges(&self) -> NumEdges {
        self.number_of_edges
    }
}

impl AdjacencyList for AdjMatrix {
    type NeighborIter<'a> = impl Iterator<Item = Node> + 'a
    where
        Self: 'a;

    fn neighbors_of(&self, u: Node) -> Self::NeighborIter<'_> {
        self.adj_of(u).iter_set_bits()
    }

    fn degree_of(&self, u: Node) -> NumNodes {
        self.adj_of(u).cardinality()
    }

    type NeighborsStream<'a> = impl BitmaskStream + 'a  where Self: 'a;
    fn neighbors_of_as_stream(&self, u: Node) -> Self::NeighborsStream<'_> {
        self.adj_of(u).bitmask_stream()
    }
}

impl ColoredAdjacencyList for AdjMatrix {
    type BlackNeighborIter<'a> = impl Iterator<Item = Node> + 'a
    where
        Self: 'a;

    type RedNeighborIter<'a> = impl Iterator<Item = Node> + 'a
        where
            Self: 'a;

    type BlackNeighborsStream<'a> = impl BitmaskStream + 'a  where Self: 'a;
    type RedNeighborsStream<'a> = impl BitmaskStream + 'a  where Self: 'a;

    fn black_neighbors_of(&self, u: Node) -> Self::BlackNeighborIter<'_> {
        (self.adj_of(u).bitmask_stream() - self.red_adj_of(u).bitmask_stream()).iter_set_bits()
    }

    fn black_neighbors_of_as_bitset(&self, node: Node) -> BitSet {
        self.black_neighbors_of_as_stream(node).into_bitset()
    }

    fn black_neighbors_of_as_stream(&self, node: Node) -> Self::BlackNeighborsStream<'_> {
        self.adj_of(node).bitmask_stream() - self.red_adj_of(node).bitmask_stream()
    }

    fn red_neighbors_of(&self, u: Node) -> Self::RedNeighborIter<'_> {
        self.red_adj_of(u).iter_set_bits()
    }

    fn red_neighbors_of_as_stream(&self, node: Node) -> Self::RedNeighborsStream<'_> {
        self.red_adj_of(node).bitmask_stream()
    }

    fn black_degree_of(&self, u: Node) -> NumNodes {
        self.degree_of(u) - self.red_degree_of(u)
    }

    fn red_degree_of(&self, u: Node) -> NumNodes {
        self.red_adj_of(u).cardinality()
    }
}

impl AdjacencyTest for AdjMatrix {
    fn has_edge(&self, u: Node, v: Node) -> bool {
        self.adj_of(u).get_bit(v)
    }
}

impl ColoredAdjacencyTest for AdjMatrix {
    fn has_black_edge(&self, u: Node, v: Node) -> bool {
        self.has_edge(u, v) && !self.has_red_edge(u, v)
    }

    fn has_red_edge(&self, u: Node, v: Node) -> bool {
        self.red_adj_of(u).get_bit(v)
    }

    fn type_of_edge(&self, u: Node, v: Node) -> EdgeKind {
        if self.has_red_edge(u, v) {
            EdgeKind::Red
        } else if self.has_edge(u, v) {
            EdgeKind::Black
        } else {
            EdgeKind::None
        }
    }
}

impl GraphNew for AdjMatrix {
    fn new(number_of_nodes: NumNodes) -> Self {
        Self {
            adj: (0..2 * number_of_nodes)
                .map(|_| BitSet::new(number_of_nodes))
                .collect(),
            number_of_edges: 0,
        }
    }
}

fn to_edge_kind(exists: bool, is_red: bool) -> EdgeKind {
    // TODO: look at assembly; this should be branch-free
    if is_red {
        EdgeKind::Red
    } else if exists {
        EdgeKind::Black
    } else {
        EdgeKind::None
    }
}

impl GraphEdgeEditing for AdjMatrix {
    fn try_add_edge(&mut self, u: Node, v: Node, color: EdgeColor) -> EdgeKind {
        let prev_edge = self.adj_of_mut(u).set_bit(v);
        let prev_red = self.red_adj_of_mut(u).assign_bit(v, color.is_red());
        let prev = to_edge_kind(prev_edge, prev_red);

        if prev != color && u != v {
            self.adj_of_mut(v).set_bit(u);
            self.red_adj_of_mut(v).assign_bit(u, color.is_red());
        }

        if prev.is_none() {
            self.number_of_edges += 1;
        }

        prev
    }

    fn try_remove_edge(&mut self, u: Node, v: Node) -> EdgeKind {
        let prev_edge = self.adj_of_mut(u).clear_bit(v);
        if !prev_edge {
            return EdgeKind::None;
        }

        let prev_red = self.red_adj_of_mut(u).clear_bit(v);
        let prev = to_edge_kind(prev_edge, prev_red);

        if u != v {
            self.adj_of_mut(v).clear_bit(u);
            self.red_adj_of_mut(v).clear_bit(u);
        }

        self.number_of_edges -= 1;
        prev
    }

    fn remove_edges_at_node(&mut self, u: Node) {
        let neighbors = std::mem::take(self.adj_of_mut(u));
        self.red_adj_of_mut(u).clear_all();
        self.number_of_edges -= neighbors.cardinality() as NumEdges;

        for v in neighbors.iter_set_bits() {
            self.adj_of_mut(v).clear_bit(u);
            self.red_adj_of_mut(v).clear_bit(u);
        }
    }

    fn merge_node_into(&mut self, removed: Node, survivor: Node) {
        assert_ne!(removed, survivor);

        let reds = self.red_neighbors_after_merge(removed, survivor, true);

        *self.adj_of_mut(survivor) |= &reds;
        *self.red_adj_of_mut(survivor) |= &reds;

        for red_neigh in reds.iter_set_bits() {
            self.adj_of_mut(red_neigh).set_bit(survivor);
            self.red_adj_of_mut(red_neigh).set_bit(survivor);
        }

        debug_assert!(!self.has_edge(survivor, survivor));

        self.remove_edges_at_node(removed);
    }

    fn red_neighbors_after_merge(&self, removed: Node, survivor: Node, only_new: bool) -> BitSet {
        let mut turned_red = self.black_neighbors_of_as_bitset(survivor);

        turned_red ^= &self.black_neighbors_of_as_stream(removed); // flip bits
        turned_red |= &self.red_neighbors_of_as_stream(removed);

        if only_new {
            turned_red -= &self.red_neighbors_of_as_stream(survivor);
        } else {
            turned_red |= &self.red_neighbors_of_as_stream(survivor);
        }

        turned_red.clear_bit(removed);
        turned_red.clear_bit(survivor);

        turned_red
    }
}

impl AdjMatrix {
    pub fn unordered_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.vertices_range()
            .flat_map(|u| self.neighbors_of(u).map(move |v| Edge(u, v)))
    }

    pub fn unordered_colored_edges(&self) -> impl Iterator<Item = ColoredEdge> + '_ {
        self.vertices_range().flat_map(|u| {
            self.black_neighbors_of(u)
                .map(move |v| ColoredEdge(u, v, EdgeColor::Black))
                .chain(
                    self.red_neighbors_of(u)
                        .map(move |v| ColoredEdge(u, v, EdgeColor::Red)),
                )
        })
    }

    pub fn test_only_from(edges: impl Clone + IntoIterator<Item = impl Into<Edge>>) -> Self {
        let n = edges
            .clone()
            .into_iter()
            .map(|e| e.into())
            .map(|e| e.0.max(e.1) + 1)
            .max()
            .unwrap_or(0);
        let mut graph = Self::new(n as NumNodes);

        graph.add_edges(edges, EdgeColor::Black);

        graph
    }
}

super::graph_tests::impl_graph_tests!(AdjMatrix);
