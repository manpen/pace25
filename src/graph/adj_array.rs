use super::*;
use std::fmt;

#[derive(Clone)]
pub struct AdjArray {
    adj: Vec<Neighborhood>,
    number_of_edges: NumEdges,
}

macro_rules! forward {
    ($single : ident, $internal : ident, $type : ty) => {
        fn $single(&self, node: Node) -> $type {
            self.adj[node as usize].$internal()
        }
    };
}

impl GraphNodeOrder for AdjArray {
    type VertexIter<'a> = impl Iterator<Item = Node> + 'a;

    fn number_of_nodes(&self) -> NumNodes {
        self.adj.len() as NumNodes
    }

    fn vertices_range(&self) -> Range<Node> {
        0..self.number_of_nodes()
    }

    fn vertices(&self) -> Self::VertexIter<'_> {
        self.vertices_range()
    }
}

impl GraphEdgeOrder for AdjArray {
    fn number_of_edges(&self) -> NumEdges {
        self.number_of_edges
    }
}

impl AdjacencyList for AdjArray {
    type NeighborIter<'a>
        = impl Iterator<Item = Node> + 'a
    where
        Self: 'a;

    forward!(degree_of, degree, NumNodes);
    forward!(neighbors_of, neighbors, Self::NeighborIter<'_>);

    type NeighborsStream<'a>
        = BitsetStream<Node>
    where
        Self: 'a;
    fn neighbors_of_as_stream(&self, u: Node) -> Self::NeighborsStream<'_> {
        self.neighbors_of_as_bitset(u).into_bitmask_stream()
    }
}

impl AdjacencyTest for AdjArray {
    fn has_edge(&self, u: Node, v: Node) -> bool {
        self.adj[u as usize].has_neighbor(v)
    }
}

impl GraphNew for AdjArray {
    fn new(number_of_nodes: NumNodes) -> Self {
        Self {
            adj: vec![Default::default(); number_of_nodes as usize],
            number_of_edges: 0,
        }
    }
}

impl GraphEdgeEditing for AdjArray {
    fn try_add_edge(&mut self, u: Node, v: Node, color: EdgeColor) -> EdgeKind {
        let prev = self.adj[u as usize].try_add_edge(v, color);

        if prev != color && u != v {
            assert_eq!(self.adj[v as usize].try_add_edge(u, color), prev)
        }

        if prev.is_none() {
            self.number_of_edges += 1;
        }

        prev
    }

    fn try_remove_edge(&mut self, u: Node, v: Node) -> EdgeKind {
        let prev = self.adj[u as usize].try_delete_edge(v);

        if prev.is_some() && u != v {
            let _other = self.adj[v as usize].try_delete_edge(u);
            debug_assert_eq!(prev, _other);
        }

        if prev.is_some() {
            self.number_of_edges -= 1;
        }

        prev
    }

    fn remove_edges_at_node(&mut self, u: Node) {
        let neighbors = std::mem::take(&mut self.adj[u as usize]);
        self.number_of_edges -= neighbors.nodes.len() as NumEdges;

        for &v in &neighbors.nodes {
            assert!(self.adj[v as usize].try_delete_edge(u).is_some());
        }
    }
}

impl AdjArray {
    pub fn unordered_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.vertices_range()
            .flat_map(|u| self.neighbors_of(u).map(move |v| Edge(u, v)))
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

/// The Csr-Represntation should have self-loops
impl ExtractCsrRepr for AdjArray {
    fn extract_csr_repr(&self) -> CsrEdges {
        let mut offsets = Vec::with_capacity(self.len() + 1);
        offsets.push(0);

        let mut csr_edges = Vec::with_capacity(self.number_of_edges() as usize + self.len());
        for u in self.vertices() {
            debug_assert!(!self.has_edge(u, u));
            csr_edges.push(u);
            csr_edges.extend_from_slice(self.as_neighbors_slice(u));
            offsets.push(offsets[u as usize] + self.degree_of(u) + 1);
        }

        CsrEdges::new(csr_edges, offsets)
    }
}

#[derive(Default, Clone)]
struct Neighborhood {
    nodes: Vec<Node>,
    red_degree: NumNodes,
}

impl Neighborhood {
    fn degree(&self) -> NumNodes {
        self.nodes.len() as NumNodes
    }

    fn black_degree(&self) -> NumNodes {
        self.nodes.len() as NumNodes - self.red_degree
    }

    fn red_degree(&self) -> NumNodes {
        self.red_degree
    }

    fn neighbors(&self) -> impl Iterator<Item = Node> + '_ {
        self.nodes.iter().copied()
    }

    fn black_neighbors(&self) -> impl Iterator<Item = Node> + '_ {
        self.nodes[0..self.black_degree() as usize].iter().copied()
    }

    fn red_neighbors(&self) -> impl Iterator<Item = Node> + '_ {
        self.nodes[self.black_degree() as usize..].iter().copied()
    }

    fn edge_type_with(&self, v: Node) -> EdgeKind {
        if self.has_black_neighbor(v) {
            EdgeKind::Black
        } else if self.has_red_neighbor(v) {
            EdgeKind::Red
        } else {
            EdgeKind::None
        }
    }

    fn has_neighbor(&self, v: Node) -> bool {
        self.neighbors().any(|u| u == v)
    }

    fn has_black_neighbor(&self, v: Node) -> bool {
        self.black_neighbors().any(|u| u == v)
    }

    fn has_red_neighbor(&self, v: Node) -> bool {
        self.red_neighbors().any(|u| u == v)
    }

    fn try_add_edge(&mut self, v: Node, kind: EdgeColor) -> EdgeKind {
        let (position, previous) = self.find_neighbor(v);

        match previous {
            EdgeKind::None => {
                self.push_red(v);
                if kind == EdgeColor::Black {
                    self.recolor_red_to_black(self.nodes.len() - 1);
                }
            }

            EdgeKind::Black => {
                if kind == EdgeColor::Red {
                    self.recolor_black_to_red(position.unwrap());
                }
            }

            EdgeKind::Red => {
                if kind == EdgeColor::Black {
                    self.recolor_red_to_black(position.unwrap());
                }
            }
        }

        previous
    }

    fn find_neighbor(&self, v: Node) -> (Option<usize>, EdgeKind) {
        let position = self.neighbors().position(|x| x == v);

        match position {
            None => (None, EdgeKind::None),
            Some(x) if x < self.black_degree() as usize => (Some(x), EdgeKind::Black),
            Some(x) => (Some(x), EdgeKind::Red),
        }
    }

    fn try_delete_edge(&mut self, v: Node) -> EdgeKind {
        let (position, previous) = self.find_neighbor(v);

        if previous.is_none() {
            return EdgeKind::None;
        }

        let idx = if previous.is_black() {
            self.recolor_black_to_red(position.unwrap())
        } else {
            position.unwrap()
        };

        self.nodes.swap_remove(idx);
        self.red_degree -= 1;

        previous
    }

    fn push_red(&mut self, v: Node) {
        self.nodes.push(v);
        self.red_degree += 1;
    }

    fn recolor_red_to_black(&mut self, idx: usize) -> usize {
        let first_red = self.black_degree() as usize;
        debug_assert!(idx >= first_red);
        self.nodes.swap(first_red, idx);
        self.red_degree -= 1;
        first_red
    }

    fn recolor_black_to_red(&mut self, idx: usize) -> usize {
        let last_black = self.black_degree().checked_sub(1).unwrap() as usize;
        debug_assert!(idx <= last_black);
        self.nodes.swap(last_black, idx);
        self.red_degree += 1;
        last_black
    }
}

impl fmt::Debug for AdjArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use super::super::io::DotWriter;
        use std::str;

        let mut buf = Vec::new();
        if self.try_write_dot(&mut buf).is_ok() {
            f.write_str(str::from_utf8(&buf).unwrap().trim())?;
        }

        Ok(())
    }
}

impl NeighborsSlice for AdjArray {
    #[inline(always)]
    fn as_neighbors_slice(&self, u: Node) -> &[Node] {
        &self.adj[u as usize].nodes
    }

    #[inline(always)]
    fn as_neighbors_slice_mut(&mut self, u: Node) -> &mut [Node] {
        &mut self.adj[u as usize].nodes
    }
}

super::graph_tests::impl_graph_tests!(AdjArray);
