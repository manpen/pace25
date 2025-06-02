use smallvec::SmallVec;

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

    fn has_neighbors<const N: usize>(&self, u: Node, neighbors: [Node; N]) -> [bool; N] {
        self.adj[u as usize].has_neighbors(neighbors)
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
    fn try_add_edge(&mut self, u: Node, v: Node) -> bool {
        let prev = self.adj[u as usize].try_add_edge(v);

        if !prev && u != v {
            let _other = self.adj[v as usize].try_add_edge(u);
            debug_assert!(!_other);
        }

        self.number_of_edges += !prev as NumEdges;

        prev
    }

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        let prev = self.adj[u as usize].try_delete_edge(v);

        if prev && u != v {
            let _other = self.adj[v as usize].try_delete_edge(u);
            debug_assert_eq!(prev, _other);
        }

        self.number_of_edges -= prev as NumEdges;

        prev
    }

    fn remove_edges_at_node(&mut self, u: Node) {
        let neighbors = std::mem::take(&mut self.adj[u as usize]);
        self.number_of_edges -= neighbors.nodes.len() as NumEdges;

        for &v in &neighbors.nodes {
            assert!(self.adj[v as usize].try_delete_edge(u));
        }
    }
}

impl UnsafeGraphEditing for AdjArray {
    unsafe fn remove_half_edges_at_if<F: FnMut(Node) -> bool>(
        &mut self,
        u: Node,
        predicate: F,
    ) -> NumNodes {
        self.adj[u as usize].remove_neighbors_if(predicate)
    }

    unsafe fn remove_half_edges_at(&mut self, u: Node) -> NumNodes {
        let size_before = self.adj[u as usize].degree();
        self.adj[u as usize].clear();
        size_before
    }

    unsafe fn set_number_of_edges(&mut self, m: NumEdges) {
        debug_assert_eq!(2 * m, self.adj.iter().map(|a| a.degree()).sum::<NumEdges>());
        self.number_of_edges = m;
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

        graph.add_edges(edges);

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
    nodes: SmallVec<[Node; 8]>,
}

impl Neighborhood {
    fn degree(&self) -> NumNodes {
        self.nodes.len() as NumNodes
    }

    fn neighbors(&self) -> impl Iterator<Item = Node> + '_ {
        self.nodes.iter().copied()
    }

    fn has_neighbor(&self, v: Node) -> bool {
        self.neighbors().any(|u| u == v)
    }

    fn has_neighbors<const N: usize>(&self, neighbors: [Node; N]) -> [bool; N] {
        let mut res = [false; N];
        for &node in &self.nodes {
            for i in 0..N {
                if neighbors[i] == node {
                    res[i] = true;
                }
            }
        }
        res
    }

    fn try_add_edge(&mut self, v: Node) -> bool {
        if self.nodes.contains(&v) {
            return true;
        }

        self.nodes.push(v);
        false
    }

    fn try_delete_edge(&mut self, v: Node) -> bool {
        if let Some((pos, _)) = self.nodes.iter().find_position(|&&x| x == v) {
            self.nodes.swap_remove(pos);
            true
        } else {
            false
        }
    }

    pub fn remove_neighbors_if<F: FnMut(Node) -> bool>(&mut self, mut predicate: F) -> NumNodes {
        let size_before = self.nodes.len();
        self.nodes.retain(|x| !predicate(*x));
        (size_before - self.nodes.len()) as NumNodes
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
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
