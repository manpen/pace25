use crate::impl_static_graph_tests;

use super::{sliced_buffer::SlicedBuffer, *};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct NodeWithCrossPos {
    pub node: Node,
    pub cross_pos: NumNodes,
}

#[derive(Debug, Clone)]
pub struct CsrGraph {
    neighborhoods: SlicedBuffer<NodeWithCrossPos>,
    max_degree: NumNodes,
}

impl GraphNodeOrder for CsrGraph {
    type VertexIter<'a> = impl Iterator<Item = Node> + 'a;

    fn number_of_nodes(&self) -> NumNodes {
        self.neighborhoods.number_of_nodes()
    }

    fn vertices_range(&self) -> Range<Node> {
        0..self.number_of_nodes()
    }

    fn vertices(&self) -> Self::VertexIter<'_> {
        self.vertices_range()
    }
}

impl GraphEdgeOrder for CsrGraph {
    fn number_of_edges(&self) -> NumEdges {
        self.neighborhoods.number_of_edges() - self.number_of_nodes() as NumEdges
    }
}

impl ExtractCsrRepr for CsrGraph {
    fn extract_csr_repr(&self) -> CsrEdges {
        CsrEdges::new(
            self.neighborhoods
                .raw_buffer_slice()
                .iter()
                .map(|x| x.node)
                .collect(),
            self.neighborhoods.raw_offset_slice().to_vec(),
        )
    }
}

impl AdjacencyList for CsrGraph {
    type NeighborIter<'a>
        = impl Iterator<Item = Node> + 'a
    where
        Self: 'a;

    fn neighbors_of(&self, u: Node) -> Self::NeighborIter<'_> {
        self.neighborhoods[u].iter().map(|x| x.node)
    }

    fn degree_of(&self, u: Node) -> NumNodes {
        self.neighborhoods.degree_of(u)
    }

    fn max_degree(&self) -> NumNodes {
        self.max_degree
    }

    type NeighborsStream<'a>
        = BitsetStream<Node>
    where
        Self: 'a;

    fn neighbors_of_as_stream(&self, u: Node) -> Self::NeighborsStream<'_> {
        self.neighbors_of_as_bitset(u).into_bitmask_stream()
    }
}

impl AdjacencyTest for CsrGraph {
    fn has_edge(&self, u: Node, v: Node) -> bool {
        // TODO: for skewed degree distrs, it might be beneficial to search the smaller neighborhood
        self.neighborhoods[u].iter().any(|x| x.node == v)
    }
}

impl IndexedAdjacencyList for CsrGraph {
    #[inline(always)]
    fn ith_neighbor(&self, u: Node, i: NumNodes) -> Node {
        self.neighborhoods[u][i as usize].node
    }

    #[inline(always)]
    fn ith_cross_position(&self, u: Node, i: NumNodes) -> NumNodes {
        self.neighborhoods[u][i as usize].cross_pos
    }

    fn swap_neighbors(&mut self, u: Node, nb1_pos: NumNodes, nb2_pos: NumNodes) {
        let nb1 = self.neighborhoods[u][nb1_pos as usize];
        let nb2 = self.neighborhoods[u][nb2_pos as usize];

        debug_assert_eq!(self.ith_neighbor(nb1.node, nb1.cross_pos), u);
        debug_assert_eq!(self.ith_neighbor(nb2.node, nb2.cross_pos), u);
        debug_assert_eq!(self.ith_cross_position(nb1.node, nb1.cross_pos), nb1_pos);
        debug_assert_eq!(self.ith_cross_position(nb2.node, nb2.cross_pos), nb2_pos);

        self.neighborhoods[nb1.node][nb1.cross_pos as usize].cross_pos = nb2_pos;
        self.neighborhoods[nb2.node][nb2.cross_pos as usize].cross_pos = nb1_pos;

        self.neighborhoods[u].swap(nb1_pos as usize, nb2_pos as usize);
    }
}

impl GraphFromReader for CsrGraph {
    fn from_edges(n: NumNodes, edges: impl IntoIterator<Item = impl Into<Edge>>) -> Self {
        assert!(n > 0);

        let n = n as usize;

        let mut num_of_neighbors: Vec<NumNodes> = vec![0; n];
        let temp_edges: Vec<Edge> = edges
            .into_iter()
            .map(|edge| {
                let Edge(u, v) = edge.into();

                num_of_neighbors[u as usize] += 1;
                num_of_neighbors[v as usize] += 1;

                Edge(u, v)
            })
            .collect();

        let m = temp_edges.len() * 2 + n;

        let mut max_degree = 0;

        let mut offsets = Vec::with_capacity(n + 1);
        let mut edges = vec![NodeWithCrossPos::default(); m];

        offsets.push(0);

        let mut running_offset = num_of_neighbors[0] + 1;
        for u in 1..n {
            offsets.push(running_offset as NumEdges);
            edges[running_offset as usize].node = u as Node;

            running_offset += num_of_neighbors[u] + 1;

            num_of_neighbors[u] = 1;
            max_degree = max_degree.max(offsets[u] - offsets[u - 1]);
        }

        offsets.push(running_offset as NumEdges);
        num_of_neighbors[0] = 1;
        max_degree = max_degree.max(offsets[n] - offsets[n - 1]);

        for Edge(u, v) in temp_edges {
            let addr_u = offsets[u as usize] as usize + num_of_neighbors[u as usize] as usize;
            let addr_v = offsets[v as usize] as usize + num_of_neighbors[v as usize] as usize;

            edges[addr_u] = NodeWithCrossPos {
                node: v,
                cross_pos: num_of_neighbors[v as usize],
            };
            edges[addr_v] = NodeWithCrossPos {
                node: u,
                cross_pos: num_of_neighbors[u as usize],
            };

            num_of_neighbors[u as usize] += 1;
            num_of_neighbors[v as usize] += 1;
        }

        Self {
            neighborhoods: SlicedBuffer::new(edges, offsets),
            max_degree: max_degree as NumNodes,
        }
    }
}

impl SelfLoop for CsrGraph {}

impl_static_graph_tests!(CsrGraph);

/// Basic Csr-Representation only stores neighbors of nodes.
pub type CsrEdges = SlicedBuffer<Node>;
