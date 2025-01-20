use super::*;

#[derive(Debug, Clone)]
pub struct CsrGraph {
    edges: Vec<Node>,
    offsets: Vec<NumEdges>,
    cross_references: Vec<NumNodes>,

    max_degree: NumNodes,
}

impl CsrGraph {
    #[inline(always)]
    fn offset_range(&self, u: Node) -> Range<usize> {
        (self.offsets[u as usize] as usize)..(self.offsets[(u + 1) as usize] as usize)
    }

    #[inline(always)]
    fn neighbor_slice(&self, u: Node) -> &[Node] {
        &self.edges[self.offset_range(u)]
    }

    #[inline(always)]
    fn cross_pos_slice(&self, u: Node) -> &[NumNodes] {
        &self.cross_references[self.offset_range(u)]
    }

    #[inline(always)]
    fn neighbor_slice_mut(&mut self, u: Node) -> &mut [Node] {
        let range = self.offset_range(u);
        &mut self.edges[range]
    }

    #[inline(always)]
    fn cross_pos_slice_mut(&mut self, u: Node) -> &mut [NumNodes] {
        let range = self.offset_range(u);
        &mut self.cross_references[range]
    }
}

impl GraphNodeOrder for CsrGraph {
    type VertexIter<'a> = impl Iterator<Item = Node> + 'a;

    fn number_of_nodes(&self) -> NumNodes {
        (self.offsets.len() - 1) as NumNodes
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
        self.edges.len() as NumEdges - self.number_of_nodes() as NumEdges
    }
}

impl AdjacencyList for CsrGraph {
    type NeighborIter<'a>
        = impl Iterator<Item = Node> + 'a
    where
        Self: 'a;

    fn neighbors_of(&self, u: Node) -> Self::NeighborIter<'_> {
        self.edges[self.offset_range(u)].iter().copied()
    }

    fn degree_of(&self, u: Node) -> NumNodes {
        (self.offsets[(u + 1) as usize] - self.offsets[u as usize]) as NumNodes
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
        self.edges[self.offset_range(u)].contains(&v)
    }
}

impl IndexedAdjacencyList for CsrGraph {
    fn ith_neighbor(&self, u: Node, i: NumNodes) -> Node {
        self.neighbor_slice(u)[i as usize]
    }

    fn ith_cross_position(&self, u: Node, i: NumNodes) -> NumNodes {
        self.cross_pos_slice(u)[i as usize]
    }

    fn swap_neighbors(&mut self, u: Node, nb1_pos: NumNodes, nb2_pos: NumNodes) {
        let nb1 = self.ith_neighbor(u, nb1_pos);
        let nb2 = self.ith_neighbor(u, nb2_pos);

        let nb1_cross_pos = self.ith_cross_position(u, nb1_pos);
        let nb2_cross_pos = self.ith_cross_position(u, nb2_pos);

        debug_assert_eq!(self.ith_neighbor(nb1, nb1_cross_pos), u);
        debug_assert_eq!(self.ith_neighbor(nb2, nb2_cross_pos), u);

        debug_assert_eq!(self.ith_cross_position(nb1, nb1_cross_pos), nb1_pos);
        debug_assert_eq!(self.ith_cross_position(nb2, nb2_cross_pos), nb2_pos);

        self.cross_pos_slice_mut(nb1)[nb1_cross_pos as usize] = nb2_pos;
        self.cross_pos_slice_mut(nb2)[nb2_cross_pos as usize] = nb1_pos;

        self.cross_pos_slice_mut(u)
            .swap(nb1_pos as usize, nb2_pos as usize);
        self.neighbor_slice_mut(u)
            .swap(nb1_pos as usize, nb2_pos as usize);
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
        let mut cross_references = vec![0; m];
        let mut edges = vec![0; m];

        offsets.push(0);

        let mut running_offset = num_of_neighbors[0] + 1;
        for u in 1..n {
            offsets.push(running_offset as NumEdges);
            edges[running_offset as usize] = u as Node;

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

            edges[addr_u] = v;
            edges[addr_v] = u;

            cross_references[addr_u] = num_of_neighbors[v as usize];
            cross_references[addr_v] = num_of_neighbors[u as usize];

            num_of_neighbors[u as usize] += 1;
            num_of_neighbors[v as usize] += 1;
        }

        Self {
            max_degree: max_degree as NumNodes,
            edges,
            offsets,
            cross_references,
        }
    }
}

impl CsrEdgeList for CsrGraph {
    fn get_csr_edges(&self) -> (Vec<Node>, Vec<NumEdges>) {
        (self.edges.clone(), self.offsets.clone())
    }
}

impl SelfLoop for CsrGraph {}
