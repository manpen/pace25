use super::*;
use std::fmt;
use std::ops::Range;

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
    type NeighborIter<'a> = impl Iterator<Item = Node> + 'a
    where
        Self: 'a;

    forward!(degree_of, degree, NumNodes);
    forward!(neighbors_of, neighbors, Self::NeighborIter<'_>);
}

impl ColoredAdjacencyList for AdjArray {
    type BlackNeighborIter<'a> = impl Iterator<Item = Node> + 'a
    where
        Self: 'a;

    type RedNeighborIter<'a> = impl Iterator<Item = Node> + 'a
        where
            Self: 'a;

    forward!(black_degree_of, black_degree, NumNodes);
    forward!(red_degree_of, red_degree, NumNodes);
    forward!(
        black_neighbors_of,
        black_neighbors,
        Self::BlackNeighborIter<'_>
    );
    forward!(red_neighbors_of, red_neighbors, Self::RedNeighborIter<'_>);
}

impl AdjacencyTest for AdjArray {
    fn has_edge(&self, u: Node, v: Node) -> bool {
        self.adj[u as usize].has_neighbor(v)
    }
}

impl ColoredAdjacencyTest for AdjArray {
    fn has_black_edge(&self, u: Node, v: Node) -> bool {
        self.adj[u as usize].has_black_neighbor(v)
    }

    fn has_red_edge(&self, u: Node, v: Node) -> bool {
        self.adj[u as usize].has_red_neighbor(v)
    }

    fn type_of_edge(&self, u: Node, v: Node) -> EdgeKind {
        self.adj[u as usize].edge_type_with(v)
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

    fn merge_node_into(&mut self, removed: Node, survivor: Node) {
        assert_ne!(removed, survivor);

        let reds = self.red_neighbors_after_merge(removed, survivor, true);

        for red_neigh in reds.iter() {
            self.try_add_edge(survivor, red_neigh as Node, EdgeColor::Red);
        }

        debug_assert!(!self.has_edge(survivor, survivor));

        self.remove_edges_at_node(removed);
    }

    fn red_neighbors_after_merge(&self, removed: Node, survivor: Node, only_new: bool) -> BitSet {
        let mut turned_red =
            BitSet::new_all_unset_but(self.number_of_nodes(), self.black_neighbors_of(survivor));

        for v in self.black_neighbors_of(removed) {
            if turned_red.set_bit(v) {
                // flip bit!
                turned_red.unset_bit(v);
            }
        }
        turned_red.set_bits(self.red_neighbors_of(removed));

        if only_new {
            turned_red.unset_bits(self.red_neighbors_of(survivor));
        } else {
            turned_red.set_bits(self.red_neighbors_of(survivor));
        }

        turned_red.unset_bit(removed);
        turned_red.unset_bit(survivor);

        turned_red
    }
}

impl AdjArray {
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

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use itertools::Itertools;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    use super::*;

    #[test]
    fn new() {
        for n in 1..50 {
            let graph = AdjArray::new(n);

            assert_eq!(graph.number_of_edges(), 0);
            assert_eq!(graph.number_of_nodes(), n);

            assert_eq!(graph.vertices_range().len(), n as usize);
            assert_eq!(graph.vertices().collect_vec(), (0..n).collect_vec());
        }
    }

    fn get_random_graph(rng: &mut impl Rng, n: NumNodes, m: NumEdges) -> AdjArray {
        let mut graph = AdjArray::new(n);

        while graph.number_of_edges() < m {
            let u = rng.gen_range(0..n);
            let v = rng.gen_range(0..n);
            if Edge(u, v).is_loop() {
                continue;
            }

            let color = if rng.gen_bool(0.5) {
                EdgeColor::Black
            } else {
                EdgeColor::Red
            };

            if !graph.try_add_edge(u, v, color).is_none() {
                graph.remove_edge(u, v);
            }
        }

        graph
    }

    #[test]
    fn neighborhoods() {
        let mut rng = Pcg64::seed_from_u64(123345);

        for _ in 0..100 {
            let n = rng.gen_range(5..50);
            let m = rng.gen_range(1..(n * (n - 1) / 4)) as NumEdges;

            let graph = get_random_graph(&mut rng, n, m);
            assert_eq!(
                graph.degrees().map(|x| x as NumEdges).sum::<NumEdges>(),
                2 * m
            );

            for u in 0..n {
                let mut red_black = graph
                    .red_neighbors_of(u)
                    .chain(graph.black_neighbors_of(u))
                    .collect_vec();

                red_black.sort();

                let mut neighs = graph.neighbors_of(u).collect_vec();
                neighs.sort();

                assert_eq!(neighs.len(), graph.degree_of(u) as usize);
                assert_eq!(red_black, neighs);
            }
        }
    }

    #[test]
    fn random_add() {
        let mut rng = Pcg64::seed_from_u64(1235);

        for n in 5..50 {
            let mut graph = AdjArray::new(n);
            let num_edges = rng.gen_range(1..(n * (n - 1) / 4)) as NumEdges;
            let mut edges: Vec<_> = Vec::with_capacity(2 * num_edges as usize);
            let mut edges_hash = HashSet::with_capacity(2 * num_edges as usize);

            let mut black_degrees = vec![0; n as usize];
            let mut red_degrees = vec![0; n as usize];

            for m in 0..num_edges {
                assert_eq!(graph.number_of_edges(), m);

                let (u, v) = loop {
                    let u = rng.gen_range(0..n);
                    let v = rng.gen_range(0..n);
                    if Edge(u, v).is_loop() {
                        continue;
                    }

                    if edges_hash.insert(Edge(u, v).normalized()) {
                        break (u, v);
                    }
                };

                let color = if rng.gen_bool(0.5) {
                    EdgeColor::Black
                } else {
                    EdgeColor::Red
                };

                graph.add_edge(u, v, color);

                if color.is_black() {
                    black_degrees[u as usize] += 1;
                    black_degrees[v as usize] += 1;
                } else {
                    red_degrees[u as usize] += 1;
                    red_degrees[v as usize] += 1;
                }

                // check edge iterators
                edges.push(ColoredEdge(u, v, color));
                edges.push(ColoredEdge(v, u, color));
                edges.sort();

                let mut graph_edges = graph.unordered_colored_edges().collect_vec();
                graph_edges.sort();

                assert_eq!(graph_edges, edges);

                // check degrees
                assert_eq!(graph.red_degrees().collect_vec(), red_degrees);
                assert_eq!(graph.black_degrees().collect_vec(), black_degrees);
                assert_eq!(
                    graph.degrees().collect_vec(),
                    black_degrees
                        .iter()
                        .zip(&red_degrees)
                        .map(|(&b, &r)| b + r)
                        .collect_vec()
                );
            }
        }
    }

    #[test]
    fn delete_edges_at_node() {
        let mut graph = AdjArray::new(4);
        graph.add_edges([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)], EdgeColor::Black);

        assert_eq!(graph.number_of_edges(), 5);
        assert_eq!(graph.degrees().collect_vec(), [3, 2, 3, 2]);

        graph.remove_edges_at_node(1);

        assert_eq!(graph.number_of_edges(), 3);
        assert_eq!(graph.degrees().collect_vec(), [2, 0, 2, 2]);
    }

    #[test]
    fn recoloring_insert() {
        let mut path = AdjArray::new(3);
        path.add_edges([(0, 1), (1, 2)], EdgeColor::Black);

        assert_eq!(path.red_degrees().collect_vec(), [0, 0, 0]);
        assert_eq!(path.black_degrees().collect_vec(), [1, 2, 1]);

        assert!(path.try_add_edge(0, 1, EdgeColor::Red).is_black());
        assert_eq!(path.red_degrees().collect_vec(), [1, 1, 0]);
        assert_eq!(path.black_degrees().collect_vec(), [0, 1, 1]);

        assert!(path.try_add_edge(1, 2, EdgeColor::Red).is_black());
        assert_eq!(path.red_degrees().collect_vec(), [1, 2, 1]);
        assert_eq!(path.black_degrees().collect_vec(), [0, 0, 0]);

        assert!(path.try_add_edge(1, 2, EdgeColor::Black).is_red());
        assert_eq!(path.red_degrees().collect_vec(), [1, 1, 0]);
        assert_eq!(path.black_degrees().collect_vec(), [0, 1, 1]);

        assert!(path.try_add_edge(0, 2, EdgeColor::Red).is_none());
        assert_eq!(path.red_degrees().collect_vec(), [2, 1, 1]);
        assert_eq!(path.black_degrees().collect_vec(), [0, 1, 1]);

        assert!(path.try_add_edge(0, 1, EdgeColor::Black).is_red());
        assert_eq!(path.red_degrees().collect_vec(), [1, 0, 1]);
        assert_eq!(path.black_degrees().collect_vec(), [1, 2, 1]);

        assert!(path.try_add_edge(2, 0, EdgeColor::Black).is_red());
        assert_eq!(path.red_degrees().collect_vec(), [0, 0, 0]);
        assert_eq!(path.black_degrees().collect_vec(), [2, 2, 2]);

        assert_eq!(path.number_of_edges(), 3);
    }

    #[test]
    fn merge() {
        let mut path = AdjArray::new(3);
        path.add_edges([(0, 1), (1, 2)], EdgeColor::Black);

        {
            let mut path = path.clone();
            path.merge_node_into(0, 1);

            assert_eq!(path.number_of_edges(), 1);
            assert_eq!(path.red_degrees().collect_vec(), [0, 1, 1]);
            assert_eq!(path.black_degrees().collect_vec(), [0, 0, 0]);

            assert_eq!(
                path.unordered_colored_edges().next().unwrap(),
                ColoredEdge(1, 2, EdgeColor::Red)
            );
        }

        {
            let mut path = path.clone();
            path.merge_node_into(0, 2);

            assert_eq!(path.number_of_edges(), 1);
            assert_eq!(path.red_degrees().collect_vec(), [0, 0, 0]);
            assert_eq!(path.black_degrees().collect_vec(), [0, 1, 1]);

            assert_eq!(
                path.unordered_colored_edges().next().unwrap(),
                ColoredEdge(1, 2, EdgeColor::Black)
            );
        }
    }

    #[test]
    fn loops() {
        let mut graph = AdjArray::new(1);

        assert!(graph.try_add_edge(0, 0, EdgeColor::Black).is_none());
        assert!(graph.try_add_edge(0, 0, EdgeColor::Red).is_black());
        assert!(graph.try_add_edge(0, 0, EdgeColor::Black).is_red());
        assert!(graph.try_remove_edge(0, 0).is_black());
        assert!(graph.try_remove_edge(0, 0).is_none());
    }
}
