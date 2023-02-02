use std::{borrow::Borrow, collections::HashSet, ops::Range};

use itertools::Itertools;

use super::*;

#[derive(Clone)]
pub struct AdjList {
    adj: Vec<Neighborhood>,
    number_of_edges: NumEdges,
}

macro_rules! forward {
    ($single : ident, $internal : ident, $type : ty) => {
        pub fn $single(&self, node: Node) -> $type {
            self.adj[node as usize].$internal()
        }
    };
}

macro_rules! node_iterator {
    ($iter : ident, $single : ident, $type : ty) => {
        pub fn $iter(&self) -> impl Iterator<Item = $type> + '_ {
            self.nodes().map(|u| self.$single(u))
        }
    };
}

macro_rules! forward_with_iter {
    ($single : ident, $iter : ident, $internal : ident, $type : ty) => {
        forward!($single, $internal, $type);
        node_iterator!($iter, $single, $type);
    };
}

impl AdjList {
    pub fn new(number_of_nodes: NumNodes) -> Self {
        Self {
            adj: vec![Default::default(); number_of_nodes as usize],
            number_of_edges: 0,
        }
    }

    pub fn number_of_nodes(&self) -> NumNodes {
        self.adj.len() as NumNodes
    }

    pub fn number_of_edges(&self) -> NumEdges {
        self.number_of_edges
    }

    pub fn nodes_range(&self) -> Range<Node> {
        0..self.number_of_nodes()
    }

    pub fn nodes(&self) -> impl Iterator<Item = Node> {
        self.nodes_range()
    }

    pub fn unordered_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.nodes()
            .flat_map(|u| self.neighbors_of(u).iter().map(move |&v| Edge(u, v)))
    }

    pub fn unordered_colored_edges(&self) -> impl Iterator<Item = ColoredEdge> + '_ {
        self.nodes().flat_map(|u| {
            self.black_neighbors_of(u)
                .iter()
                .map(move |&v| ColoredEdge(u, v, EdgeColor::Black))
                .chain(
                    self.red_neighbors_of(u)
                        .iter()
                        .map(move |&v| ColoredEdge(u, v, EdgeColor::Red)),
                )
        })
    }

    forward_with_iter!(degree_of, degrees, degree, NumNodes);
    forward_with_iter!(black_degree_of, black_degrees, black_degree, NumNodes);
    forward_with_iter!(red_degree_of, red_degrees, red_degree, NumNodes);
    forward_with_iter!(neighbors_of, neighbors, neighbors, &[Node]);
    forward_with_iter!(
        black_neighbors_of,
        black_neighbors,
        black_neighbors,
        &[Node]
    );
    forward_with_iter!(red_neighbors_of, red_neighbors, red_neighbors, &[Node]);

    pub fn has_neighbor(&self, u: Node, v: Node) -> bool {
        self.adj[u as usize].has_neighbor(v)
    }

    pub fn has_black_neighbor(&self, u: Node, v: Node) -> bool {
        self.adj[u as usize].has_black_neighbor(v)
    }

    pub fn has_red_neighbor(&self, u: Node, v: Node) -> bool {
        self.adj[u as usize].has_red_neighbor(v)
    }

    pub fn edge_type(&self, u: Node, v: Node) -> EdgeKind {
        self.adj[u as usize].edge_type_with(v)
    }

    /// Inserts the colored edge as specified and returns the previous state of edge
    pub fn try_add_edge(&mut self, u: Node, v: Node, color: EdgeColor) -> EdgeKind {
        let prev = self.adj[u as usize].try_add_edge(v, color);

        if prev != color {
            assert_eq!(self.adj[v as usize].try_add_edge(u, color), prev)
        }

        if prev == EdgeKind::None {
            self.number_of_edges += 1;
        }

        prev
    }

    /// Inserts the colored edge as specified. Panics if already exists
    pub fn add_edge(&mut self, u: Node, v: Node, kind: EdgeColor) {
        assert_eq!(self.try_add_edge(u, v, kind), EdgeKind::None)
    }

    pub fn add_edges(
        &mut self,
        edges: impl IntoIterator<Item = impl Into<Edge>>,
        color: EdgeColor,
    ) {
        for Edge(u, v) in edges.into_iter().map(|d| d.into()) {
            self.add_edge(u, v, color);
        }
    }

    pub fn add_colored_edges<I>(
        &mut self,
        edges: impl IntoIterator<Item = impl Borrow<ColoredEdge>>,
    ) {
        for ColoredEdge(u, v, color) in edges.into_iter().map(|d| *d.borrow()) {
            self.add_edge(u, v, color);
        }
    }

    pub fn try_delete_edge(&mut self, u: Node, v: Node) -> EdgeKind {
        let prev = self.adj[u as usize].try_delete_edge(v);

        if prev.is_some() {
            let _other = self.adj[v as usize].try_delete_edge(u);
            debug_assert_eq!(prev, _other);
        }

        self.number_of_edges -= 1;

        prev
    }

    pub fn delete_edge(&mut self, u: Node, v: Node) {
        assert!(self.try_delete_edge(u, v).is_some())
    }

    pub fn delete_edges_at_node(&mut self, u: Node) {
        let neighbors = std::mem::take(&mut self.adj[u as usize]);
        self.number_of_edges -= neighbors.nodes.len() as NumEdges;

        for &v in &neighbors.nodes {
            assert!(self.adj[v as usize].try_delete_edge(u).is_some());
        }
    }

    pub fn merge_node_into(&mut self, removed: Node, survivor: Node) {
        assert_ne!(removed, survivor);

        let black_rem: HashSet<Node> = self.black_neighbors_of(removed).iter().copied().collect();
        let black_sur: HashSet<Node> = self.black_neighbors_of(survivor).iter().copied().collect();
        let turned_red = black_rem.symmetric_difference(&black_sur);

        let reds = self
            .red_neighbors_of(survivor)
            .iter()
            .copied()
            .collect_vec();

        for &red_neigh in reds.iter().chain(turned_red) {
            if red_neigh == survivor {
                continue;
            }
            self.try_add_edge(survivor, red_neigh, EdgeColor::Red);
        }

        self.delete_edges_at_node(removed);
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

    fn neighbors(&self) -> &[Node] {
        &self.nodes
    }

    fn black_neighbors(&self) -> &[Node] {
        &self.nodes[0..self.black_degree() as usize]
    }

    fn red_neighbors(&self) -> &[Node] {
        &self.nodes[self.black_degree() as usize..]
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
        self.neighbors().iter().any(|&u| u == v)
    }

    fn has_black_neighbor(&self, v: Node) -> bool {
        self.black_neighbors().iter().any(|&u| u == v)
    }

    fn has_red_neighbor(&self, v: Node) -> bool {
        self.red_neighbors().iter().any(|&u| u == v)
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
        let position = self.neighbors().iter().position(|&x| x == v);

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
            let graph = AdjList::new(n);

            assert_eq!(graph.number_of_edges(), 0);
            assert_eq!(graph.number_of_nodes(), n);

            assert_eq!(graph.nodes_range().len(), n as usize);
            assert_eq!(
                graph.nodes().collect_vec(),
                (0..n).into_iter().collect_vec()
            );
        }
    }

    fn get_random_graph(rng: &mut impl Rng, n: NumNodes, m: NumEdges) -> AdjList {
        let mut graph = AdjList::new(n);

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
                graph.delete_edge(u, v);
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
                    .iter()
                    .chain(graph.black_neighbors_of(u))
                    .copied()
                    .collect_vec();

                red_black.sort();

                let mut neighs = graph.neighbors_of(u).iter().copied().collect_vec();
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
            let mut graph = AdjList::new(n);
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
        let mut graph = AdjList::new(4);
        graph.add_edges([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)], EdgeColor::Black);

        assert_eq!(graph.number_of_edges(), 5);
        assert_eq!(graph.degrees().collect_vec(), [3, 2, 3, 2]);

        graph.delete_edges_at_node(1);

        assert_eq!(graph.number_of_edges(), 3);
        assert_eq!(graph.degrees().collect_vec(), [2, 0, 2, 2]);
    }

    #[test]
    fn recoloring_insert() {
        let mut path = AdjList::new(3);
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
        let mut path = AdjList::new(3);
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
}
