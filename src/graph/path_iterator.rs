use super::*;

pub trait PathIterator: AdjacencyList {
    /// Returns an iterator to induced subpath in a graph.
    /// An induced subpath consists of an adjacent chain of k degree 2 node(s) with k > 0.
    /// A path is returned as a Vec where the first and last elements are the endpoints,
    /// which are of degree 2 iff the path is an induced cycle.
    fn path_iter(&self) -> Paths<'_, Self>;
}

impl<G: AdjacencyList> PathIterator for G {
    fn path_iter(&self) -> Paths<'_, Self> {
        Paths::new(self)
    }
}

pub struct Paths<'a, G: AdjacencyList> {
    graph: &'a G,
    visited: BitSet,
    search_at: Node,
}

impl<'a, G: AdjacencyList> Paths<'a, G> {
    fn new(graph: &'a G) -> Self {
        Self {
            graph,
            visited: graph.vertex_bitset_unset(),
            search_at: 0,
        }
    }

    fn complete_path(&mut self, u: Node, parent: Node, path: &mut Vec<Node>) {
        if self.graph.degree_of(u) != 2 || self.visited.set_bit(u) {
            path.push(u);
            return;
        }

        if let Some((n1, n2)) = self.graph.neighbors_of(u).collect_tuple() {
            path.push(u);
            if n1 != parent {
                self.complete_path(n1, u, path);
            }
            if n2 != parent {
                self.complete_path(n2, u, path);
            }
        }
    }
}

impl<'a, G: AdjacencyList> Iterator for Paths<'a, G> {
    type Item = Vec<Node>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut path = Vec::with_capacity(32);
        while self.search_at < self.graph.number_of_nodes() {
            let start_node = self.search_at;
            self.search_at += 1;

            if self.graph.degree_of(start_node) != 2 || self.visited.set_bit(start_node) {
                continue;
            }

            let (n1, n2) = self.graph.neighbors_of(start_node).collect_tuple().unwrap();
            path.push(start_node);
            self.complete_path(n1, start_node, &mut path);
            path.reverse();

            if path.len() == 1 || path[0] != start_node {
                // we need the check to properly deal with circles
                self.complete_path(n2, start_node, &mut path);
            }

            return Some(path);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, seq::SliceRandom};
    use rand_pcg::Pcg64Mcg;

    use super::*;

    fn match_or_reverse(a: &[Node], b: &[Node]) -> bool {
        if a == b {
            return true;
        }
        let mut c = Vec::from(b);
        c.reverse();
        a == c
    }

    #[test]
    fn single_path() {
        let graph = AdjArray::test_only_from([(0, 3), (3, 2), (2, 1)]);
        let paths = graph.path_iter().collect_vec();
        assert_eq!(paths.len(), 1);
        assert!(
            paths[0] == vec![0, 3, 2, 1] || paths[0] == vec![1, 2, 3, 0],
            "paths: {paths:?}"
        );
    }

    #[test]
    fn single_path_high_degree() {
        let graph = AdjArray::test_only_from([(0, 3), (3, 2), (2, 1), (0, 4), (0, 5)]);
        let paths = graph.path_iter().collect_vec();
        assert_eq!(paths.len(), 1);
        assert!(
            paths[0] == vec![0, 3, 2, 1] || paths[0] == vec![1, 2, 3, 0],
            "paths: {paths:?}"
        );
    }

    #[test]
    fn starlike() {
        let graph = AdjArray::test_only_from([
            (0, 1),
            (1, 2),
            (0, 3),
            (3, 4),
            (4, 5),
            (0, 6),
            (6, 7),
            (7, 8),
            (8, 9),
        ]);
        let mut paths = graph.path_iter().collect_vec();
        assert_eq!(paths.len(), 3);
        paths.sort_by_key(|x| x.len());

        assert!(match_or_reverse(&paths[0], &[0, 1, 2]));
        assert!(match_or_reverse(&paths[1], &[0, 3, 4, 5]));
        assert!(match_or_reverse(&paths[2], &[0, 6, 7, 8, 9]));
    }

    #[test]
    fn single_circle() {
        let graph = AdjArray::test_only_from([(0, 3), (3, 2), (2, 1), (1, 0)]);
        let paths = graph.path_iter().collect_vec();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 5, "{paths:?}");
        assert_eq!(paths[0][0], paths[0][4], "{paths:?}");
    }

    #[test]
    fn randomized_single_path() {
        let mut rng = Pcg64Mcg::seed_from_u64(12345);
        for _ in 0..1000 {
            let n = rng.gen_range(3..100);
            let mut order = (0..n as Node).collect_vec();
            order.shuffle(&mut rng);

            let edges = (0..n - 1).map(|i| (order[i], order[i + 1])).collect_vec();
            let graph = AdjArray::from_edges(n as Node, &edges);
            let paths = graph.path_iter().collect_vec();
            assert_eq!(paths.len(), 1, "{order:?}");

            assert!(match_or_reverse(&paths[0], &order));
        }
    }

    #[test]
    fn randomized_two_path() {
        let mut rng = Pcg64Mcg::seed_from_u64(12345);
        for _ in 0..1000 {
            let n = rng.gen_range(6..100);
            let n1 = rng.gen_range(3..n - 2);
            if 2 * n1 == n {
                // if both subproblems have the same length, we would need addtional logic to find the correct path order; so ignore that case
                continue;
            }
            let mut order = (0..n as Node).collect_vec();
            order.shuffle(&mut rng);

            let edges = (0..n - 1)
                .filter_map(|i| (i + 1 != n1).then_some((order[i], order[i + 1])))
                .collect_vec();
            let graph = AdjArray::from_edges(n as Node, &edges);
            let mut paths = graph.path_iter().collect_vec();
            assert_eq!(paths.len(), 2, "{order:?}");

            if paths[0].len() != n1 {
                paths.reverse();
            }
            assert_eq!(paths[0].len(), n1, "{paths:?}");
            assert_eq!(paths[1].len(), n - n1, "{paths:?}");

            assert!(
                match_or_reverse(&paths[0], &order[..n1]),
                "{paths:?} {n1} {order:?}"
            );
            assert!(
                match_or_reverse(&paths[1], &order[n1..]),
                "{paths:?} {n1} {order:?}"
            );
        }
    }
}
