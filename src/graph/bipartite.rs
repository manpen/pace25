use super::*;

pub trait BipartiteTest {
    /// Tests whether the given candidate partition is a valid bipartition.
    ///
    /// # Examples
    /// ```
    /// use dss::prelude::*;
    /// let mut graph = AdjArray::new(2);
    /// graph.add_edge(0, 1, EdgeColor::Black);
    ///
    /// assert!(graph.is_bipartition(&BitSet::new_with_bits_set(2, [0u32])));
    /// assert!(!graph.is_bipartition(&BitSet::new(2)));
    /// ```
    fn is_bipartition(&self, candidate: &BitSet) -> bool;

    /// Computes a valid bipartition of the graph, if one exists.
    ///
    /// # Examples
    /// ```
    /// use dss::prelude::*;
    /// let mut graph = AdjArray::new(4); // path graph
    /// graph.add_edges([(0, 1), (1, 2), (2, 3)], EdgeColor::Black);
    ///
    /// let partition = graph.compute_bipartition().unwrap();
    /// assert!(graph.is_bipartition(&partition));
    ///
    /// graph.add_edge(0, 2, EdgeColor::Black);
    /// assert!(graph.compute_bipartition().is_none());
    /// ```
    fn compute_bipartition(&self) -> Option<BitSet>;

    /// Tests whether the graph is bipartite.
    ///
    /// # Examples
    /// use dss::prelude::*;
    /// let mut graph = AdjArray::new(4); // path graph
    /// graph.add_edges([(0, 1), (1, 2), (2, 3)], EdgeColor::Black);
    ///
    /// assert!(graph.is_bipartite());
    ///
    /// graph.add_edge(0, 2, EdgeColor::Black);
    /// assert!(!graph.is_bipartite());
    /// ```
    fn is_bipartite(&self) -> bool {
        self.compute_bipartition().is_some()
    }
}

impl<G> BipartiteTest for G
where
    G: AdjacencyList,
{
    fn is_bipartition(&self, candidate: &BitSet) -> bool {
        self.edges(true)
            .all(|Edge(u, v)| candidate.get_bit(u) != candidate.get_bit(v))
    }

    fn compute_bipartition(&self) -> Option<BitSet> {
        let candidate = propose_possibly_illegal_bipartition(self);
        self.is_bipartition(&candidate).then_some(candidate)
    }
}

pub trait BipartiteEdit {
    /// Remove all edges that connect nodes in the same partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use dss::prelude::*;
    /// let mut graph = AdjArray::new(4);
    /// graph.add_edges([(0, 1), (1, 2), (2, 3)], EdgeColor::Black);
    /// let partition = BitSet::new_with_bits_set(4, [0u32, 2]);
    ///
    /// // no change, since it's actually bipartite
    /// graph.remove_edges_within_bipartition_class(&partition);
    /// assert_eq!(graph.number_of_edges(), 3);
    ///
    /// // add non-bipartite edges
    /// graph.add_edges([(0, 2), (1, 3)], EdgeColor::Black);
    /// assert_eq!(graph.number_of_edges(), 5);
    /// graph.remove_edges_within_bipartition_class(&partition);
    /// assert_eq!(graph.number_of_edges(), 3);
    /// ```
    fn remove_edges_within_bipartition_class(&mut self, bipartition: &BitSet);
}

impl<G> BipartiteEdit for G
where
    G: AdjacencyList + GraphEdgeEditing,
{
    fn remove_edges_within_bipartition_class(&mut self, bipartition: &BitSet) {
        let to_delete: Vec<_> = self
            .edges(true)
            .filter(|&Edge(u, v)| bipartition.get_bit(u) == bipartition.get_bit(v))
            .collect();
        self.remove_edges(to_delete);
    }
}

// Compute a bipartition of `graph` if `graph` is bipartite; otherwise an arbitrary
// partition is returned
fn propose_possibly_illegal_bipartition<G: AdjacencyList>(graph: &G) -> BitSet {
    let mut bfs = graph.bfs_with_predecessor(0);

    // propose a bipartition
    let mut bipartition = BitSet::new(graph.number_of_nodes());
    loop {
        for (node, pred) in bfs
            .by_ref()
            .filter_map(|x| Some((x.item(), x.predecessor()?)))
        {
            if !bipartition.get_bit(pred) {
                bipartition.set_bit(node);
            }
        }

        if !bfs.try_restart_at_unvisited() {
            break;
        }
    }

    bipartition
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn path() {
        for n in 1..10 {
            let mut graph = AdjArray::new(n);
            for u in 0..n - 1 {
                graph.add_edge(u, u + 1, EdgeColor::Black);
            }

            assert!(graph.compute_bipartition().is_some());

            if n > 2 {
                let mut graph = graph.clone();
                graph.remove_edge(n / 2, n / 2 + 1);
                assert!(graph.compute_bipartition().is_some());
            }

            if n > 2 {
                let mut graph = graph.clone();
                graph.add_edge(1 - (n % 2), n - 1, EdgeColor::Black);
                assert!(graph.compute_bipartition().is_none());
            }
        }
    }
}
