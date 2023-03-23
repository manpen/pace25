use super::*;

pub trait BipartiteTest {
    fn is_bipartition(&self, candidate: &BitSet) -> bool;
    fn compute_bipartition(&self) -> Option<BitSet>;
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
            .all(|Edge(u, v)| candidate[u] != candidate[v])
    }

    fn compute_bipartition(&self) -> Option<BitSet> {
        let candidate = propose_possibly_illegal_bipartition(self);
        self.is_bipartition(&candidate).then_some(candidate)
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
            if !bipartition[pred] {
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
