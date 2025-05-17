use super::*;
use crate::utils::DominatingSet;
use itertools::Itertools;

pub trait CuthillMcKee {
    /// Computes a node label mapping intended to minimize the bandwidth of the
    /// graph's adjacency matrix. This can might improve performance down the
    /// road as it may reduce the cache misses when working with the algorithm.
    /// The algorithm does not map singleton nodes (degree = 0)
    fn cuthill_mckee(&self) -> NodeMapper;
}

impl<G: AdjacencyList> CuthillMcKee for G {
    fn cuthill_mckee(&self) -> NodeMapper {
        let mut mapper = NodeMapper::with_capacity(self.number_of_nodes());
        let mut queue = Vec::with_capacity(self.len());

        // find a start node of smallest positive degree; if none exists, all nodes
        // are singletons and we exit earlier.
        let start_node = match self
            .vertices()
            .filter(|u| self.degree_of(*u) > 0)
            .map(|u| (self.degree_of(u), u))
            .min()
        {
            Some((_deg, node)) => node,
            None => return mapper,
        };
        queue.push(start_node);
        mapper.map_node_to(start_node, 0);

        // We reuse the set-functionality of DominatingSet as we need a fast iterator
        let mut candidates = DominatingSet::complete_set(self.number_of_nodes());
        candidates.remove_nodes(self.vertices_range().filter(|u| self.degree_of(*u) == 0));
        let non_isolated_nodes = candidates.len();

        let mut i = 0usize;
        loop {
            if queue.len() >= non_isolated_nodes {
                break;
            }

            if i >= queue.len() {
                let (_deg, new_min_node) = candidates
                    .iter()
                    .filter(|&u| mapper.new_id_of(u).is_none())
                    .map(|u| (self.degree_of(u), u))
                    .min()
                    .unwrap();

                candidates.remove_node(new_min_node);
                mapper.map_node_to(new_min_node, queue.len() as Node);
                queue.push(new_min_node);
            }

            let mut adj = self
                .neighbors_of(queue[i])
                .filter(|&u| mapper.new_id_of(u).is_none())
                .map(|u| (self.degree_of(u), u))
                .collect_vec();
            adj.sort_unstable();

            for (_deg, u) in adj {
                mapper.map_node_to(u, queue.len() as Node);
                queue.push(u);
                if candidates.is_in_domset(u) {
                    candidates.remove_node(u);
                }
            }

            i += 1;
        }

        mapper
    }
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    use super::*;
    use crate::graph::{AdjArray, Getter, GnpGenerator, GraphFromReader};

    #[test]
    fn small_toy() {
        let graph = AdjArray::test_only_from([(1, 2), (1, 0), (0, 0)]);
        let mapping = graph.cuthill_mckee();
        assert_eq!(mapping.len(), 3);
        assert_eq!(mapping.new_id_of(0), Some(2));
        assert_eq!(mapping.new_id_of(1), Some(1));
        assert_eq!(mapping.new_id_of(2), Some(0));
    }

    #[test]
    fn small_toy_with_singletons() {
        let graph = AdjArray::from_edges(5, [(2, 3), (2, 1), (1, 1)]);
        let mapping = graph.cuthill_mckee();
        assert_eq!(mapping.len(), 3);
        assert!(mapping.new_id_of(0).is_none());
        assert_eq!(mapping.new_id_of(1), Some(2));
        assert_eq!(mapping.new_id_of(2), Some(1));
        assert_eq!(mapping.new_id_of(3), Some(0));
        assert!(mapping.new_id_of(4).is_none());
    }

    #[test]
    fn randomized() {
        let mut rng = Pcg64Mcg::seed_from_u64(123456);
        for n in 1..100 {
            let p = rng.gen_range(0.5..(10.min(n) as f64)) / (n as f64);
            let graph = AdjArray::random_black_gnp(&mut rng, n, p);
            let mapping = graph.cuthill_mckee();
            assert_eq!(
                mapping.len(),
                graph.vertices_with_neighbors().count() as Node
            );
        }
    }
}
