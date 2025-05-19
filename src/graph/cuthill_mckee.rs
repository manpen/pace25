use super::*;
use crate::utils::DominatingSet;
use itertools::Itertools;

pub trait CuthillMcKee {
    /// Computes a node label mapping intended to minimize the bandwidth of the
    /// graph's adjacency matrix. This can might improve performance down the
    /// road as it may reduce the cache misses when working with the algorithm.
    /// The algorithm does not map singleton nodes (degree = 0)
    fn cuthill_mckee(&self) -> NodeMapper {
        self.cuthill_mckee_cc(false).0
    }

    /// Same as `cuthill_mckee`, but also returns a vector of all non-trivial
    /// connected components (i.e. ccs with at least 2 nodes) if requested.
    ///
    /// `return_ccs = false` implies that the vector returns is empty and of
    /// capacity 0.
    fn cuthill_mckee_cc(&self, return_ccs: bool) -> (NodeMapper, Vec<Range<Node>>);
}

impl<G: AdjacencyList> CuthillMcKee for G {
    fn cuthill_mckee_cc(&self, return_ccs: bool) -> (NodeMapper, Vec<Range<Node>>) {
        let mut ccs = Vec::with_capacity(return_ccs as usize);
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
            None => return (mapper, ccs),
        };
        queue.push(start_node);
        mapper.map_node_to(start_node, 0);

        // We reuse the set-functionality of DominatingSet as we need a fast iterator
        let mut candidates = DominatingSet::complete_set(self.number_of_nodes());
        candidates.remove_nodes(self.vertices_range().filter(|u| self.degree_of(*u) == 0));
        let non_isolated_nodes = candidates.len();

        let mut cc_start_node: Node = 0;
        let mut i = 0usize;
        while queue.len() < non_isolated_nodes {
            if i >= queue.len() {
                let (deg, new_min_node) = candidates
                    .iter()
                    .filter(|&u| mapper.new_id_of(u).is_none())
                    .map(|u| (self.degree_of(u), u))
                    .min()
                    .unwrap();

                debug_assert!(deg > 0);

                candidates.remove_node(new_min_node);
                mapper.map_node_to(new_min_node, queue.len() as Node);

                if return_ccs {
                    let cc = cc_start_node..(queue.len() as Node);
                    cc_start_node = cc.end;
                    assert!(cc.len() > 1);
                    ccs.push(cc);
                }

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

        if return_ccs {
            let cc = cc_start_node..(queue.len() as Node);
            if cc.len() > 1 {
                ccs.push(cc);
            }
        }

        (mapper, ccs)
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

    #[test]
    fn randomized_with_ccs() {
        let mut rng = Pcg64Mcg::seed_from_u64(1234567);
        for n in 1..100 {
            let p = rng.gen_range(0.5..(2.min(n) as f64)) / (n as f64);
            let graph = AdjArray::random_black_gnp(&mut rng, n, p);
            let (mapping, ccs) = graph.cuthill_mckee_cc(true);
            let partition = graph.partition_into_connected_components(true);

            assert_eq!(
                mapping.len(),
                graph.vertices_with_neighbors().count() as Node
            );

            assert_eq!(
                ccs.iter().map(|r| r.len() as Node).sum::<Node>(),
                mapping.len()
            );

            let mut cm_cc_sizes = ccs.into_iter().map(|cc| cc.len() as NumNodes).collect_vec();
            cm_cc_sizes.sort();

            let mut ref_cc_sizes = (0..partition.number_of_classes())
                .map(|c| partition.number_in_class(c))
                .collect_vec();
            ref_cc_sizes.sort();

            assert_eq!(cm_cc_sizes, ref_cc_sizes);
        }
    }
}
