#[macro_export]
macro_rules! impl_graph_tests {
    ($graph : ty) => {
        #[cfg(test)]
        mod graph_tests {
            use itertools::Itertools;

            use super::*;

            #[test]
            fn new() {
                for n in 1..50 {
                    let graph = <$graph>::new(n);

                    assert_eq!(graph.number_of_edges(), 0);
                    assert_eq!(graph.number_of_nodes(), n);

                    assert_eq!(graph.vertices_range().len(), n as usize);
                    assert_eq!(graph.vertices().collect_vec(), (0..n).collect_vec());
                }
            }

            #[test]
            fn delete_edges_at_node() {
                let mut graph = <$graph>::new(4);
                graph.add_edges([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]);

                assert_eq!(graph.number_of_edges(), 5);
                assert_eq!(graph.degrees().collect_vec(), [3, 2, 3, 2]);

                graph.remove_edges_at_node(1);

                assert_eq!(graph.number_of_edges(), 3);
                assert_eq!(graph.degrees().collect_vec(), [2, 0, 2, 2]);
            }

            #[test]
            fn loops() {
                let mut graph = <$graph>::new(1);

                assert!(!graph.try_add_edge(0, 0));
                assert!(graph.try_add_edge(0, 0));
                assert!(graph.try_remove_edge(0, 0));
                assert!(!graph.try_remove_edge(0, 0));
            }
        }
    };
}

pub use impl_graph_tests;

#[macro_export]
macro_rules! impl_static_graph_tests {
    ($graph : ty) => {
        #[cfg(test)]
        mod static_graph_tests {
            use rand::{Rng, SeedableRng};
            use rand_pcg::Pcg64;

            use super::*;

            fn get_random_graph(rng: &mut impl Rng, n: NumNodes, m: NumEdges) -> $graph {
                let mut set = BitSet::new(n * n);
                let mut edges: Vec<Edge> = Vec::with_capacity(m as usize);
                while edges.len() < m as usize {
                    let u = rng.gen_range(0..n);
                    let v = rng.gen_range(0..n);
                    if Edge(u, v).is_loop() {
                        continue;
                    }

                    if !set.set_bit(u * v) {
                        edges.push(Edge(u, v));
                    }
                }

                <$graph>::from_edges(n, edges)
            }

            #[test]
            fn indexed_adjacency_list() {
                let mut rng = Pcg64::seed_from_u64(123345);

                for _ in 0..100 {
                    let n = rng.gen_range(5..50);
                    let m = rng.gen_range(1..(n * (n - 1) / 4)) as NumEdges;

                    let graph = get_random_graph(&mut rng, n, m);

                    for u in graph.vertices() {
                        for i in 0..graph.degree_of(u) {
                            assert_eq!(
                                graph.ith_neighbor(
                                    graph.ith_neighbor(u, i),
                                    graph.ith_cross_position(u, i)
                                ),
                                u
                            );
                        }
                    }
                }
            }
        }
    };
}
