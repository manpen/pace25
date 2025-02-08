#[macro_export]
macro_rules! impl_graph_tests {
    ($graph : ty) => {
        #[cfg(test)]
        mod graph_tests {
            use std::collections::HashSet;

            use itertools::Itertools;
            use rand::{Rng, SeedableRng};
            use rand_pcg::Pcg64;

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

            fn get_random_graph(rng: &mut impl Rng, n: NumNodes, m: NumEdges) -> $graph {
                let mut graph = <$graph>::new(n);

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
                    let mut graph = <$graph>::new(n);
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
                let mut graph = <$graph>::new(4);
                graph.add_edges([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)], EdgeColor::Black);

                assert_eq!(graph.number_of_edges(), 5);
                assert_eq!(graph.degrees().collect_vec(), [3, 2, 3, 2]);

                graph.remove_edges_at_node(1);

                assert_eq!(graph.number_of_edges(), 3);
                assert_eq!(graph.degrees().collect_vec(), [2, 0, 2, 2]);
            }

            #[test]
            fn recoloring_insert() {
                let mut path = <$graph>::new(3);
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
                let mut path = <$graph>::new(3);
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
                let mut graph = <$graph>::new(1);

                assert!(graph.try_add_edge(0, 0, EdgeColor::Black).is_none());
                assert!(graph.try_add_edge(0, 0, EdgeColor::Red).is_black());
                assert!(graph.try_add_edge(0, 0, EdgeColor::Black).is_red());
                assert!(graph.try_remove_edge(0, 0).is_black());
                assert!(graph.try_remove_edge(0, 0).is_none());
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

            //#[test]
            //fn sorted_neighbors() {
            //    let mut rng = Pcg64::seed_from_u64(123345);
            //
            //    for _ in 0..100 {
            //        let n = rng.gen_range(5..50);
            //        let m = rng.gen_range(1..(n * (n - 1) / 4)) as NumEdges;
            //
            //        let mut graph = get_random_graph(&mut rng, n, m);
            //
            //        graph.sort_all_neighbors();
            //        assert!(graph.are_all_neighbors_sorted());
            //    }
            //}
        }
    };
}
