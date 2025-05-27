use super::*;

pub trait Complement: ColoredAdjacencyList + GraphNew + GraphEdgeEditing {
    /// Produces a copy of the graph where every red edge stays while
    /// black edges become non-edges and vice versa.
    fn trigraph_complement(&self, ignore_isolated: bool) -> Self {
        let mut complement = Self::new(self.number_of_nodes());
        let mut mask = BitSet::new_all_set(self.number_of_nodes());

        if ignore_isolated {
            mask.clear_bits(self.vertices().filter(|&u| self.degree_of(u) == 0));
        }

        for u in self.vertices() {
            if !mask.clear_bit(u) {
                continue;
            }

            let mut neighbors = self.neighbors_of_as_bitset(u);
            neighbors.flip_all();
            neighbors &= &mask;

            complement.add_edges(neighbors.iter_set_bits().map(|v| (u, v)), EdgeColor::Black);
            complement.add_edges(
                self.red_neighbors_of(u)
                    .filter_map(|v| (u < v).then_some((u, v))),
                EdgeColor::Red,
            );
        }

        complement
    }

    /// Returns the size of the complement graph without constructing it
    fn number_of_edges_in_trigraph_complement(&self, ignore_isolated: bool) -> NumEdges {
        let num_nodes = if ignore_isolated {
            self.vertices_with_neighbors().count() as NumEdges
        } else {
            self.number_of_nodes() as NumEdges
        };

        let total_edges = num_nodes * num_nodes.saturating_sub(1) / 2;

        let num_black_edges = self
            .black_degrees()
            .map(|d| d as NumEdges)
            .sum::<NumEdges>()
            / 2;

        total_edges - num_black_edges
    }
}

impl Complement for AdjArray {}

#[cfg(test)]
mod test {
    use super::*;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    fn complement_adj_array() {
        complement_for::<AdjArray>();
    }

    fn complement_for<G: Complement + FullfledgedGraph>() {
        let mut rng = Pcg64::seed_from_u64(0x263741);

        for p in [0.1, 0.5] {
            for ignore_isolated in [false, true] {
                for n in 0..50 {
                    let graph = G::random_colored_gnp(&mut rng, n, p, 0.5);
                    let complement = graph.trigraph_complement(ignore_isolated);

                    assert_eq!(
                        graph.number_of_edges_in_trigraph_complement(ignore_isolated),
                        complement.number_of_edges()
                    );

                    // identical red edges
                    assert_eq!(
                        graph
                            .ordered_colored_edges(true)
                            .filter(|&e| e.is_red())
                            .collect_vec(),
                        complement
                            .ordered_colored_edges(true)
                            .filter(|&e| e.is_red())
                            .collect_vec(),
                    );

                    // black edges do not exist
                    assert!(
                        graph
                            .colored_edges(true)
                            .filter(|&e| e.is_black())
                            .all(|ColoredEdge(u, v, _)| !complement.has_edge(u, v))
                    );

                    assert!(
                        complement
                            .colored_edges(true)
                            .filter(|&e| e.is_black())
                            .all(|ColoredEdge(u, v, _)| !graph.has_edge(u, v))
                    );
                }
            }
        }
    }
}
