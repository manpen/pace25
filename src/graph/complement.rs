use super::*;

pub trait Complement {
    /// Produces a copy of the graph where every red edge stays while
    /// black edges become non-edges and vice versa.
    fn trigraph_complement(&self) -> Self;
}

impl<G> Complement for G
where
    G: ColoredAdjacencyList + GraphNew + GraphEdgeEditing,
{
    fn trigraph_complement(&self) -> Self {
        let mut complement = Self::new(self.number_of_nodes());
        let mut mask = BitSet::new_all_set(self.number_of_nodes());

        for u in self.vertices() {
            let mut neighbors = self.neighbors_of_as_bitset(u);
            neighbors.not();
            mask.unset_bit(u);
            neighbors.and(&mask);

            complement.add_edges(neighbors.iter().map(|v| (u, v)), EdgeColor::Black);
            complement.add_edges(
                self.red_neighbors_of(u)
                    .iter()
                    .filter_map(|&v| (u < v).then_some((u, v))),
                EdgeColor::Red,
            );
        }

        complement
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    fn complement() {
        let mut rng = Pcg64::seed_from_u64(0x263741);

        for n in 0..50 {
            let graph = AdjArray::random_colored_gnp(&mut rng, n, 0.5, 0.5);
            let complement = graph.trigraph_complement();

            // identical red edges
            assert_eq!(
                graph.red_only().ordered_edges(true).collect_vec(),
                complement.red_only().ordered_edges(true).collect_vec(),
            );

            // black edges do not exist
            assert!(graph
                .black_only()
                .edges(true)
                .all(|Edge(u, v)| !complement.has_edge(u, v)));

            assert!(complement
                .black_only()
                .edges(true)
                .all(|Edge(u, v)| !graph.has_edge(u, v)));
        }
    }
}
