use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Distance {
    Two = 2,
    Three = 3,
}

pub struct DistancePairsIterator<'a, G: AdjacencyList> {
    graph: &'a G,
    node: Node,
    distance: Distance,

    neighbors: BitSet,
    neighbor_lb: Node,
}

pub trait DistancePairs {
    type PairIterator<'a>: Iterator<Item = (Node, Node)> + 'a
    where
        Self: 'a;

    fn distance_two_pairs(&self) -> Self::PairIterator<'_>;
    fn distance_three_pairs(&self) -> Self::PairIterator<'_>;
}

impl<G: AdjacencyList> DistancePairs for G {
    type PairIterator<'a> = DistancePairsIterator<'a, Self> where Self: 'a;

    fn distance_two_pairs(&self) -> Self::PairIterator<'_> {
        DistancePairsIterator::new(self, Distance::Two)
    }

    fn distance_three_pairs(&self) -> Self::PairIterator<'_> {
        DistancePairsIterator::new(self, Distance::Three)
    }
}

impl<'a, G: AdjacencyList> DistancePairsIterator<'a, G> {
    pub fn new(graph: &'a G, distance: Distance) -> Self {
        let n = graph.number_of_nodes();

        let mut inst = Self {
            graph,
            node: 0,
            neighbors: BitSet::new(n),
            neighbor_lb: 0,
            distance,
        };

        inst.setup_node();

        inst
    }

    fn setup_node(&mut self) {
        self.neighbors.unset_all();

        for &v in self.graph.neighbors_of(self.node) {
            self.neighbors
                .set_bits(self.graph.neighbors_of(v).iter().copied());
            self.neighbors.set_bit(v);
        }

        if self.distance == Distance::Three {
            let mut dist_three = self.neighbors.clone();
            for x in self.neighbors.iter() {
                dist_three.set_bits(self.graph.neighbors_of(x).iter().copied());
            }
            self.neighbors = dist_three;
        }

        self.neighbor_lb = self.node + 1;
    }
}

impl<'a, G: AdjacencyList> Iterator for DistancePairsIterator<'a, G> {
    type Item = (Node, Node);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.neighbors.get_next_set(self.neighbor_lb) {
            self.neighbor_lb = (v + 1) as Node;
            return Some((self.node, v as Node));
        }

        loop {
            self.node += 1;

            if self.node + 1 >= self.graph.number_of_nodes() {
                return None;
            }

            self.setup_node();

            if self.neighbors.cardinality() > 0 {
                return self.next();
            }
        }
    }
}
