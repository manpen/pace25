use super::*;

pub struct DistanceTwoPairsIterator<'a, G: AdjacencyList> {
    graph: &'a G,
    node: Node,

    neighbors: BitSet,
    neighbor_lb: Node,
}

pub trait DistanceTwoPairs {
    type DistanceTwoPairsIterator<'a>: Iterator<Item = (Node, Node)> + 'a
    where
        Self: 'a;
    fn distance_two_pairs(&self) -> Self::DistanceTwoPairsIterator<'_>;
}

impl<G: AdjacencyList> DistanceTwoPairs for G {
    type DistanceTwoPairsIterator<'a> = DistanceTwoPairsIterator<'a, Self> where Self: 'a;

    fn distance_two_pairs(&self) -> Self::DistanceTwoPairsIterator<'_> {
        DistanceTwoPairsIterator::new(self)
    }
}

impl<'a, G: AdjacencyList> DistanceTwoPairsIterator<'a, G> {
    pub fn new(graph: &'a G) -> Self {
        let n = graph.number_of_nodes();

        let mut inst = Self {
            graph,
            node: 0,
            neighbors: BitSet::new(n),
            neighbor_lb: 0,
        };

        inst.setup_node();

        inst
    }

    fn setup_node(&mut self) {
        self.neighbors.unset_all();

        for &v in self.graph.neighbors_of(self.node) {
            for &w in self
                .graph
                .neighbors_of(v)
                .iter()
                .filter(|&&w| w > self.node)
            {
                self.neighbors.set_bit(w);
            }
            self.neighbors.set_bit(v);
        }

        self.neighbor_lb = self.node + 1;
    }
}

impl<'a, G: AdjacencyList> Iterator for DistanceTwoPairsIterator<'a, G> {
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
