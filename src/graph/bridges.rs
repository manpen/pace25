use std::marker::PhantomData;

use super::*;

pub trait Bridges {
    fn compute_bridges(&self) -> Vec<Edge>;
}

pub trait ColoredBridges {
    fn compute_colored_bridges(&self) -> Vec<ColoredEdge>;
}

impl<G: AdjacencyList> Bridges for G {
    fn compute_bridges(&self) -> Vec<Edge> {
        BridgeSearch::new(self, |u: Node| self.edges_of(u, false).map(|e| (e.1, e))).compute()
    }
}

impl<G: AdjacencyList + ColoredAdjacencyList> ColoredBridges for G {
    fn compute_colored_bridges(&self) -> Vec<ColoredEdge> {
        BridgeSearch::new(self, |u: Node| {
            self.colored_edges_of(u, false).map(|e| (e.1, e))
        })
        .compute()
    }
}

struct BridgeSearch<'a, G, E, N, I> {
    graph: &'a G,
    visited: BitSet,
    nodes_info: Vec<NodeInfo>,
    time: Node,
    bridges: Vec<E>,
    get_neighbors: N,
    _i: PhantomData<I>,
}

impl<'a, G, E, N, I> BridgeSearch<'a, G, E, N, I>
where
    G: AdjacencyList,
    N: Fn(Node) -> I + 'a,
    I: Iterator<Item = (Node, E)>,
{
    fn new(graph: &'a G, get_neighbors: N) -> Self {
        let n = graph.number_of_nodes();
        Self {
            graph,
            visited: BitSet::new(n),
            nodes_info: vec![NodeInfo::default(); n as usize],
            time: 0,
            bridges: Vec::new(),
            get_neighbors,
            _i: PhantomData,
        }
    }

    fn compute(mut self) -> Vec<E> {
        for u in self.graph.vertices() {
            if self.graph.degree_of(u) == 0 || self.visited.set_bit(u) {
                continue;
            }

            self.compute_node(u, u);
        }

        self.bridges
    }

    fn compute_node(&mut self, parent: Node, u: Node) -> NodeInfo {
        self.time += 1;

        self.nodes_info[u as usize] = NodeInfo {
            parent,
            discovery: self.time,
            low: self.time,
        };

        for (v, edge) in (self.get_neighbors)(u) {
            if !self.visited.set_bit(v) {
                let info_v = self.compute_node(u, v);

                self.nodes_info[u as usize].update_low(info_v.low);

                if info_v.low > self.nodes_info[u as usize].discovery {
                    self.bridges.push(edge)
                }
            } else if v != self.nodes_info[u as usize].parent {
                let v_disc = self.nodes_info[v as usize].discovery;
                self.nodes_info[u as usize].update_low(v_disc);
            }
        }

        self.nodes_info[u as usize]
    }
}

#[derive(Clone, Copy, Default)]
struct NodeInfo {
    low: Node,
    discovery: Node,
    parent: Node,
}

impl NodeInfo {
    fn update_low(&mut self, value: Node) {
        self.low = self.low.min(value);
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::prelude::*;

    #[test]
    fn bridges_in_path() {
        for n in [0, 1, 5, 10, 15] {
            let mut graph = AdjArray::new(n);
            for u in 0..n.saturating_sub(1) {
                graph.add_edge(u, u + 1);
            }

            let mut bridges = graph.compute_bridges();
            bridges.sort();

            assert_eq!(bridges, graph.ordered_edges(true).collect_vec());
        }
    }

    #[test]
    fn bridge_in_example() {
        let mut graph = AdjArray::new(6);
        graph.add_edges([(0, 1), (0, 2), (2, 1), (1, 3), (3, 4), (4, 5), (5, 3)]);

        assert_eq!(graph.compute_bridges(), vec![Edge(1, 3)]);
    }
}
