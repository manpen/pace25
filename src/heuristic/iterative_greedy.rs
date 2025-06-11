use itertools::Itertools;
use log::info;
use rand::Rng;

use crate::{
    graph::*,
    prelude::IterativeAlgorithm,
    utils::{dominating_set::DominatingSet, radix::NodeHeap},
};

/// # The basic Greedy-Approximation
///
/// 1. Fixes nodes in complete subgraphs where at most one node has neighbors outside the subgraph
///     - Singletons
///     - Nodes that are the only neighbors of some other node
///     - Nodes part of a triangle where the other two nodes are only incident to the triangle
/// 2. Greedily adds nodes neighbored to the highest number of uncovered nodes to DomSet until
///    covered
/// 3. Remove nodes that now redundant, ie. do not cover any nodes that not other DomSet-Node
///    covers
///
/// Returns the solution
pub struct IterativeGreedy<'a, G, R> {
    rng: &'a mut R,
    graph: &'a G,
    never_select: &'a BitSet,

    max_degree: NumNodes,
    total_covered: NumNodes,
    num_covered: Vec<NumNodes>,

    best_solution: Option<DominatingSet>,

    rand_bits: u32,
    rand_mask: NumNodes,

    round: u64,

    strategy: GreedyStrategy,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum GreedyStrategy {
    #[default]
    UnitValue,
    DegreeValue,
}

impl<'a, G: AdjacencyList + AdjacencyTest, R: Rng> IterativeGreedy<'a, G, R> {
    pub fn new(
        rng: &'a mut R,
        graph: &'a G,
        covered_nodes: &'a BitSet,
        never_select: &'a BitSet,
    ) -> Self {
        let mut num_covered = vec![0; graph.number_of_nodes() as usize];

        covered_nodes
            .iter_set_bits()
            .for_each(|u| num_covered[u as usize] = 1);
        let total_covered = covered_nodes.cardinality();

        let rand_bits = (graph.max_degree() + 1).leading_zeros();
        Self {
            rng,
            graph,
            never_select,
            max_degree: graph.max_degree(),
            total_covered,
            num_covered,
            best_solution: None,
            rand_bits,
            rand_mask: (1 << rand_bits) - 1,
            round: 0,
            strategy: Default::default(),
        }
    }

    pub fn set_strategy(&mut self, strategy: GreedyStrategy) {
        self.strategy = strategy;
    }

    fn step_impl(&mut self) -> DominatingSet {
        let mut solution = DominatingSet::new(self.graph.number_of_nodes());

        const MAX_RAND: u32 = 64;
        let max_value = u32::MAX / (1 + self.max_degree) - MAX_RAND;

        let mut node_values = match self.strategy {
            GreedyStrategy::UnitValue => self.num_covered.clone(),
            GreedyStrategy::DegreeValue => self
                .graph
                .vertices_range()
                .map(|u| {
                    if self.num_covered[u as usize] > 0 {
                        0
                    } else {
                        let coverage_candidated = self
                            .graph
                            .closed_neighbors_of(u)
                            .filter(|&v| self.never_select.get_bit(v))
                            .count() as NumNodes;

                        let random = self.rng.gen_range(0..MAX_RAND);
                        (max_value / (1 + coverage_candidated)).max(1) + random
                    }
                })
                .collect_vec(),
        };

        // Compute scores for non-fixed nodes
        let mut heap = NodeHeap::new(self.graph.number_of_nodes() as usize, 0);
        for u in self.graph.vertices() {
            if self.never_select.get_bit(u) {
                continue;
            }

            // Neighborhood no longer closed
            match self.strategy {
                GreedyStrategy::UnitValue => {
                    let mut node_score =
                        self.max_degree + (node_values[u as usize] > 0) as NumNodes;
                    for v in self.graph.neighbors_of(u) {
                        node_score -= (node_values[v as usize] == 0) as NumNodes;
                    }

                    if node_score <= self.max_degree {
                        let rand_word: u32 = self.rng.r#gen();
                        let rand_score =
                            (node_score << self.rand_bits) | (rand_word & self.rand_mask);

                        heap.push(rand_score, u);
                    }
                }
                GreedyStrategy::DegreeValue => {
                    let value: u32 = self
                        .graph
                        .closed_neighbors_of(u)
                        .map(|v| node_values[v as usize])
                        .sum();

                    if value > 0 {
                        heap.push(u32::MAX - value, u);
                    }
                }
            }
        }

        let mut total_covered = self.total_covered;

        // Compute rest of DomSet via GreedyAlgorithm
        while total_covered < self.graph.number_of_nodes() {
            let (_, node) = heap.pop().unwrap();
            solution.add_node(node);

            match self.strategy {
                GreedyStrategy::UnitValue => {
                    for u in self.graph.closed_neighbors_of(node) {
                        node_values[u as usize] += 1;
                        if node_values[u as usize] == 1 {
                            total_covered += 1;
                            for v in self.graph.closed_neighbors_of(u) {
                                if v != node && !self.never_select.get_bit(v) {
                                    let current_score = heap.remove(v);
                                    heap.push(current_score + 1 + self.rand_mask, v);
                                }
                            }
                        }
                    }
                }
                GreedyStrategy::DegreeValue => {
                    for u in self.graph.closed_neighbors_of(node) {
                        let prev_value = node_values[u as usize];
                        if prev_value == 0 {
                            continue;
                        }
                        node_values[u as usize] = 0;
                        total_covered += 1;

                        for v in self.graph.closed_neighbors_of(u) {
                            if v != node && !self.never_select.get_bit(v) {
                                let current_score = heap.remove(v) + prev_value;
                                if current_score < u32::MAX {
                                    heap.push(current_score, v);
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut index = 0;
        while index < solution.len() {
            let node = solution.ith_node(index);
            if self
                .graph
                .closed_neighbors_of(node)
                .all(|u| node_values[u as usize] > 1)
            {
                for u in self.graph.closed_neighbors_of(node) {
                    node_values[u as usize] -= 1;
                }
                solution.remove_node(node);
                continue;
            }
            index += 1;
        }

        solution
    }
}

impl<G: AdjacencyList + AdjacencyTest, R: Rng> IterativeAlgorithm<DominatingSet>
    for IterativeGreedy<'_, G, R>
{
    fn execute_step(&mut self) {
        self.round += 1;
        let candidate = self.step_impl();
        if self
            .best_solution
            .as_ref()
            .is_none_or(|bs| bs.len() > candidate.len())
        {
            info!(
                "Greedy found a solution of size {:7} in round {:5}",
                candidate.len(),
                self.round
            );
            self.best_solution = Some(candidate);
        }
    }

    fn is_completed(&self) -> bool {
        false
    }

    fn best_known_solution(&mut self) -> Option<DominatingSet> {
        self.best_solution.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng as _;
    use rand_pcg::Pcg64Mcg;

    #[test]
    /// Randomly generate G(n,p) graphs and check that the algorithm produces a feasible solution
    fn full_graph() {
        let mut rng = Pcg64Mcg::seed_from_u64(123456);
        for i in 0..3000 {
            let graph = AdjArray::random_gnp(&mut rng, 100, 0.03);

            let mut covered = graph.vertex_bitset_unset();
            let mut never_select = graph.vertex_bitset_unset();

            for _ in 0..(i % 11) {
                let cand = rng.gen_range(graph.vertices_range());
                covered.set_bit(cand);
            }

            for _ in 0..(i % 7) {
                let cand = rng.gen_range(graph.vertices_range());
                never_select.set_bit(cand);
            }

            never_select -= &covered;

            // test that instance remains feasible
            {
                let mut tmp = DominatingSet::new(graph.number_of_nodes());
                tmp.add_nodes(never_select.iter_cleared_bits());
                if !tmp.is_valid_given_previous_cover(&graph, &covered) {
                    continue;
                }
            }

            let mut algo = IterativeGreedy::new(&mut rng, &graph, &covered, &never_select);

            for _ in (i % 4)..5 {
                algo.execute_step();
            }

            let domset = algo.best_known_solution().unwrap();

            assert!(domset.is_valid_given_previous_cover(&graph, &covered));
        }
    }

    #[test]
    /// Randomly generate G(n,p) graphs and check that the algorithm produces a feasible solution
    fn full_graph_degree_score() {
        let mut rng = Pcg64Mcg::seed_from_u64(123456);
        for i in 0..3000 {
            let graph = AdjArray::random_gnp(&mut rng, 100, 0.03);

            let mut covered = graph.vertex_bitset_unset();
            let mut never_select = graph.vertex_bitset_unset();

            for _ in 0..(i % 11) {
                let cand = rng.gen_range(graph.vertices_range());
                covered.set_bit(cand);
            }

            for _ in 0..(i % 7) {
                let cand = rng.gen_range(graph.vertices_range());
                never_select.set_bit(cand);
            }

            never_select -= &covered;

            // test that instance remains feasible
            {
                let mut tmp = DominatingSet::new(graph.number_of_nodes());
                tmp.add_nodes(never_select.iter_cleared_bits());
                if !tmp.is_valid_given_previous_cover(&graph, &covered) {
                    continue;
                }
            }

            let mut algo = IterativeGreedy::new(&mut rng, &graph, &covered, &never_select);
            algo.set_strategy(GreedyStrategy::DegreeValue);

            for _ in (i % 4)..5 {
                algo.execute_step();
            }

            let domset = algo.best_known_solution().unwrap();

            assert!(domset.is_valid_given_previous_cover(&graph, &covered));
        }
    }
}
