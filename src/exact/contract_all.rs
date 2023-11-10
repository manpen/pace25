use std::{
    hash::BuildHasherDefault,
    time::{Duration, Instant},
};

use fxhash::FxHashMap;
use itertools::Itertools;
#[allow(unused_imports)]
use log::{info, trace};
use rand::seq::SliceRandom;

use crate::prelude::*;

#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    Timeout,
    Infeasible,
}

pub type SolverResultCache = FxHashMap<digest::Output<sha2::Sha256>, bool>;
type ResultType<T> = std::result::Result<T, Error>;

pub struct ContractAll<G: FullfledgedGraph> {
    cache: SolverResultCache,

    cache_miss: u64,
    cache_hit_feasible: u64,
    cache_hit_infeasible: u64,

    result: Vec<(G, Vec<(Node, Node)>)>,
    tww: NumNodes,
    stop_at_instant: Option<Instant>,
}

impl<G: FullfledgedGraph> ContractAll<G> {
    pub fn new(tww: NumNodes, timeout: Option<Duration>) -> Self {
        let cache =
            SolverResultCache::with_capacity_and_hasher(1_000_000, BuildHasherDefault::default());

        let result = Vec::new();
        let stop_at_instant = timeout.map(|t| Instant::now() + t);

        Self {
            cache,
            result,
            tww,
            stop_at_instant,

            cache_miss: 0,
            cache_hit_feasible: 0,
            cache_hit_infeasible: 0,
        }
    }

    pub fn solve(&mut self, input_graph: &G) -> ResultType<()> {
        let n = input_graph.number_of_nodes();
        if n < self.tww || input_graph.number_of_edges() == 0 {
            return Ok(());
        }

        let mut optimal_contractions = Vec::new();

        for survivor in 0..n {
            if input_graph.degree_of(survivor) == 0 {
                continue;
            }

            for removed in (survivor + 1)..n {
                if input_graph.degree_of(removed) == 0 {
                    continue;
                }

                let mut local_graph = input_graph.clone();
                local_graph.merge_node_into(removed, survivor);

                let optimal = self.is_feasible(&local_graph)?;

                if optimal {
                    optimal_contractions.push((survivor, removed));
                }
            }
        }

        if optimal_contractions.is_empty() {
            return Err(Error::Infeasible);
        }

        let (survivor, removed) = optimal_contractions
            .choose(&mut rand::thread_rng())
            .copied()
            .unwrap();

        self.result
            .push((input_graph.clone(), optimal_contractions));

        // recurse on random (optimal) subgraph
        let mut local_graph = input_graph.clone();
        local_graph.merge_node_into(removed, survivor);
        self.solve(&local_graph)
    }

    fn is_feasible(&mut self, graph: &G) -> ResultType<bool> {
        if self.stop_at_instant.map_or(false, |s| Instant::now() > s) {
            return Err(Error::Timeout);
        }

        let n = graph.number_of_nodes_with_neighbors();

        if n <= self.tww + 1 {
            return Ok(true);
        }

        if graph.max_red_degree() > self.tww {
            return Ok(false);
        }

        let candidates = self.contraction_candidates(graph);

        for (_, (survivor, removed)) in candidates {
            assert!(survivor < removed);

            let mut local_graph = graph.clone();
            local_graph.merge_node_into(removed, survivor);
            let hash = local_graph.binary_digest_sha256();

            match self.cache.get(&hash) {
                Some(false) => {
                    self.cache_hit_infeasible += 1;
                    continue;
                }

                Some(true) => {
                    self.cache_hit_feasible += 1;
                    return Ok(true);
                }

                _ => {
                    self.cache_miss += 1;
                }
            };

            let optimal = self.is_feasible(&local_graph)?;
            self.cache.insert(hash, optimal);

            if optimal {
                return Ok(true);
            }
        }

        Ok(false) // failed to find contraction
    }

    fn contraction_candidates(&self, graph: &G) -> Vec<(u32, (u32, u32))> {
        let mut pairs = Vec::new();

        let red_degs_of_black_neighbors = graph
            .vertices()
            .map(|u| {
                graph
                    .black_neighbors_of(u)
                    .map(|v| graph.red_degree_of(v) + 1)
                    .max()
                    .unwrap_or(0)
            })
            .collect_vec();

        for u in graph.vertices_range() {
            let degree_u = graph.degree_of(u);
            if degree_u == 0 {
                continue;
            }

            let two_neighbors = graph.closed_two_neighborhood_of(u);

            for v in two_neighbors.iter_set_bits() {
                if v >= u {
                    break;
                }

                let mut red_neighs = graph.red_neighbors_after_merge(u, v, false);
                let mut red_card = red_neighs.cardinality();

                if red_card > self.tww {
                    continue;
                }

                red_neighs.clear_bits(graph.red_neighbors_of(u));
                red_neighs.clear_bits(graph.red_neighbors_of(v));

                for new_red in red_neighs.iter_set_bits() {
                    red_card = red_card.max(graph.red_degree_of(new_red) + 1);
                }

                if red_neighs.cardinality() <= self.tww {
                    pairs.push((red_neighs.cardinality(), (v, u)));
                }
            }

            if degree_u > self.tww {
                continue;
            }

            let distant_nodes = {
                let mut distant_nodes = two_neighbors;
                distant_nodes.flip_all();
                distant_nodes
            };

            let red_deg_of_black_neighbors = graph
                .black_neighbors_of(u)
                .map(|v| graph.red_degree_of(v) + 1)
                .max()
                .unwrap_or(0);

            if red_degs_of_black_neighbors[u as usize] > self.tww {
                continue;
            }

            for v in distant_nodes.iter_set_bits() {
                if v >= u {
                    break;
                }

                let degree_v = graph.degree_of(v);
                let red_degree = red_deg_of_black_neighbors.max(degree_u + degree_v);
                if degree_v > 0
                    && red_degree <= self.tww
                    && graph
                        .black_neighbors_of(v)
                        .all(|w| graph.red_degree_of(w) < self.tww)
                {
                    pairs.push((red_degree, (v, u)));
                }
            }
        }
        pairs.sort();
        pairs
    }

    pub fn get_result(self) -> Vec<(G, Vec<(Node, Node)>)> {
        self.result
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::branch_and_bound::FeatureConfiguration;

    use super::*;

    #[test]
    fn cross_with_naive() {
        for (i, tww) in [1, 2, 0, 0, 3, 0, 2, 4, 1, 2].into_iter().enumerate() {
            if i == 4 {
                continue; // too slow
            }

            let filename = format!("instances/tiny/tiny{:>03}.gr", i + 1);
            println!("Testing on {filename} with assume TWW {tww}");
            let graph = AdjArray::try_read_pace_file(filename).expect("Cannot read tiny graph");

            let mut solver = ContractAll::new(tww, None);

            solver
                .solve(&graph)
                .expect("This graph should be solveable");

            for (graph, contracts) in solver.result {
                println!("{contracts:?}");
                for (rem, sur) in contracts {
                    let mut local_graph = graph.clone();
                    local_graph.merge_node_into(rem, sur);

                    let mut algo =
                        branch_and_bound::BranchAndBound::new_with_bounds(local_graph, tww, tww);

                    algo.configure_features(FeatureConfiguration::pessimitic());

                    assert!(algo.solve().is_some());
                }
            }
        }
    }
}
