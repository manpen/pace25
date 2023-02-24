use crate::prelude::{
    contraction_sequence::ContractionSequence,
    reductions::{prune_leaves, prune_twins},
    AdjacencyList, BitSet, ColoredAdjacencyList, ColoredAdjacencyTest, DistanceTwoPairs,
    GraphEdgeEditing, GraphEdgeOrder, absolute_twin_width_sat_encoding::AbsoluteTwinWidthSatEncoding, lower_bound_lb1::{self, lower_bound},
};
use std::{fmt::Debug, cmp::Reverse, f32::consts::E, collections::btree_set::SymmetricDifference};

use super::partial_monte_carlo_search_tree::PartialMonteCarloSearchTree;

/*
instances/exact-public/exact_002.gr                |     20 |       69 |      6 (     6) |    233 ms | 5 lb
instances/exact-public/exact_004.gr                |     25 |      181 |      7 (     7) |   1410 ms | 4 lb
instances/exact-public/exact_006.gr                |     28 |      131 |      7 (     7) |   2276 ms | 3 lb
instances/exact-public/exact_008.gr                |     28 |      210 |     10 (    10) |   7856 ms | 10 lb
instances/exact-public/exact_010.gr                |     28 |      235 |      6 (     6) |   2863 ms | 3 lb
instances/exact-public/exact_012.gr                |     29 |      180 |      8 (     8) |   2315 ms | 5 lb
instances/exact-public/exact_014.gr                |     30 |      175 |      8 (     8) |   3327 ms | 4 lb
instances/exact-public/exact_016.gr                |     30 |      195 |      8 (     8) |   2836 ms | 5 lb
instances/exact-public/exact_018.gr                |     31 |       52 |      3 (     3) |   1072 ms | 0 lb
instances/exact-public/exact_020.gr                |     32 |       90 |      5 (     5) |   1863 ms | 1 lb
instances/exact-public/exact_022.gr                |     33 |      135 |      5 (     5) |   4324 ms | 0 lb
instances/exact-public/exact_024.gr                |     40 |       89 |      5 (     5) |   7355 ms | 0 lb
instances/exact-public/exact_026.gr                |     44 |       95 |      4 (     4) |   7163 ms | 0 lb
instances/exact-public/exact_028.gr                |     47 |       79 |      3 (     3) |   5387 ms | 0 lb
instances/exact-public/exact_030.gr                |     48 |       78 |      3 (     3) |   5598 ms | 0 lb
instances/exact-public/exact_032.gr                |     48 |       94 |      3 (     3) |  16096 ms | 0 lb
instances/exact-public/exact_034.gr                |     51 |      240 |      4 (     4) |    938 ms | 0 lb
instances/exact-public/exact_036.gr                |     52 |       87 |      3 (     3) |   7453 ms | 0 lb
instances/exact-public/exact_038.gr                |     52 |      141 |      4 (     3) |  19133 ms | 0 lb
instances/exact-public/exact_040.gr                |     54 |       85 |      3 (     3) |  16155 ms | 0 lb
instances/exact-public/exact_042.gr                |     54 |      124 |      3 (     3) |   6562 ms | 0 lb
instances/exact-public/exact_044.gr                |     56 |       64 |      2 (     2) |   1364 ms | 0 lb
instances/exact-public/exact_046.gr                |     57 |      107 |      3 (     3) |   8644 ms | 1 lb
instances/exact-public/exact_048.gr                |     60 |      103 |      3 (     3) |  17079 ms | 1 lb
instances/exact-public/exact_050.gr                |     62 |      108 |      2 (     2) |   2178 ms | 0 lb
instances/exact-public/exact_052.gr                |     66 |      122 |      3 (     3) |  19806 ms | 0 lb
instances/exact-public/exact_054.gr                |     73 |      132 |      5 (     ?) |  81724 ms | 0 lb
instances/exact-public/exact_056.gr                |     74 |      116 |      2 (     2) |   6213 ms | 0 lb
instances/exact-public/exact_058.gr                |     78 |       93 |      3 (     ?) |  49525 ms | 1 lb
instances/exact-public/exact_060.gr                |     80 |       94 |      3 (     ?) |  59170 ms | 1 lb
instances/exact-public/exact_062.gr                |     81 |      155 |      2 (     2) |   8361 ms | 0 lb
instances/exact-public/exact_064.gr                |     84 |       85 |      2 (     2) |  17830 ms | 0 lb
instances/exact-public/exact_066.gr                |     94 |       98 |      2 (     2) |  16592 ms | 0 lb
instances/exact-public/exact_068.gr                |    103 |      151 |      2 (     2) |  25981 ms | 0 lb
instances/exact-public/exact_070.gr                |    140 |     6480 |     31 (     ?) |   8077 ms | 3 lb
instances/exact-public/exact_072.gr                |    141 |     6417 |     30 (     ?) |   7703 ms | 8 lb
instances/exact-public/exact_074.gr                |    141 |     6607 |     33 (     ?) |   7210 ms | 12 lb
instances/exact-public/exact_076.gr                |    141 |     6731 |     31 (     ?) |   8597 ms | 5 lb
instances/exact-public/exact_078.gr                |    143 |     6625 |     32 (     ?) |   6982 ms | 1 lb
instances/exact-public/exact_080.gr                |    143 |     7061 |     30 (     ?) |   8730 ms | 7 lb
instances/exact-public/exact_082.gr                |    143 |     7209 |     30 (     ?) |   8484 ms | 12 lb
instances/exact-public/exact_084.gr                |    200 |    12048 |     84 (     ?) |  43063 ms | 69 lb
instances/exact-public/exact_086.gr                |    200 |    13868 |     77 (     ?) |  45196 ms | 56 lb
instances/exact-public/exact_088.gr                |    200 |    13930 |     47 (     ?) |  40791 ms | 33 lb
instances/exact-public/exact_090.gr                |    202 |     4574 |     20 (     ?) |  17353 ms | 0 lb
instances/exact-public/exact_092.gr                |    478 |     8562 |     44 (     ?) | 289165 ms | 0 lb
instances/exact-public/exact_094.gr                |    822 |     4518 |     31 (     ?) | 659482 ms | 1 lb
instances/exact-public/exact_096.gr                |    822 |     4698 |     50 (     ?) |    577 ms | 2 lb
instances/exact-public/exact_098.gr                |    909 |     2782 |     39 (     ?) |    360 ms | 0 lb
instances/exact-public/exact_100.gr                |    916 |     2994 |     38 (     ?) |    429 ms | 0 lb
instances/exact-public/exact_102.gr                |   1010 |     3648 |     31 (     ?) |    314 ms | 0 lb
instances/exact-public/exact_104.gr                |   1024 |     4916 |     49 (     ?) |    911 ms | 0 lb
instances/exact-public/exact_106.gr                |   1049 |    13426 |     48 (     ?) |    126 ms | 0 lb
instances/exact-public/exact_108.gr                |   1182 |    17447 |     10 (     ?) |     51 ms | 0 lb
instances/exact-public/exact_110.gr                |   1224 |     5822 |     12 (     ?) |     77 ms | 0 lb
instances/exact-public/exact_112.gr                |   1258 |     7513 |     51 (     ?) |    705 ms | 2 lb
instances/exact-public/exact_114.gr                |   1350 |     5215 |     29 (     ?) |    520 ms | 6 lb
instances/exact-public/exact_116.gr                |   1365 |     5263 |     28 (     ?) |    465 ms | 6 lb
instances/exact-public/exact_118.gr                |   1413 |     3955 |     14 (     ?) |    151 ms | 0 lb
instances/exact-public/exact_120.gr                |   1488 |     3777 |     14 (     ?) |    135 ms | 0 lb
instances/exact-public/exact_122.gr                |   1505 |     4700 |     18 (     ?) |     51 ms | 0 lb
instances/exact-public/exact_124.gr                |   1586 |     5511 |     23 (     ?) |    251 ms | 0 lb
instances/exact-public/exact_126.gr                |   1632 |     7281 |     43 (     ?) |   1178 ms | 6 lb
instances/exact-public/exact_128.gr                |   1700 |     4200 |     39 (     ?) |    423 ms | 2 lb
instances/exact-public/exact_130.gr                |   1742 |     5081 |     24 (     ?) |    182 ms | 0 lb
instances/exact-public/exact_132.gr                |   1813 |     6367 |    160 (     ?) |    738 ms | 0 lb
instances/exact-public/exact_134.gr                |   1813 |     6384 |    160 (     ?) |    713 ms | 0 lb
instances/exact-public/exact_136.gr                |   1813 |     6384 |    160 (     ?) |    710 ms | 0 lb
instances/exact-public/exact_138.gr                |   1813 |     6384 |    160 (     ?) |    724 ms | 0 lb
instances/exact-public/exact_140.gr                |   1813 |     6384 |    160 (     ?) |    703 ms | 0 lb
instances/exact-public/exact_142.gr                |   1813 |     6384 |    160 (     ?) |    708 ms | 0 lb
instances/exact-public/exact_144.gr                |   1813 |     6384 |    160 (     ?) |    695 ms | 0 lb
instances/exact-public/exact_146.gr                |   1813 |     6384 |    160 (     ?) |    713 ms | 0 lb
instances/exact-public/exact_148.gr                |   1813 |     6384 |    160 (     ?) |    712 ms | 0 lb
instances/exact-public/exact_150.gr                |   1813 |     6384 |    160 (     ?) |    705 ms | 0 lb
instances/exact-public/exact_152.gr                |   1814 |     6387 |    145 (     ?) |    702 ms | 0 lb
instances/exact-public/exact_154.gr                |   2017 |    14708 |     14 (     ?) |    408 ms | 2 lb
instances/exact-public/exact_156.gr                |   2166 |     6373 |     38 (     ?) |    295 ms | 0 lb
instances/exact-public/exact_158.gr                |   2194 |    17836 |     14 (     ?) |    543 ms | 0 lb
instances/exact-public/exact_160.gr                |   2263 |     3107 |      7 (     ?) |     44 ms | 0 lb
instances/exact-public/exact_162.gr                |   2530 |     3057 |      8 (     ?) |     35 ms | 0 lb
instances/exact-public/exact_164.gr                |   2543 |     4244 |     14 (     ?) |     80 ms | 0 lb
instances/exact-public/exact_166.gr                |   2562 |     3256 |     24 (     ?) |     52 ms | 0 lb
instances/exact-public/exact_168.gr                |   2675 |     3185 |      9 (     ?) |     51 ms | 0 lb
instances/exact-public/exact_170.gr                |   2735 |     7704 |     20 (     ?) |    177 ms | 0 lb
instances/exact-public/exact_172.gr                |   2931 |     5369 |     12 (     ?) |    122 ms | 0 lb
instances/exact-public/exact_174.gr                |   3006 |    10806 |     18 (     ?) |     76 ms | 0 lb
instances/exact-public/exact_176.gr                |   3006 |    10806 |     17 (     ?) |     81 ms | 0 lb
instances/exact-public/exact_178.gr                |   3034 |     3297 |      7 (     ?) |     46 ms | 0 lb
instances/exact-public/exact_180.gr                |   3086 |     5764 |     16 (     ?) |    310 ms | 1 lb
instances/exact-public/exact_182.gr                |   3200 |     7840 |      8 (     ?) |    360 ms | 3 lb
instances/exact-public/exact_184.gr                |   3477 |     8337 |      6 (     ?) |     68 ms | 0 lb
instances/exact-public/exact_186.gr                |   3482 |     8237 |      6 (     ?) |     73 ms | 0 lb
instances/exact-public/exact_188.gr                |   3530 |     5473 |     11 (     ?) |     51 ms | 0 lb
instances/exact-public/exact_190.gr                |   3600 |    11500 |      8 (     ?) |    501 ms | 3 lb
instances/exact-public/exact_192.gr                |   3969 |    13555 |     78 (     ?) |    363 ms | 0 lb
instances/exact-public/exact_194.gr                |   5067 |     5540 |     23 (     ?) |     63 ms | 0 lb
instances/exact-public/exact_196.gr                |   7253 |     6711 |     15 (     ?) |     46 ms | 0 lb
instances/exact-public/exact_198.gr                |   7537 |     8559 |     31 (     ?) |    175 ms | 0 lb
instances/exact-public/exact_200.gr                |  20000 |    15000 |      0 (     0) |    165 ms | 0 lb
Heuristic Solver:   Cumulative: 131
                    Best Known: 130
                    Comparison on 31 best known.
 */

pub fn heuristic_solve<
    G: Clone
        + AdjacencyList
        + GraphEdgeOrder
        + ColoredAdjacencyList
        + ColoredAdjacencyTest
        + Debug
        + GraphEdgeEditing,
>(
    graph: &G,
) -> (u32, ContractionSequence) {
    let clone = graph.clone();
    let result : (u32, ContractionSequence) = if clone.number_of_edges() < 7000 && clone.number_of_nodes() < 200 {
        let sweeping_solver = SweepingSolver::new(clone.clone());
        let mut result = sweeping_solver.solve_greedy_pairwise_fast(None, 0, false,false,1);
        for i in 1..11 {
            let sweeping_solver_i = SweepingSolver::new(clone.clone());
            let result_i = sweeping_solver_i.solve_greedy_pairwise_fast(None, i, false,false,1);
            if result_i.0 <= result.0 {
                result = result_i;
            }
        }

        for i in 0..13 {
            let sweeping_solver_i = SweepingSolver::new(clone.clone());
            let result_i = sweeping_solver_i.solve_greedy_pairwise_fast_full(Some(result.0), i, false,false,1);
            if result_i.0 <= result.0 {
                result = result_i;
            }
        }
        
        /*for i in 0..6 {
            let sweeping_solver_i = SweepingSolver::new(clone.clone());
            let result_i = sweeping_solver_i.solve_greedy_pairwise_fast_full_sym_desced(result.0, i);
            if result_i.0 <= result.0 {
                result = result_i;
            }
        }*/

        // Improves graph 032 && 018 therefore make it faster then it would be good!
        for i in 0..8 {
            let sweeping_solver_i = SweepingSolver::new(clone.clone());
            let result_i = sweeping_solver_i.solve_greedy_pairwise_fast_full_sym_desced_only(result.0, i);
            if result_i.0 <= result.0 {
                result = result_i;
            }
        }

        // Use upper bound to refine
        /*for i in 1..8 {
            let sweeping_solver_i = SweepingSolver::new(clone.clone());
            let result_i = sweeping_solver_i.solve_greedy_pairwise_fast(Some(result.0), i, false,false,1);
            if result_i.0 < result.0 {
                result = result_i;
            }
        }

        for i in 2..5 {
            let sweeping_solver_i = SweepingSolver::new(clone.clone());
            let result_i = sweeping_solver_i.solve_greedy_pairwise_fast(Some(result.0), i, true,false,1);
            if result_i.0 < result.0 {
                result = result_i;
            }
        }*/

        result
    } else if clone.number_of_edges() < 6000 && clone.number_of_nodes() < 1000 {
        let sweeping_solver = SweepingSolver::new(clone.clone());

        let result_1 = sweeping_solver.solve_greedy_pairwise_fast(None, 2, false,false,1);
        let sweeping_solver = SweepingSolver::new(clone.clone());
        let result_2 = sweeping_solver.solve_greedy_pairwise_fast(None, 1, false,false,1);

        if result_1.0 <= result_2.0 {
            result_1
        } else {
            result_2
        }
    }
    else {
        let sweeping_solver = SweepingSolver::new(clone.clone());
        let result_1 = sweeping_solver.solve_greedy();

        result_1
    };

    result
}

pub struct SweepingSolver<G> {
    graph: G,
    preprocessing_sequence: ContractionSequence,
}

impl<
        G: Clone
            + AdjacencyList
            + GraphEdgeOrder
            + ColoredAdjacencyList
            + ColoredAdjacencyTest
            + Debug
            + GraphEdgeEditing,
    > SweepingSolver<G>
{
    pub fn new(graph: G) -> SweepingSolver<G> {
        let mut clone = graph;
        let mut preprocessing_sequence = ContractionSequence::new(clone.number_of_nodes());

        prune_leaves(&mut clone, &mut preprocessing_sequence);
        prune_twins(&mut clone, &mut preprocessing_sequence);

        SweepingSolver {
            graph: clone,
            preprocessing_sequence,
        }
    }

    pub fn new_without_preprocessing(graph: G) -> SweepingSolver<G> {
        let clone = graph.clone();
        SweepingSolver {
            graph: clone,
            preprocessing_sequence: ContractionSequence::new(graph.number_of_nodes()),
        }
    }

    //Pretty slow for larger graphs
    pub fn solve_greedy(mut self) -> (u32, ContractionSequence) {
        let mut remaining_nodes = self.preprocessing_sequence.remaining_nodes().unwrap();
        if remaining_nodes.cardinality() == 1 {
            return (0, self.preprocessing_sequence);
        }

        let mut allowed_tww = 1;

        let mut tww = 0;
        let mut contraction_sequence = ContractionSequence::new(self.graph.number_of_nodes());

        loop {
            if remaining_nodes.cardinality() <= 1 {
                break;
            }

            let mut merged = false;
            for node in remaining_nodes.clone().iter() {
                let neighbors = self.graph.neighbors_of(node);
                let mut bitset = self.graph.neighbors_of_as_bitset(node);
                for neighbor in neighbors.iter() {
                    bitset.or(&self.graph.neighbors_of_as_bitset(*neighbor));
                }
                bitset.unset_bit(node);
                // No neighbors?
                if bitset.cardinality() == 0 {
                    remaining_nodes.unset_bit(node);
                    continue;
                }

                for neighbors in bitset.iter() {
                    let new_red_deg = self.graph.red_degree_after_merge(neighbors, node);
                    if new_red_deg <= allowed_tww {
                        merged = true;
                        self.graph.merge_node_into(neighbors, node);
                        tww = tww.max(self.graph.red_degrees().max().unwrap());
                        remaining_nodes.unset_bit(neighbors);
                        contraction_sequence.merge_node_into(neighbors, node);
                        break;
                    }
                }
            }
            if !merged {
                allowed_tww += 1;
            }
        }
        self.preprocessing_sequence.append(&contraction_sequence);
        let _ = self.preprocessing_sequence.remaining_nodes().unwrap();
        (tww, self.preprocessing_sequence)
    }

    fn play_greedy_multiple_levels(
        graph: &G,
        first_move: (u32, u32),
        mut remaining_nodes: BitSet,
        mut num_levels: u32,
    ) -> (u32, u32) {
        let mut cloned = graph.clone();
        cloned.merge_node_into(first_move.0, first_move.1);
        remaining_nodes.unset_bit(first_move.0);
        let mut tww = cloned.red_degrees().max().unwrap();

        loop {
            if remaining_nodes.cardinality() <= 1 {
                break;
            }

            let minimum = cloned
                .distance_two_pairs()
                .map(|(u, v)| {
                    let red_neighs = cloned.red_neighbors_after_merge(u, v, false);
                    (red_neighs.cardinality(), (u, v))
                })
                .min();

            if let Some((_, (u, v))) = minimum {
                cloned.merge_node_into(u, v);
                remaining_nodes.unset_bit(u);

                tww = tww.max(cloned.red_degrees().max().unwrap());
            } else {
                break;
            }

            num_levels -= 1;
            if num_levels == 0 {
                break;
            }
        }
        (tww, cloned.red_degrees().sum())
    }

    fn play_greedy_multiple_levels_sym(
        graph: &G,
        first_move: (u32, u32),
        upper_bound: u32,
        mut remaining_nodes: BitSet,
        mut num_levels: u32,
    ) -> (u32, u32) {
        let mut cloned = graph.clone();
        cloned.merge_node_into(first_move.0, first_move.1);
        remaining_nodes.unset_bit(first_move.0);
        let mut tww = cloned.red_degrees().max().unwrap();

        loop {
            if remaining_nodes.cardinality() <= 1 {
                break;
            }

            let minimum = cloned
                .distance_two_pairs()
                .map(|(u, v)| {
                    let red_neighs = cloned.red_neighbors_after_merge(u, v, false);
                    (red_neighs.cardinality(), (u, v))
                })
                .min();

            if let Some((_, (u, v))) = minimum {
                cloned.merge_node_into(u, v);
                remaining_nodes.unset_bit(u);

                tww = tww.max(cloned.red_degrees().max().unwrap());
            } else {
                break;
            }

            num_levels -= 1;
            if num_levels == 0 {
                break;
            }
        }
        let mut sym_sum = 0;
        for x in 0..cloned.number_of_nodes() {
            let sym = SweepingSolver::<G>::get_node_min_sim(&cloned, x);
            if sym > upper_bound-1 {
                sym_sum+=sym-(upper_bound-1);
            }
        }
        (tww, sym_sum)
    }


    #[inline]
    pub fn red_degree_total_deg_after_merge(g: &G, next_move: (u32,u32)) -> (u32,u32) {
        let edge_0 = g.neighbors_of(next_move.0).len();
        let edge_1 = g.neighbors_of(next_move.1).len();
        let new_neighbors = g.red_neighbors_after_merge(next_move.0, next_move.1, false);
        let max = new_neighbors.cardinality();
        for x in g.nei
    }

    fn play_greedy_multiple_levels_sym_full(
        graph: &G,
        first_move: (u32, u32),
        upper_bound: u32,
        mut remaining_nodes: BitSet,
        mut num_levels: u32,
    ) -> (u32, u32) {
        let mut cloned = graph.clone();
        cloned.merge_node_into(first_move.0, first_move.1);
        remaining_nodes.unset_bit(first_move.0);
        let mut tww = cloned.red_degrees().max().unwrap();

        loop {
            if remaining_nodes.cardinality() <= 1 {
                break;
            }

            let minimum = cloned
                .distance_two_pairs()
                .flat_map(|(u, v)| {
                    let mut graph = cloned.clone();
                    graph.merge_node_into(u,v);
                    if graph.red_degrees().max().unwrap()>upper_bound-1 {
                        None
                    }
                    else {
                        let deg:u32 = graph.degrees().sum();
                        Some((deg, (u, v)))
                    }
                })
                .min();

            if let Some((_, (u, v))) = minimum {
                cloned.merge_node_into(u, v);
                remaining_nodes.unset_bit(u);

                tww = tww.max(cloned.red_degrees().max().unwrap());
            } else {
                break;
            }

            num_levels -= 1;
            if num_levels == 0 {
                break;
            }
        }

        let mut sym_sum = 0;
        for x in 0..cloned.number_of_nodes() {
            let sym = SweepingSolver::<G>::get_node_min_sim(&cloned, x);
            if sym > upper_bound-1 {
                sym_sum+=sym-(upper_bound-1);
            }
        }
        (tww, sym_sum)
    }

    #[allow(unused)]
    fn play_some_multiple_levels(
        graph: &G,
        first_move: (u32, u32),
        mut remaining_nodes: BitSet,
        num_levels: u32,
        ub: u32,
    ) -> (u32, u32) {
        let mut cloned = graph.clone();
        cloned.merge_node_into(first_move.0, first_move.1);
        remaining_nodes.unset_bit(first_move.0);
        let tww = cloned.red_degrees().max().unwrap();
        let minor: u32 = cloned.red_degrees().sum();

        if num_levels == 0 {
            return (tww, minor);
        }

        if remaining_nodes.cardinality() <= 2 {
            return (tww, minor);
        }

        let mut minimum: Vec<(u32, (u32, u32))> = cloned
            .distance_two_pairs()
            .map(|(u, v)| {
                let mut graph = cloned.clone();
                graph.merge_node_into(u, v);
                (graph.red_degrees().max().unwrap(), (u, v))
            })
            .collect();

        if minimum.is_empty() {
            return (tww, minor);
        }

        minimum.sort();

        let min_red = minimum[0].0;

        let mut min_tww = std::u32::MAX;
        let mut minor_size = std::u32::MAX;
        for x in minimum.into_iter() {
            if (x.0 >= ub) {
                let res = SweepingSolver::<G>::play_greedy_multiple_levels(&cloned, x.1, remaining_nodes, num_levels-1);
                min_tww = min_tww.min(res.0);
                break;
            }
            let mut min = SweepingSolver::<G>::play_some_multiple_levels(
                &cloned,
                x.1,
                remaining_nodes.clone(),
                num_levels - 1,
                ub,
            );
            min.0 = min.0.max(tww);
            if min_tww < min.0 && minor_size < min.1 {
                min_tww = min.0;
                minor_size = min.1;
            }
        }
        (min_tww, minor)
    }

    #[allow(unused)]
    fn play_complete_multiple_levels(
        graph: &G,
        first_move: (u32, u32),
        mut remaining_nodes: BitSet,
        num_levels: u32,
        ub: u32,
    ) -> (u32, u32) {
        let mut cloned = graph.clone();
        cloned.merge_node_into(first_move.0, first_move.1);
        remaining_nodes.unset_bit(first_move.0);
        let tww = cloned.red_degrees().max().unwrap();
        let minor: u32 = cloned.red_degrees().sum();

        if num_levels == 0 {
            return (tww, minor);
        }

        if remaining_nodes.cardinality() <= 2 {
            return (tww, minor);
        }

        let mut minimum: Vec<(u32, (u32, u32))> = cloned
            .distance_two_pairs()
            .map(|(u, v)| {
                let mut graph = cloned.clone();
                graph.merge_node_into(u, v);
                (graph.red_degrees().max().unwrap(), (u, v))
            })
            .collect();

        if minimum.is_empty() {
            return (tww, minor);
        }

        minimum.sort();

        let mut min_tww = std::u32::MAX;
        let mut minor_size = std::u32::MAX;
        for x in minimum.into_iter() {
            if x.0 >= ub {
                min_tww = min_tww.min(tww);
                break;
            }
            let mut min = SweepingSolver::<G>::play_complete_multiple_levels(
                &cloned,
                x.1,
                remaining_nodes.clone(),
                num_levels - 1,
                ub,
            );
            min.0 = min.0.max(tww);
            if min_tww < min.0 && minor_size < min.1 {
                min_tww = min.0;
                minor_size = min.1;
            }
        }
        (min_tww, minor)
    }

    pub fn solve_greedy_pairwise(mut self) -> (u32, ContractionSequence) {
        let remaining_nodes = self.preprocessing_sequence.remaining_nodes().unwrap();
        if remaining_nodes.cardinality() == 1 {
            return (0, self.preprocessing_sequence);
        }

        let mut global_array = vec![(
            0,
            self.graph.clone(),
            remaining_nodes,
            ContractionSequence::new(self.graph.number_of_nodes()),
        )];

        loop {
            let mut new_array = Vec::new();
            let mut global_min_tww = std::u32::MAX;

            for (mut tww, graph, remaining_nodes, contraction_sequence) in global_array.into_iter()
            {
                if remaining_nodes.cardinality() <= 1 {
                    self.preprocessing_sequence.append(&contraction_sequence);
                    let _ = self.preprocessing_sequence.remaining_nodes().unwrap();
                    return (tww, self.preprocessing_sequence);
                }

                let mut minimum: Vec<_> = graph
                    .distance_two_pairs()
                    .map(|(u, v)| {
                        let red_neighs = graph.red_neighbors_after_merge(u, v, false);
                        (red_neighs.cardinality(), (u, v))
                    })
                    .collect();

                if minimum.is_empty() {
                    self.preprocessing_sequence.append(&contraction_sequence);
                    let _ = self.preprocessing_sequence.remaining_nodes().unwrap();
                    return (tww, self.preprocessing_sequence);
                }

                minimum.sort();

                let min_red = minimum[0].0;

                let mut max_red = std::u32::MAX;
                let mut promising = Vec::new();

                for (red, (u, v)) in minimum.iter() {
                    // Only allow uphill at most two
                    if *red > min_red + 1 {
                        break;
                    }

                    let (max_reds, _) = SweepingSolver::play_greedy_multiple_levels(
                        &graph,
                        (*u, *v),
                        remaining_nodes.clone(),
                        5,
                    );
                    if max_reds < max_red {
                        max_red = max_reds;
                        promising.push((max_reds, (*u, *v)));
                    }
                }

                let mut markov = PartialMonteCarloSearchTree::new(&graph, &minimum, 4, max_red);
                markov.play_games((minimum.len() * 25) as u32);
                let (tww_mark, markov_move) = markov.into_best_choice_with_tww();
                if tww_mark < max_red {
                    promising.push((tww_mark, markov_move));
                }

                for best_moves in promising.into_iter() {
                    let mut cloned = graph.clone();
                    let mut remaining = remaining_nodes.clone();
                    let mut contraction = contraction_sequence.clone();

                    cloned.merge_node_into(best_moves.1 .0, best_moves.1 .1);
                    contraction.merge_node_into(best_moves.1 .0, best_moves.1 .1);
                    remaining.unset_bit(best_moves.0);
                    tww = tww.max(graph.red_degrees().max().unwrap());
                    global_min_tww = global_min_tww.min(tww);
                    new_array.push((tww, cloned, remaining, contraction));
                }
            }

            new_array.sort_by_key(|x| x.0);

            if new_array.len() > 16 {
                new_array.drain(16..);
            }
            global_array = new_array;
        }
    }

//Pretty slow for larger graphs
pub fn solve_greedy_pairwise_fast_full(
    mut self,
    upper_bound: Option<u32>,
    played_level: u32,
    enable_markov: bool,
    enable_complete_lookahead: bool,
    max_delta_beg: u32
) -> (u32, ContractionSequence) {
    let mut remaining_nodes = self.preprocessing_sequence.remaining_nodes().unwrap();
    if remaining_nodes.cardinality() == 1 {
        return (0, self.preprocessing_sequence);
    }

    let mut tww = 0;
    let mut contraction_sequence = ContractionSequence::new(self.graph.number_of_nodes());

    loop {
        if remaining_nodes.cardinality() <= 1 {
            break;
        }

        let mut minimum: Vec<_> = self
            .graph
            .distance_two_pairs()
            .map(|(u, v)| {
                let mut cloned = self.graph.clone();
                cloned.merge_node_into(u,v);
                (cloned.red_degrees().max().unwrap(), (u, v))
            })
            .collect();

        if minimum.is_empty() {
            break;
        }

        minimum.sort();

        let min_red = minimum[0].0;

        let mut min_move = minimum[0].1;
        let mut max_red = std::u32::MAX;
        let mut min_total_red_deg = std::u32::MAX;

        let mut frontier_size = 0;
        for (red, (u, v)) in minimum.iter() {
            // Only allow uphill at most one
            if *red > upper_bound.map(|x| x-1).unwrap_or(min_red + max_delta_beg) {
                break;
            }
            frontier_size += 1;

            if enable_complete_lookahead {
                // Only use this to try to keep the tww
                let (max_reds, total_red_deg) = SweepingSolver::play_some_multiple_levels(
                    &self.graph,
                    (*u, *v),
                    remaining_nodes.clone(),
                    played_level,
                    upper_bound.unwrap()
                );

                if (max_reds < max_red)
                    || (max_reds == max_red && total_red_deg < min_total_red_deg)
                {
                    max_red = max_reds;
                    min_move = (*u, *v);
                    min_total_red_deg = total_red_deg;
                }
            }


            let (max_reds, total_red_deg) = SweepingSolver::play_greedy_multiple_levels(
                &self.graph,
                (*u, *v),
                remaining_nodes.clone(),
                played_level,
            );

            if (max_reds < max_red)
                || (max_reds == max_red && total_red_deg < min_total_red_deg)
            {
                max_red = max_reds;
                min_move = (*u, *v);
                min_total_red_deg = total_red_deg;
            }
        }

        if enable_markov {
            let mut markov =
                PartialMonteCarloSearchTree::new(&self.graph, &minimum, played_level, max_red);
            markov.play_games((frontier_size) as u32 * 10);
            let (tww_mark, markov_move) = markov.into_best_choice_with_tww();
            if tww_mark < max_red {
                min_move = markov_move;
            }
        }

        self.graph.merge_node_into(min_move.0, min_move.1);
        contraction_sequence.merge_node_into(min_move.0, min_move.1);
        remaining_nodes.unset_bit(min_move.0);
        tww = tww.max(self.graph.red_degrees().max().unwrap());
    }
    self.preprocessing_sequence.append(&contraction_sequence);
    let _ = self.preprocessing_sequence.remaining_nodes().unwrap();
    (tww, self.preprocessing_sequence)
}

    //Pretty slow for larger graphs
    pub fn solve_greedy_pairwise_fast(
        mut self,
        upper_bound: Option<u32>,
        played_level: u32,
        enable_markov: bool,
        enable_complete_lookahead: bool,
        max_delta_beg: u32
    ) -> (u32, ContractionSequence) {
        let mut remaining_nodes = self.preprocessing_sequence.remaining_nodes().unwrap();
        if remaining_nodes.cardinality() == 1 {
            return (0, self.preprocessing_sequence);
        }

        let mut tww = 0;
        let mut contraction_sequence = ContractionSequence::new(self.graph.number_of_nodes());

        loop {
            if remaining_nodes.cardinality() <= 1 {
                break;
            }

            let mut minimum: Vec<_> = self
                .graph
                .distance_two_pairs()
                .map(|(u, v)| {
                    let red_neighs = self.graph.red_degree_after_merge(u, v);
                    (red_neighs, (u, v))
                })
                .collect();

            if minimum.is_empty() {
                break;
            }

            minimum.sort();

            let min_red = minimum[0].0;

            let mut min_move = minimum[0].1;
            let mut max_red = std::u32::MAX;
            let mut min_total_red_deg = std::u32::MAX;

            let mut frontier_size = 0;
            for (red, (u, v)) in minimum.iter() {
                // Only allow uphill at most one
                if *red > upper_bound.map(|x| x-1).unwrap_or(min_red + max_delta_beg) {
                    break;
                }
                frontier_size += 1;

                if enable_complete_lookahead {
                    // Only use this to try to keep the tww
                    let (max_reds, total_red_deg) = SweepingSolver::play_some_multiple_levels(
                        &self.graph,
                        (*u, *v),
                        remaining_nodes.clone(),
                        played_level,
                        tww
                    );
    
                    if (max_reds < max_red)
                        || (max_reds == max_red && total_red_deg < min_total_red_deg)
                    {
                        max_red = max_reds;
                        min_move = (*u, *v);
                        min_total_red_deg = total_red_deg;
                    }
                }


                let (max_reds, total_red_deg) = SweepingSolver::play_greedy_multiple_levels(
                    &self.graph,
                    (*u, *v),
                    remaining_nodes.clone(),
                    played_level,
                );

                if (max_reds < max_red)
                    || (max_reds == max_red && total_red_deg < min_total_red_deg)
                {
                    max_red = max_reds;
                    min_move = (*u, *v);
                    min_total_red_deg = total_red_deg;
                }
            }

            if enable_markov {
                let mut markov =
                    PartialMonteCarloSearchTree::new(&self.graph, &minimum, played_level, max_red);
                markov.play_games((frontier_size) as u32 * 10);
                let (tww_mark, markov_move) = markov.into_best_choice_with_tww();
                if tww_mark < max_red {
                    min_move = markov_move;
                }
            }

            self.graph.merge_node_into(min_move.0, min_move.1);
            contraction_sequence.merge_node_into(min_move.0, min_move.1);
            remaining_nodes.unset_bit(min_move.0);
            tww = tww.max(self.graph.red_degrees().max().unwrap());
        }
        self.preprocessing_sequence.append(&contraction_sequence);
        let _ = self.preprocessing_sequence.remaining_nodes().unwrap();
        (tww, self.preprocessing_sequence)
    }

    //Pretty slow for larger graphs
    pub fn solve_markov_pairwise(mut self) -> (u32, ContractionSequence) {
        let mut remaining_nodes = self.preprocessing_sequence.remaining_nodes().unwrap();
        if remaining_nodes.cardinality() == 1 {
            return (0, self.preprocessing_sequence);
        }

        let mut tww = 0;
        let mut contraction_sequence = ContractionSequence::new(self.graph.number_of_nodes());

        loop {
            if remaining_nodes.cardinality() <= 1 {
                break;
            }

            let mut minimum: Vec<_> = self
                .graph
                .distance_two_pairs()
                .map(|(u, v)| {
                    let red_neighs = self.graph.red_neighbors_after_merge(u, v, false);
                    (red_neighs.cardinality(), (u, v))
                })
                .collect();

            if minimum.is_empty() {
                break;
            }

            minimum.sort();

            let mut markov =
                PartialMonteCarloSearchTree::new(&self.graph, &minimum, 4, std::u32::MAX);
            markov.play_games((minimum.len() * 10) as u32);
            let min_move = markov.into_best_choice();

            self.graph.merge_node_into(min_move.0, min_move.1);
            contraction_sequence.merge_node_into(min_move.0, min_move.1);
            remaining_nodes.unset_bit(min_move.0);
            tww = tww.max(self.graph.red_degrees().max().unwrap());
        }
        self.preprocessing_sequence.append(&contraction_sequence);
        let _ = self.preprocessing_sequence.remaining_nodes().unwrap();
        (tww, self.preprocessing_sequence)
    }

    // Delta adjust the level of uncertainty which is allowed in the contraction
    pub fn solve_sweep(mut self, delta: u32) -> (u32, ContractionSequence) {
        let mut remaining_nodes = self.preprocessing_sequence.remaining_nodes().unwrap();
        if remaining_nodes.cardinality() == 1 {
            return (0, self.preprocessing_sequence);
        }

        let mut tww = 0;
        let mut contraction_sequence = ContractionSequence::new(self.graph.number_of_nodes());

        loop {
            // Set all nodes
            let mut unvisited_nodes = BitSet::new_all_set(self.graph.number_of_nodes());
            if remaining_nodes.cardinality() <= 1 {
                break;
            }
            let mut minimum: Vec<_> = self
                .graph
                .distance_two_pairs()
                .map(|(u, v)| {
                    let red_neighs = self.graph.red_neighbors_after_merge(u, v, false);
                    (red_neighs.cardinality(), (u, v))
                })
                .collect();
            if minimum.is_empty() {
                break;
            }
            minimum.sort();
            let min_red = minimum[0].0;
            let max_red = min_red + delta;

            for (reds, (u, v)) in minimum.into_iter() {
                if reds <= max_red {
                    if unvisited_nodes.unset_bit(u) && unvisited_nodes.unset_bit(v) {
                        self.graph.merge_node_into(u, v);
                        contraction_sequence.merge_node_into(u, v);
                        remaining_nodes.unset_bit(u);

                        unvisited_nodes.unset_bit(u);
                        unvisited_nodes.unset_bit(v);
                        tww = tww.max(self.graph.red_degrees().max().unwrap());
                    }
                } else {
                    break;
                }
            }
        }
        self.preprocessing_sequence.append(&contraction_sequence);
        self.preprocessing_sequence
            .add_unmerged_singletons(&self.graph);
        (tww, self.preprocessing_sequence)
    }

    pub fn get_node_min_sim(g: &G, node: u32) -> u32 {
        let mut bitset = g.neighbors_of_as_bitset(node);

        let neighbors = g.neighbors_of(node);
        let mut best_tww = std::u32::MAX;

        for n in neighbors {
            bitset.or(&g.neighbors_of_as_bitset(*n));
        }
        
        bitset.unset_bit(node);

        if bitset.cardinality() == 0 {
            return 0;
        }

        for n in bitset.iter() {
            let mut merge_g = g.clone();
            merge_g.merge_node_into(n, node);
            let tww = merge_g.red_degrees().max().unwrap();
            best_tww = best_tww.min(tww);
        }
        best_tww
    }

    pub fn next_best_move(g: &G, rem_nodes: &BitSet, ub: u32, played_level: u32) -> Option<(u32,u32)> {
        let mut minimum: Vec<_> = g
                .distance_two_pairs()
                .flat_map(|(u, v)| {
                    let mut merge_g = g.clone();
                    merge_g.merge_node_into(u, v);
                    let tww = merge_g.red_degrees().max().unwrap();
                    if tww <= ub-1 {
                        Some((tww, (u, v)))
                    }
                    else {
                        None
                    }
                })
                .collect();

            if minimum.is_empty() {
                return None;
            }

            minimum.sort();


            let mut min_move = None;
            let mut max_red = std::u32::MAX;
            let mut min_total_red_deg = std::u32::MAX;

            for (_, (u, v)) in minimum.iter() {
                let (max_reds, total_red_deg) = SweepingSolver::play_greedy_multiple_levels(
                    g,
                    (*u, *v),
                    rem_nodes.clone(),
                    played_level,
                );

                if (max_reds < max_red)
                    || (max_reds == max_red && total_red_deg < min_total_red_deg)
                {
                    max_red = max_reds;
                    min_move = Some((*u, *v));
                    min_total_red_deg = total_red_deg;
                }
            }
            return min_move;
    }
    

    pub fn collapse_onto_hard_nodes(mut self, ub: u32) -> (u32,ContractionSequence) {
        let mut remaining_nodes = self.preprocessing_sequence.remaining_nodes().unwrap();
        if remaining_nodes.cardinality() == 1 {
            return (0, self.preprocessing_sequence);
        }

        let mut hard_nodes : fxhash::FxHashSet<u32> = fxhash::FxHashSet::default();

        for x in 0..self.graph.number_of_nodes() {
            if SweepingSolver::<G>::get_node_min_sim(&self.graph, x) > ub-1 {
                hard_nodes.insert(x);
            }
        }

        let mut tww = 0;
        let mut contraction_sequence = ContractionSequence::new(self.graph.number_of_nodes());

        loop {
            let mut min_move = (0,0);
            let mut hard_node = 0;
            let mut possible_moves = Vec::new();

            // Try to improve near a hard node without hitting the upper bound
            for x in hard_nodes.iter() {
                let neighbors = self.graph.neighbors_of(*x);
                let sym = SweepingSolver::<G>::get_node_min_sim(&self.graph, *x);

                // First order collapsing
                for first in neighbors.iter() {
                    for second in neighbors.iter() {
                        if *first == *second {
                            continue;
                        }
                        let mut merge_g = self.graph.clone();
                        merge_g.merge_node_into(*first, *second);
                        let tww = merge_g.red_degrees().max().unwrap();
                        if tww <= ub-1 {
                            possible_moves.push((tww,*x,(*first,*second)));
                            min_move = (*first,*second);
                        }
                    }
                }


                // Try second order collapse
                for first in neighbors.iter() {
                    let nb = self.graph.neighbors_of(*first);

                    for second in nb.iter() {
                        if *first == *second  {
                            continue;
                        }
                        let mut merge_g = self.graph.clone();
                        merge_g.merge_node_into(*first, *second);
                        let tww = merge_g.red_degrees().max().unwrap();
                        if tww <= ub-1 && sym > SweepingSolver::<G>::get_node_min_sim(&merge_g, *x) {
                            possible_moves.push((tww,*x,(*first,*second)));
                        }
                    }
                }        
           }

           if !possible_moves.is_empty() {
                possible_moves.sort();

                min_move = possible_moves[0].2;
                let mut max_red = std::u32::MAX;
                let mut min_total_red_deg = std::u32::MAX;
    
                for (_,nd, (u, v)) in possible_moves.iter() {
                    let (max_reds, total_red_deg) = SweepingSolver::play_greedy_multiple_levels(
                        &self.graph,
                        (*u, *v),
                        remaining_nodes.clone(),
                        4,
                    );
    
                    if (max_reds < max_red)
                        || (max_reds == max_red && total_red_deg < min_total_red_deg)
                    {
                        max_red = max_reds;
                        min_move = (*u, *v);
                        hard_node = *nd;
                        min_total_red_deg = total_red_deg;
                    }
                }
           }

           if min_move != (0,0) {
                self.graph.merge_node_into(min_move.0, min_move.1);
                contraction_sequence.merge_node_into(min_move.0, min_move.1);
                remaining_nodes.unset_bit(min_move.0);
                tww = tww.max(self.graph.red_degrees().max().unwrap());
                if SweepingSolver::<G>::get_node_min_sim(&self.graph, hard_node) <= ub-1 {
                    hard_nodes.remove(&hard_node);
                }
            }
            else {
                let mut sample_best_moves = Vec::new();
                for i in 0..13 {
                    if let Some(result) = SweepingSolver::<G>::next_best_move(&self.graph, &remaining_nodes, ub, i) {
                        sample_best_moves.push(result);
                    }
                }
                
                if sample_best_moves.is_empty() {
                    println!("Remaining {}",remaining_nodes.cardinality());
                    return (ub+1,self.preprocessing_sequence);
                }

                let mut min_move = (0,0);
                let mut min_tww = std::u32::MAX;
                for x in sample_best_moves.into_iter() {
                    let mut gc_clone = self.graph.clone();
                    gc_clone.merge_node_into(x.0, x.1);
                    let tww = self.graph.red_degrees().max().unwrap();
                    if tww <= min_tww {
                        min_move = x;
                        min_tww = tww;
                    }
                }
                self.graph.merge_node_into(min_move.0, min_move.1);
                contraction_sequence.merge_node_into(min_move.0, min_move.1);
                remaining_nodes.unset_bit(min_move.0);
                tww = tww.max(self.graph.red_degrees().max().unwrap());
            }


            for x in 0..self.graph.number_of_nodes() {
                let sym = SweepingSolver::<G>::get_node_min_sim(&self.graph, x);
                if sym > ub-1 {
                    println!("Sym {} Node {}",sym,x);
                    hard_nodes.insert(x);
                }
            }

            if remaining_nodes.cardinality() <= 1 {
                break;
            }
        }

        self.preprocessing_sequence.append(&contraction_sequence);
        self.preprocessing_sequence
            .add_unmerged_singletons(&self.graph);
        (tww, self.preprocessing_sequence)
    }


//Pretty slow for larger graphs
pub fn solve_greedy_pairwise_fast_full_sym_desced(
    mut self,
    upper_bound: u32,
    played_level: u32
) -> (u32, ContractionSequence) {
    let mut remaining_nodes = self.preprocessing_sequence.remaining_nodes().unwrap();
    if remaining_nodes.cardinality() == 1 {
        return (0, self.preprocessing_sequence);
    }

    let mut tww = 0;
    let mut contraction_sequence = ContractionSequence::new(self.graph.number_of_nodes());

    loop {
        if remaining_nodes.cardinality() <= 1 {
            break;
        }

        let mut minimum: Vec<_> = self
            .graph
            .distance_two_pairs()
            .map(|(u, v)| {
                let mut cloned = self.graph.clone();
                cloned.merge_node_into(u,v);
                (cloned.red_degrees().max().unwrap(), (u, v))
            })
            .collect();

        if minimum.is_empty() {
            break;
        }

        minimum.sort();

        let min_red = minimum[0].0;

        let mut min_move = minimum[0].1;
        let mut max_red = std::u32::MAX;
        let mut min_total_red_deg = std::u32::MAX;

        let mut frontier_size = 0;
        for (red, (u, v)) in minimum.iter() {
            // Only allow uphill at most one
            if *red > upper_bound-1 {
                break;
            }
            frontier_size += 1;

            let (max_reds, total_red_deg) = SweepingSolver::play_greedy_multiple_levels_sym(
                &self.graph,
                (*u, *v),
                upper_bound,
                remaining_nodes.clone(),
                played_level,
            );

            if (max_reds < max_red)
                || (max_reds == max_red && total_red_deg < min_total_red_deg)
            {
                max_red = max_reds;
                min_move = (*u, *v);
                min_total_red_deg = total_red_deg;
            }
        }

        self.graph.merge_node_into(min_move.0, min_move.1);
        contraction_sequence.merge_node_into(min_move.0, min_move.1);
        remaining_nodes.unset_bit(min_move.0);
        tww = tww.max(self.graph.red_degrees().max().unwrap());
    }
    self.preprocessing_sequence.append(&contraction_sequence);
    let _ = self.preprocessing_sequence.remaining_nodes().unwrap();
    (tww, self.preprocessing_sequence)
}
    
    //Pretty slow for larger graphs
pub fn solve_greedy_pairwise_fast_full_sym_desced_only(
    mut self,
    upper_bound: u32,
    played_level: u32
) -> (u32, ContractionSequence) {
    let mut remaining_nodes = self.preprocessing_sequence.remaining_nodes().unwrap();
    if remaining_nodes.cardinality() == 1 {
        return (0, self.preprocessing_sequence);
    }

    let mut tww = 0;
    let mut contraction_sequence = ContractionSequence::new(self.graph.number_of_nodes());

    loop {
        if remaining_nodes.cardinality() <= 1 {
            break;
        }

        let mut minimum: Vec<_> = self
            .graph
            .distance_two_pairs()
            .map(|(u, v)| {
                let mut cloned = self.graph.clone();
                cloned.merge_node_into(u,v);
                (cloned.red_degrees().max().unwrap(), (u, v))
            })
            .collect();

        if minimum.is_empty() {
            break;
        }

        minimum.sort();

        let mut min_move = minimum[0].1;
        let mut min_total_red_deg = std::u32::MAX;

        for (red, (u, v)) in minimum.iter() {
            // Only allow uphill at most one
            if *red > upper_bound-1 {
                break;
            }

            let (max_reds, total_red_deg) = SweepingSolver::play_greedy_multiple_levels_sym_full(
                &self.graph,
                (*u, *v),
                upper_bound,
                remaining_nodes.clone(),
                played_level,
            );

            if (max_reds <= upper_bound-1) && (total_red_deg < min_total_red_deg)
            {
                min_move = (*u, *v);
                min_total_red_deg = total_red_deg;
            }
        }

        self.graph.merge_node_into(min_move.0, min_move.1);
        contraction_sequence.merge_node_into(min_move.0, min_move.1);
        remaining_nodes.unset_bit(min_move.0);
        tww = tww.max(self.graph.red_degrees().max().unwrap());
    }
    self.preprocessing_sequence.append(&contraction_sequence);
    let _ = self.preprocessing_sequence.remaining_nodes().unwrap();
    (tww, self.preprocessing_sequence)
}
}
