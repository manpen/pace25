use crate::prelude::*;

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

pub fn heuristic_solve<G: FullfledgedGraph>(graph: &G) -> (u32, ContractionSequence) {
    let clone = graph.clone();
    if clone.number_of_edges() < 1000 && clone.number_of_nodes() < 200 {
        let sweeping_solver = SweepingSolver::new(clone.clone());
        // Since no upper bound is set it is guaranteed to return a solution
        let mut result = sweeping_solver
            .solve_greedy_pairwise_fast(None, 0, 1)
            .unwrap();
        for i in 1..11 {
            let sweeping_solver_i = SweepingSolver::new(clone.clone());
            let result_i = sweeping_solver_i.solve_greedy_pairwise_fast(Some(result.0), i, 1);
            if let Some(result_i) = result_i
                && result_i.0 <= result.0 {
                result = result_i;
            }
        }

        for i in 0..13 {
            let sweeping_solver_i = SweepingSolver::new(clone.clone());
            let result_i = sweeping_solver_i.solve_greedy_pairwise_fast_full(Some(result.0), i, 1);
            if let Some(result_i) = result_i
                && result_i.0 <= result.0 {
                result = result_i;
            }
        }

        // Improves graph 032 && 018 therefore make it faster then it would be good!
        for i in 0..8 {
            let sweeping_solver_i = SweepingSolver::new(clone.clone());
            let result_i =
                sweeping_solver_i.solve_greedy_pairwise_fast_full_sym_descend_only(result.0, i);
            if let Some(result_i) = result_i
                && result_i.0 <= result.0 {
                result = result_i;
            }
        }

        result
    } else if clone.number_of_edges() < 6000 && clone.number_of_nodes() < 1000 {
        let sweeping_solver = SweepingSolver::new(clone.clone());

        let result_1 = sweeping_solver
            .solve_greedy_pairwise_fast(None, 2, 1)
            .unwrap();
        let sweeping_solver = SweepingSolver::new(clone);
        let result_2 = sweeping_solver.solve_greedy_pairwise_fast(Some(result_1.0), 1, 1);

        if let Some(result_2) = result_2
            && result_1.0 > result_2.0 {
            result_2
        } else {
            result_1
        }
    } else {
        let sweeping_solver = SweepingSolver::new(clone);
        sweeping_solver.solve_greedy()
    }
}

pub struct SweepingSolver<G> {
    graph: G,
    preprocessing_sequence: ContractionSequence,
}

impl<G: FullfledgedGraph> SweepingSolver<G> {
    pub fn new(graph: G) -> SweepingSolver<G> {
        let mut clone = graph;
        let mut preprocessing_sequence = ContractionSequence::new(clone.number_of_nodes());

        // Preprocess the graph
        Kernelization::new(&mut clone, &mut preprocessing_sequence).run_first_round();

        SweepingSolver {
            graph: clone,
            preprocessing_sequence,
        }
    }

    pub fn new_without_preprocessing(graph: G) -> SweepingSolver<G> {
        // Sidestep preprocessing if it was made before
        let clone = graph.clone();
        SweepingSolver {
            graph: clone,
            preprocessing_sequence: ContractionSequence::new(graph.number_of_nodes()),
        }
    }

    // Solve by iterating over all remaining nodes and contract every node which has a neighbor in the
    // 2-Neighborhood which permits a contraction which limits the red degree to at most k
    // Increase k by one if no contraction can be found which induces a red degree <= k
    // Is performing worse than the normal greedy solver which evaluates the position after every move but is
    // faster for large graphs. Since no upper bound is given it is guaranteed to return a valid contraction sequence
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
            for node in remaining_nodes.clone().iter_set_bits() {
                let neighbors = self.graph.neighbors_of(node);
                let mut bitset = self.graph.neighbors_of_as_bitset(node);
                for neighbor in neighbors {
                    bitset |= &self.graph.neighbors_of_as_bitset(neighbor);
                }
                bitset.clear_bit(node);
                // No neighbors?
                if bitset.cardinality() == 0 {
                    remaining_nodes.clear_bit(node);
                    continue;
                }

                for neighbors in bitset.iter_set_bits() {
                    let new_red_deg = self.graph.red_degree_after_merge(neighbors, node);
                    if new_red_deg <= allowed_tww {
                        merged = true;
                        self.graph.merge_node_into(neighbors, node);
                        tww = tww.max(self.graph.red_degrees().max().unwrap());
                        remaining_nodes.clear_bit(neighbors);
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

    // Play multiple levels greedy by always choosing the move which induces the least red degree on the surviving node
    // This function is mainly used to evaluate the viability of a contraction several steps down the road
    fn play_greedy_multiple_levels(
        graph: &G,
        first_move: (u32, u32),
        mut remaining_nodes: BitSet,
        num_levels: u32,
    ) -> (u32, u32) {
        let mut cloned = graph.clone();
        cloned.merge_node_into(first_move.0, first_move.1);
        remaining_nodes.clear_bit(first_move.0);
        let mut tww = cloned.red_degrees().max().unwrap();

        for _ in 0..num_levels {
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
                remaining_nodes.clear_bit(u);

                tww = tww.max(cloned.red_degrees().max().unwrap());
            } else {
                break;
            }
        }
        (tww, cloned.red_degrees().sum())
    }

    // Fast calculation of the total degree after the contraction as well as maximal induced tww by this move
    #[inline]
    pub fn red_degree_total_deg_after_merge(
        g: &G,
        initial_deg: u32,
        next_move: (u32, u32),
        upper_bound: u32,
    ) -> (u32, u32) {
        let mut red_neighbors_len = g.red_degree_after_merge(next_move.0, next_move.1);
        if red_neighbors_len > upper_bound - 1 {
            return (red_neighbors_len, initial_deg);
        }
        let mut n_0 = g.neighbors_of_as_bitset(next_move.0);
        let n_1 = g.neighbors_of_as_bitset(next_move.1);

        let offset = if n_0.get_bit(next_move.1) { -2 } else { 0 };

        // TODO: The following two lines are a no-op. Why are they here?
        n_0 ^= &n_1;
        n_0 &= &n_1;

        let red_neighbors_new = g.red_neighbors_after_merge(next_move.0, next_move.1, true);

        let delta_deg_a: i32 = n_0.cardinality() as i32 * 2 + offset;

        let delta_deg_b = -2 * (g.degree_of(next_move.0) as i32);

        let red_neigh_before = g.red_neighbors_of_as_bitset(next_move.0);
        for x in red_neighbors_new.iter_set_bits() {
            if !red_neigh_before.get_bit(x) {
                red_neighbors_len = red_neighbors_len.max(g.red_degree_of(x) + 1);
                if red_neighbors_len > upper_bound - 1 {
                    return (red_neighbors_len, initial_deg);
                }
            }
        }

        let total_degree_of_new_graph =
            (initial_deg as i32 + delta_deg_a + delta_deg_b).max(0) as u32;

        (red_neighbors_len, total_degree_of_new_graph)
    }

    fn play_greedy_multiple_levels_sym_full(
        graph: &G,
        first_move: (u32, u32),
        upper_bound: u32,
        mut remaining_nodes: BitSet,
        num_levels: u32,
    ) -> (u32, u32) {
        let mut cloned = graph.clone();
        cloned.merge_node_into(first_move.0, first_move.1);
        remaining_nodes.clear_bit(first_move.0);
        let mut tww = cloned.red_degrees().max().unwrap();
        let mut initial_deg: u32 = cloned.degrees().sum();
        let mut min_sim: Vec<(u32, u32)> = vec![(0, 0); cloned.number_of_nodes() as usize];

        let mut round = 1;
        for _ in 0..num_levels {
            round += 1;
            if remaining_nodes.cardinality() <= 1 {
                break;
            }

            let minimum = cloned
                .distance_two_pairs()
                .flat_map(|(u, v)| {
                    let result = Self::red_degree_total_deg_after_merge(
                        &cloned,
                        initial_deg,
                        (u, v),
                        upper_bound,
                    );
                    (result.0 < upper_bound)
                        .then(|| {
                            if min_sim[u as usize].0 < round || min_sim[u as usize].1 > result.0 {
                                min_sim[u as usize] = (round, result.0);
                            }
                            if min_sim[v as usize].0 < round || min_sim[v as usize].1 > result.0 {
                                min_sim[v as usize] = (round, result.0);
                            }
                            (result.1, (u, v))
                        })
                        .or_else(|| {
                            if min_sim[u as usize].0 < round || min_sim[u as usize].1 > result.0 {
                                min_sim[u as usize] = (round, result.0);
                            }
                            if min_sim[v as usize].0 < round || min_sim[v as usize].1 > result.0 {
                                min_sim[v as usize] = (round, result.0);
                            }
                            None
                        })
                })
                .min();

            if let Some((_, (u, v))) = minimum {
                cloned.merge_node_into(u, v);
                remaining_nodes.clear_bit(u);

                tww = tww.max(cloned.red_degrees().max().unwrap());
                initial_deg = cloned.degrees().sum();
            } else {
                break;
            }
        }

        let sym_sum_fast = min_sim
            .iter()
            .flat_map(|(_, v)| {
                if *v > upper_bound - 1 {
                    Some(*v - (upper_bound - 1))
                } else {
                    None
                }
            })
            .sum();
        (tww, sym_sum_fast)
    }

    //Pretty slow for larger graphs
    pub fn solve_greedy_pairwise_fast_full(
        mut self,
        upper_bound: Option<u32>,
        played_level: u32,
        max_delta_beg: u32,
    ) -> Option<(u32, ContractionSequence)> {
        let mut remaining_nodes = self.preprocessing_sequence.remaining_nodes().unwrap();
        if remaining_nodes.cardinality() == 1 {
            return Some((0, self.preprocessing_sequence));
        }

        let mut tww = 0;
        let mut contraction_sequence = ContractionSequence::new(self.graph.number_of_nodes());

        loop {
            if remaining_nodes.cardinality() <= 1 {
                break;
            }

            let minimum: Vec<_> = if let Some(ub) = upper_bound {
                let mut counter = 0;
                let res: Vec<_> = self
                    .graph
                    .distance_two_pairs()
                    .flat_map(|(u, v)| {
                        let mut cloned = self.graph.clone();
                        cloned.merge_node_into(u, v);
                        let deg = cloned.red_degrees().max().unwrap();
                        if deg > ub - 1 {
                            counter += 1;
                            None
                        } else {
                            Some((deg, (u, v)))
                        }
                    })
                    .collect();

                if counter > 0 && res.is_empty() {
                    return None;
                }
                // No sorting needed since every move is checked anyways
                res
            } else {
                let mut res: Vec<_> = self
                    .graph
                    .distance_two_pairs()
                    .map(|(u, v)| {
                        let mut cloned = self.graph.clone();
                        cloned.merge_node_into(u, v);
                        (cloned.red_degrees().max().unwrap(), (u, v))
                    })
                    .collect();

                if !res.is_empty() {
                    res.sort();
                }
                res
            };

            if minimum.is_empty() {
                break;
            }

            let min_red = minimum[0].0;

            let mut min_move = minimum[0].1;
            let mut max_red = std::u32::MAX;
            let mut min_total_red_deg = std::u32::MAX;

            for (red, (u, v)) in minimum.iter() {
                // Only allow uphill at most one
                if *red
                    > upper_bound
                        .map(|x| x - 1)
                        .unwrap_or(min_red + max_delta_beg)
                {
                    break;
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

            self.graph.merge_node_into(min_move.0, min_move.1);
            contraction_sequence.merge_node_into(min_move.0, min_move.1);
            remaining_nodes.clear_bit(min_move.0);
            tww = tww.max(self.graph.red_degrees().max().unwrap());
        }
        self.preprocessing_sequence.append(&contraction_sequence);
        let _ = self.preprocessing_sequence.remaining_nodes().unwrap();
        Some((tww, self.preprocessing_sequence))
    }

    //Pretty slow for larger graphs
    pub fn solve_greedy_pairwise_fast(
        mut self,
        upper_bound: Option<u32>,
        played_level: u32,
        max_delta_beg: u32,
    ) -> Option<(u32, ContractionSequence)> {
        let mut remaining_nodes = self.preprocessing_sequence.remaining_nodes().unwrap();
        if remaining_nodes.cardinality() == 1 {
            return Some((0, self.preprocessing_sequence));
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

            for (red, (u, v)) in minimum.iter() {
                // Only allow uphill at most one
                if *red
                    > upper_bound
                        .map(|x| x - 1)
                        .unwrap_or(min_red + max_delta_beg)
                {
                    break;
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

            self.graph.merge_node_into(min_move.0, min_move.1);
            contraction_sequence.merge_node_into(min_move.0, min_move.1);
            remaining_nodes.clear_bit(min_move.0);
            tww = tww.max(self.graph.red_degrees().max().unwrap());
            if let Some(ub) = upper_bound.as_ref()
                && *ub <= tww {
                    return None;
            }
        }
        self.preprocessing_sequence.append(&contraction_sequence);
        let _ = self.preprocessing_sequence.remaining_nodes().unwrap();
        Some((tww, self.preprocessing_sequence))
    }

    // Similiar to solve_greedy but allows the contraction of all nodes with induced red degree of survivor within delta of the best
    // induced red degree by a contraction. Solves really fast but the error can be quite large
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
                    if unvisited_nodes.clear_bit(u) && unvisited_nodes.clear_bit(v) {
                        self.graph.merge_node_into(u, v);
                        contraction_sequence.merge_node_into(u, v);
                        remaining_nodes.clear_bit(u);

                        unvisited_nodes.clear_bit(u);
                        unvisited_nodes.clear_bit(v);
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

    // Calculate the minimal induced tww when contracting this node. Can be used to define hard nodes and easy nodes and to measure
    // remaining problem complexity
    pub fn get_node_min_sim(g: &G, node: u32) -> u32 {
        let mut bitset = g.neighbors_of_as_bitset(node);

        let mut best_tww = std::u32::MAX;

        for n in g.neighbors_of(node) {
            bitset |= &g.neighbors_of_as_bitset(n);
        }

        bitset.clear_bit(node);

        if bitset.cardinality() == 0 {
            return 0;
        }

        for n in bitset.iter_set_bits() {
            let mut merge_g = g.clone();
            merge_g.merge_node_into(n, node);
            let tww = merge_g.red_degrees().max().unwrap();
            best_tww = best_tww.min(tww);
        }
        best_tww
    }

    // Calculates the minimum similiarity of a node to its 2 Neighborhood
    // Does not calculate induced tww.
    pub fn get_node_total_sim(g: &G, node: u32) -> u32 {
        let mut bitset = g.neighbors_of_as_bitset(node);

        let mut min_dist = std::u32::MAX;

        for n in g.neighbors_of(node) {
            bitset |= &g.neighbors_of_as_bitset(n);
        }

        bitset.clear_bit(node);

        if bitset.cardinality() == 0 {
            return 0;
        }

        let base = g.neighbors_of_as_bitset(node);
        for n in bitset.iter_set_bits() {
            let mut nb = g.neighbors_of_as_bitset(n);
            nb ^= &base;

            min_dist = min_dist.min(nb.cardinality());
        }
        min_dist
    }

    // Get the next best move which is not larger than the upper bound and evaluate each move by playing multiple levels
    pub fn next_best_move(
        g: &G,
        rem_nodes: &BitSet,
        ub: u32,
        played_level: u32,
    ) -> Option<(u32, u32)> {
        let mut minimum: Vec<_> = g
            .distance_two_pairs()
            .flat_map(|(u, v)| {
                let mut merge_g = g.clone();
                merge_g.merge_node_into(u, v);
                let tww = merge_g.red_degrees().max().unwrap();
                if tww < ub {
                    Some((tww, (u, v)))
                } else {
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

            if (max_reds < max_red) || (max_reds == max_red && total_red_deg < min_total_red_deg) {
                max_red = max_reds;
                min_move = Some((*u, *v));
                min_total_red_deg = total_red_deg;
            }
        }
        min_move
    }

    // Try to improve heuristics by minimizing the sum of node similarities across moves
    // should in theory provide a good measure of current problem complexity
    fn play_greedy_multiple_levels_nn(
        graph: &G,
        first_move: (u32, u32),
        ub: u32,
        mut remaining_nodes: BitSet,
        mut num_levels: u32,
    ) -> u32 {
        let mut cloned = graph.clone();
        cloned.merge_node_into(first_move.0, first_move.1);
        remaining_nodes.clear_bit(first_move.0);
        let mut objective: u32 = (0..graph.len())
            .map(|x| Self::get_node_total_sim(graph, x as u32))
            .sum();

        loop {
            if remaining_nodes.cardinality() <= 1 {
                break;
            }

            let minimum: Option<(u32, (u32, u32))> = cloned
                .distance_two_pairs()
                .flat_map(|(u, v)| {
                    let mut graph = cloned.clone();
                    graph.merge_node_into(u, v);

                    let red_d = graph.red_degrees().max().unwrap();
                    if red_d > ub - 1 {
                        None
                    } else {
                        let x = (0..graph.len())
                            .map(|x| Self::get_node_total_sim(&graph, x as u32))
                            .sum();
                        Some((x, (u, v)))
                    }
                })
                .min();

            if let Some((ob, (u, v))) = minimum {
                cloned.merge_node_into(u, v);
                remaining_nodes.clear_bit(u);
                objective = objective.max(ob);
            } else {
                if cloned.distance_two_pairs().count() > 0 {
                    return std::u32::MAX;
                }
                break;
            }

            num_levels -= 1;
            if num_levels == 0 {
                break;
            }
        }

        objective
    }

    // Greedily optimize node similarity across the graph, by playing at most `level` for each move
    pub fn greedy_optimize_sim_nn(mut self, ub: u32, level: u32) -> (u32, ContractionSequence) {
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

            let minimum: Vec<(u32, (u32, u32))> = self
                .graph
                .distance_two_pairs()
                .flat_map(|(u, v)| {
                    let mut graph = self.graph.clone();
                    graph.merge_node_into(u, v);

                    let red_d = graph.red_degrees().max().unwrap();
                    if red_d > ub - 1 {
                        None
                    } else {
                        let x = (0..graph.len())
                            .map(|x| Self::get_node_total_sim(&graph, x as u32))
                            .sum();
                        Some((x, (u, v)))
                    }
                })
                .collect();

            if minimum.is_empty() && self.graph.distance_two_pairs().count() > 0 {
                return (ub + 1, self.preprocessing_sequence);
            }

            let mut min_move = (0, 0);
            let mut min_objective = std::u32::MAX;

            for (_, (u, v)) in minimum.iter() {
                let objective = SweepingSolver::play_greedy_multiple_levels_nn(
                    &self.graph,
                    (*u, *v),
                    ub,
                    remaining_nodes.clone(),
                    level,
                );

                if objective < min_objective {
                    min_objective = objective;
                    min_move = (*u, *v);
                }
            }

            // No valid move anymore
            if min_move == (0, 0) {
                return (ub + 1, self.preprocessing_sequence);
            }

            self.graph.merge_node_into(min_move.0, min_move.1);
            contraction_sequence.merge_node_into(min_move.0, min_move.1);
            remaining_nodes.clear_bit(min_move.0);
            tww = tww.max(self.graph.red_degrees().max().unwrap());
        }

        self.preprocessing_sequence.append(&contraction_sequence);
        self.preprocessing_sequence
            .add_unmerged_singletons(&self.graph);
        (tww, self.preprocessing_sequence)
    }

    // Solve greedily by optimizing for tww as first criterion and total degree as second criterion
    pub fn solve_greedy_pairwise_fast_full_sym_descend_only(
        mut self,
        upper_bound: u32,
        played_level: u32,
    ) -> Option<(u32, ContractionSequence)> {
        let mut remaining_nodes = self.preprocessing_sequence.remaining_nodes().unwrap();
        if remaining_nodes.cardinality() == 1 {
            return Some((0, self.preprocessing_sequence));
        }

        let mut tww = 0;
        let mut contraction_sequence = ContractionSequence::new(self.graph.number_of_nodes());

        loop {
            if remaining_nodes.cardinality() <= 1 {
                break;
            }

            let mut counter = 0;
            let minimum: Vec<_> = self
                .graph
                .distance_two_pairs()
                .flat_map(|(u, v)| {
                    let mut cloned = self.graph.clone();
                    cloned.merge_node_into(u, v);
                    counter += 1;
                    let deg = cloned.red_degrees().max().unwrap();
                    if deg > upper_bound - 1 {
                        None
                    } else {
                        Some((deg, (u, v)))
                    }
                })
                .collect();

            if minimum.is_empty() {
                if counter > 0 {
                    return None;
                }
                break;
            }

            let mut min_move = minimum[0].1;
            let mut min_total_red_deg = std::u32::MAX;

            for (_, (u, v)) in minimum.iter() {
                let (max_reds, total_red_deg) =
                    SweepingSolver::play_greedy_multiple_levels_sym_full(
                        &self.graph,
                        (*u, *v),
                        upper_bound,
                        remaining_nodes.clone(),
                        played_level,
                    );

                if (max_reds < upper_bound) && (total_red_deg < min_total_red_deg) {
                    min_move = (*u, *v);
                    min_total_red_deg = total_red_deg;
                }
            }

            self.graph.merge_node_into(min_move.0, min_move.1);
            contraction_sequence.merge_node_into(min_move.0, min_move.1);
            remaining_nodes.clear_bit(min_move.0);
            tww = tww.max(self.graph.red_degrees().max().unwrap());
        }
        self.preprocessing_sequence.append(&contraction_sequence);
        let _ = self.preprocessing_sequence.remaining_nodes().unwrap();
        Some((tww, self.preprocessing_sequence))
    }
}
