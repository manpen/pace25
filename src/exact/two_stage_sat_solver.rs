use crate::{
    graph::{
        twin_width_sat_encoding::TwinWidthSatEncoding, AdjacencyList, ColoredAdjacencyList,
        ColoredAdjacencyTest, GraphEdgeEditing, GraphEdgeOrder,
    },
    heuristic::monte_carlo_search_tree::timeout_monte_carlo_search_tree_solver_preprocessed,
    prelude::{Connectivity, Getter},
};
use std::fmt::Debug;

use super::{
    contraction_sequence::ContractionSequence,
    reductions::{prune_leaves, prune_twins},
};

/*
instances/exact-public/exact_002.gr                |     20 |       69 |      6 (     6) |   2.806 ms
instances/exact-public/exact_004.gr                |     25 |      181 |      7 (     7) |  96.802 ms
instances/exact-public/exact_006.gr                |     28 |      131 |      7 (     7) | 1363.111 ms
instances/exact-public/exact_008.gr                |     28 |      210 |     10 (    10) |   6.342 ms
instances/exact-public/exact_010.gr                |     28 |      235 |      6 (     6) | 464.713 ms
instances/exact-public/exact_012.gr                |     29 |      180 |      8 (     8) | 565.431 ms
instances/exact-public/exact_014.gr                |     30 |      175 |      8 (     8) | 619.550 ms
instances/exact-public/exact_016.gr                |     30 |      195 |      8 (   195) | 4074.015 ms (x)
instances/exact-public/exact_018.gr                |     31 |       52 |      3 (     3) | 1220.587 ms
instances/exact-public/exact_020.gr                |     32 |       90 |      5 (     5) | 1282.447 ms
instances/exact-public/exact_022.gr                |     33 |      135 |      5 (     ?) |  42.209 ms
instances/exact-public/exact_024.gr                |     40 |       89 |      5 (     ?) | 5102.043 ms (x)
instances/exact-public/exact_026.gr                |     44 |       95 |      4 (     ?) | 2968.960 ms (x)
instances/exact-public/exact_028.gr                |     47 |       79 |      3 (     ?) | 1180.117 ms
instances/exact-public/exact_030.gr                |     48 |       78 |      3 (     ?) | 590.173 ms
instances/exact-public/exact_032.gr                |     48 |       94 |      3 (     ?) | 3358.400 ms (x)
instances/exact-public/exact_034.gr                |     51 |      240 |      4 (     4) | 666.075 ms
instances/exact-public/exact_036.gr                |     52 |       87 |      3 (     ?) | 903.383 ms
instances/exact-public/exact_038.gr                |     52 |      141 |      3 (     ?) | 460.801 ms
instances/exact-public/exact_040.gr                |     54 |       85 |      3 (     ?) | 8852.257 ms (x)
instances/exact-public/exact_042.gr                |     54 |      124 |      3 (     ?) | 352.890 ms
instances/exact-public/exact_044.gr                |     56 |       64 |      2 (     2) | 522.085 ms
instances/exact-public/exact_046.gr                |     57 |      107 |      3 (     ?) | 1136.419 m
instances/exact-public/exact_048.gr                |     60 |      103 |      3 (     ?) | 791.709 ms
instances/exact-public/exact_050.gr                |     62 |      108 |      2 (     ?) | 1149.333 ms
instances/exact-public/exact_052.gr                |     66 |      122 |      3 (     ?) | 15294.252 ms (x)
kissat (no pruning, no connected components, first stage 300ms)
 */

/*
instances/exact-public/exact_002.gr                |     20 |       69 |      6 (     6) |  10.854 ms
instances/exact-public/exact_004.gr                |     25 |      181 |      7 (     7) | 158.481 ms
instances/exact-public/exact_006.gr                |     28 |      131 |      7 (     7) | 458.251 ms
instances/exact-public/exact_008.gr                |     28 |      210 |     10 (    10) |  66.925 ms
instances/exact-public/exact_010.gr                |     28 |      235 |      6 (     6) | 487.053 ms
instances/exact-public/exact_012.gr                |     29 |      180 |      8 (     8) | 757.704 ms
instances/exact-public/exact_014.gr                |     30 |      175 |      8 (     8) | 947.885 ms
instances/exact-public/exact_016.gr                |     30 |      195 |      8 (     8) | 3803.858 ms (x)
instances/exact-public/exact_018.gr                |     31 |       52 |      3 (     3) | 1231.356 ms

CadiCal (with pruning, connected components, incremental solving, first stage 300ms)
 */
pub struct TwoStageSatSolver<G> {
    graph: G,
    preprocessing_sequence: ContractionSequence,
    time: std::time::Duration,
}

impl<
        G: Clone
            + AdjacencyList
            + GraphEdgeOrder
            + ColoredAdjacencyList
            + ColoredAdjacencyTest
            + GraphEdgeEditing
            + Debug,
    > TwoStageSatSolver<G>
{
    pub fn new(graph: &G, first_stage_timeout: std::time::Duration) -> TwoStageSatSolver<G> {
        let mut preprocessing_sequence = ContractionSequence::new(graph.number_of_nodes());

        let mut clone = graph.clone();

        prune_leaves(&mut clone, &mut preprocessing_sequence);
        prune_twins(&mut clone, &mut preprocessing_sequence);

        TwoStageSatSolver {
            graph: clone,
            preprocessing_sequence,
            time: first_stage_timeout,
        }
    }

    pub fn solve(&mut self) -> (u32, ContractionSequence) {
        if self
            .preprocessing_sequence
            .remaining_nodes()
            .unwrap()
            .cardinality()
            <= 1
        {
            return (0, self.preprocessing_sequence.clone());
        }

        let (sol, heu_seq, _) =
            timeout_monte_carlo_search_tree_solver_preprocessed(&self.graph, self.time);

        // Try to improve upon the existing solution
        if sol != 0 {
            let part = self.graph.partition_into_connected_components(true);
            if part.number_of_classes() != 1
                && 10 * part.number_of_unassigned() >= self.graph.number_of_nodes()
            {
                let mut max_tww = sol - 1;
                let mut total_solution = self.preprocessing_sequence.clone();

                for (subgraph, mapper) in part.split_into_subgraphs(&self.graph) {
                    let mut sat_encoding = TwinWidthSatEncoding::new_complement_graph(&subgraph);
                    if let Some((size, seq)) = sat_encoding.solve_kissat(sol - 1) {
                        max_tww = max_tww.max(size);
                        if max_tww == sol {
                            break;
                        }
                        for &(rem, sur) in seq.merges() {
                            total_solution.merge_node_into(
                                mapper.old_id_of(rem).unwrap(),
                                mapper.old_id_of(sur).unwrap(),
                            );
                        }
                    } else {
                        // We need to improve the previous bound on every connected component
                        max_tww = sol + 1;
                        break;
                    }
                }

                // Only if we found a better solution take it, else go with the heuristic solution
                if max_tww < sol {
                    total_solution.add_unmerged_singletons(&self.graph);
                    return (max_tww, self.preprocessing_sequence.clone());
                }
            }
            // No connected components attempt solving on full graph
            else {
                let mut sat_encoding = TwinWidthSatEncoding::new_complement_graph(&self.graph);
                if let Some((tww, seq)) = sat_encoding.solve_kissat(sol - 1) {
                    self.preprocessing_sequence.append(&seq);
                    self.preprocessing_sequence
                        .add_unmerged_singletons(&self.graph);

                    //Sequence should always be legal
                    let _ = self.preprocessing_sequence.remaining_nodes().unwrap();
                    return (tww, self.preprocessing_sequence.clone());
                }
            }
        }

        // Could not find a better solution take the heuristic, since no better solution is found by the exact solvers
        // the heuristic is guaranteed to be optimal
        self.preprocessing_sequence.append(&heu_seq);
        self.preprocessing_sequence
            .add_unmerged_singletons(&self.graph);

        //Sequence should always be legal
        self.preprocessing_sequence.remaining_nodes().unwrap();
        (sol, self.preprocessing_sequence.clone())
    }
}
