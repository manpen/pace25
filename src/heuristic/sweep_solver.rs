use log::trace;

use crate::prelude::*;
use std::fmt::Debug;

use super::partial_monte_carlo_search_tree::PartialMonteCarloSearchTree;

pub fn heuristic_solve<
    G: Clone
        + AdjacencyList
        + GraphEdgeOrder
        + ColoredAdjacencyList
        + ColoredAdjacencyTest
        + ColorFilter
        + Debug
        + GraphEdgeEditing,
>(
    graph: &G,
) -> (NumNodes, ContractionSequence) {
    trace!("Start heuristic solver");

    if graph.number_of_edges() < 1000 {
        let mut upper_bound = None;
        let mut best_solution = None;

        for (played_levels, enable_markov) in [
            (2, false),
            (1, false),
            (3, false),
            (4, false),
            (3, true),
            (2, true),
        ] {
            let (tww, cs) = SweepingSolver::new(graph.clone()).solve_greedy_pairwise_fast(
                None, // todo: actually, we should be able to pass the previous ub to the next iteration
                played_levels,
                enable_markov,
            );

            if tww < upper_bound.unwrap_or(NumNodes::MAX) {
                upper_bound = Some(tww);
                best_solution = Some(cs);
            }

            if tww == 0 {
                break;
            }
        }

        (upper_bound.unwrap(), best_solution.unwrap())
    } else {
        let sweeping_solver = SweepingSolver::new(graph.clone());

        let result_1 = sweeping_solver.solve_greedy_pairwise_fast(None, 2, false);
        let sweeping_solver = SweepingSolver::new(graph.clone());
        let result_2 = sweeping_solver.solve_greedy_pairwise_fast(None, 1, false);

        if result_1.0 <= result_2.0 {
            result_1
        } else {
            result_2
        }
    }
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
            + ColorFilter
            + GraphEdgeEditing,
    > SweepingSolver<G>
{
    pub fn new(graph: G) -> SweepingSolver<G> {
        let mut clone = graph;
        let mut preprocessing_sequence = ContractionSequence::new(clone.number_of_nodes());

        initial_pruning(&mut clone, &mut preprocessing_sequence);

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

        if remaining_nodes.cardinality() <= 1 {
            return (tww, minor);
        }

        let mut minimum: Vec<(u32, (u32, u32))> = cloned
            .distance_two_pairs()
            .map(|(u, v)| {
                let red_neighs = cloned.red_neighbors_after_merge(u, v, false);
                (red_neighs.cardinality(), (u, v))
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
                min_tww = min_tww.min(ub);
                break;
            }
            let min = SweepingSolver::<G>::play_complete_multiple_levels(
                &cloned,
                x.1,
                remaining_nodes.clone(),
                num_levels - 1,
                ub,
            );
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
                    default_pruning(&mut cloned, tww, &mut contraction);
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
    pub fn solve_greedy_pairwise_fast(
        mut self,
        upper_bound: Option<u32>,
        played_level: u32,
        enable_markov: bool,
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
                    let red_neighs = self.graph.red_neighbors_after_merge(u, v, false);
                    (red_neighs.cardinality(), (u, v))
                })
                .collect();

            if minimum.is_empty() {
                break;
            }

            minimum.sort();

            let min_red = minimum[0].0;

            let mut min_move = (0, 0);
            let mut max_red = std::u32::MAX;
            let mut min_total_red_deg = std::u32::MAX;

            let mut frontier_size = 0;
            for (red, (u, v)) in minimum.iter() {
                // Only allow uphill at most one
                if *red > upper_bound.map(|x| x - 1).unwrap_or(min_red + 1) {
                    break;
                }
                frontier_size += 1;

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

            if min_move == (0, 0) {
                break;
            }

            self.graph.merge_node_into(min_move.0, min_move.1);
            contraction_sequence.merge_node_into(min_move.0, min_move.1);
            remaining_nodes.unset_bit(min_move.0);
            tww = tww.max(self.graph.red_degrees().max().unwrap());

            default_pruning(&mut self.graph, tww, &mut contraction_sequence);
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
}
