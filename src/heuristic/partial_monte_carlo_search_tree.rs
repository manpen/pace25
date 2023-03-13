use rand::{rngs::ThreadRng, thread_rng, Rng};

use crate::prelude::{
    AdjacencyList, ColoredAdjacencyList, ColoredAdjacencyTest, DistancePairs, GraphEdgeEditing,
    GraphEdgeOrder,
};
use std::fmt::Debug;

pub struct PartialMonteCarloSearchTree<'a, G> {
    graph: G,
    choices: &'a Vec<(u32, (u32, u32))>,
    max_depth: u32,

    min_tww: u32,
    min_move: (u32, u32),
    upper_bound: u32,
}

impl<
        'a,
        G: Clone
            + AdjacencyList
            + GraphEdgeOrder
            + ColoredAdjacencyList
            + ColoredAdjacencyTest
            + Debug
            + GraphEdgeEditing,
    > PartialMonteCarloSearchTree<'a, G>
{
    pub fn new(
        graph: &G,
        choices: &'a Vec<(u32, (u32, u32))>,
        max_depth: u32,
        upper_bound: u32,
    ) -> PartialMonteCarloSearchTree<'a, G> {
        PartialMonteCarloSearchTree {
            graph: graph.clone(),
            choices,
            max_depth,
            min_tww: graph.red_degrees().max().unwrap(),
            min_move: choices[0].1,
            upper_bound,
        }
    }

    pub fn get_rank_of_3rd_unique_value(choices: &Vec<(u32, (u32, u32))>) -> u32 {
        let mut rank_3_end = choices.len();

        if choices.len() > 3 {
            let mut value = choices[0].0;
            let mut counter = 0;

            for (id, x) in choices.iter().enumerate() {
                if value < x.0 {
                    counter += 1;
                    value = x.0;
                    if counter == 3 {
                        rank_3_end = id;
                        break;
                    }
                }
            }
        }
        rank_3_end as u32
    }

    pub fn sample_move(
        generator: &mut ThreadRng,
        choices: &Vec<(u32, (u32, u32))>,
        min_choice: u32,
        ub: u32,
    ) -> usize {
        let mut restarts = 0;
        loop {
            if restarts > 5 {
                return 0;
            }
            let generated = generator.gen::<f64>();
            let random_choice = (((choices.len() - 1) as f64) * generated) as usize;
            if choices[random_choice].0 >= ub {
                restarts += 1;
                continue;
            }
            if choices[random_choice].0 > min_choice {
                let coin_toss = generator.gen::<f64>();
                if coin_toss > (1.0 / ((1 + (choices[random_choice].0 - min_choice)) as f64)) {
                    restarts += 1;
                    continue;
                }
            }
            return random_choice;
        }
    }

    pub fn play_games(&mut self, mut number_of_games: u32) {
        // Expect to have at least 1 choice
        let min_choice = self.choices[0].0;
        let mut generator = thread_rng();

        // No need to decide if there is only one choice
        if self.choices.len() == 1 || min_choice >= self.upper_bound {
            self.min_move = self.choices[0].1;
            return;
        }

        let mut min_tww = self.min_tww;
        let mut min_position = 0;

        'outer: while number_of_games > 0 {
            let random_choice = PartialMonteCarloSearchTree::<'a, G>::sample_move(
                &mut generator,
                self.choices,
                min_choice,
                self.upper_bound,
            );

            let mut game_graph = self.graph.clone();
            let initial_move = self.choices[random_choice].1;

            game_graph.merge_node_into(initial_move.0, initial_move.1);
            let mut tww = game_graph.red_degrees().max().unwrap();

            let mut cnt = 0;
            while cnt < self.max_depth {
                let mut choices: Vec<_> = game_graph
                    .distance_two_pairs()
                    .map(|(u, v)| {
                        let red_neighs = self.graph.red_neighbors_after_merge(u, v, false);
                        (red_neighs.cardinality(), (u, v))
                    })
                    .collect();

                if choices.is_empty() {
                    break;
                }

                choices.sort();

                let current_minimal = choices[0].0;

                let sample_move = PartialMonteCarloSearchTree::<'a, G>::sample_move(
                    &mut generator,
                    &choices,
                    current_minimal,
                    self.upper_bound,
                );

                game_graph.merge_node_into(choices[sample_move].1 .0, choices[sample_move].1 .1);
                tww = tww.max(game_graph.red_degrees().max().unwrap());
                if tww >= self.upper_bound {
                    number_of_games -= 1;
                    continue 'outer;
                }

                cnt += 1;
            }

            if min_tww < tww {
                min_tww = tww;
                min_position = random_choice;
            }
            number_of_games -= 1;
        }

        if min_tww < self.min_tww {
            self.min_tww = min_tww;
            self.min_move = self.choices[min_position].1;
        }
    }

    pub fn into_best_choice(self) -> (u32, u32) {
        self.min_move
    }

    pub fn into_best_choice_with_tww(self) -> (u32, (u32, u32)) {
        (self.min_tww, self.min_move)
    }
}
