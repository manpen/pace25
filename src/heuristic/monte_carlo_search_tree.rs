use fxhash::{FxHashMap, FxHashSet};
use std::{collections::hash_map::Entry, time::Duration};

use rand::Rng;
use std::fmt::Debug;

use crate::prelude::{
    reductions::{prune_leaves, prune_twins},
    *,
};

/// Runs the search tree for the given period of time, will return the best score the contraction sequence and the number of games played
pub fn timeout_monte_carlo_search_tree_solver<
    G: Clone
        + AdjacencyList
        + GraphEdgeOrder
        + ColoredAdjacencyList
        + ColoredAdjacencyTest
        + Debug
        + GraphEdgeEditing,
>(
    graph: &G,
    timeout: Duration,
) -> (u32, ContractionSequence, u32) {
    let time_now = std::time::Instant::now();
    let mut full_tree = MonteCarloSearchTree::new(graph, true);

    loop {
        let mut tree = full_tree.new_game();
        tree.make_random_choice(MonteCarloSearchTreeGame::random_choice, &mut full_tree);

        full_tree.add_game(&tree);

        if time_now.elapsed() > timeout {
            break;
        }
    }

    full_tree.into_best_sequence()
}

/// Runs the search tree for the given period of time, will return the best score the contraction sequence and the number of games played
pub fn timeout_monte_carlo_search_tree_solver_preprocessed<
    G: Clone
        + AdjacencyList
        + GraphEdgeOrder
        + ColoredAdjacencyList
        + ColoredAdjacencyTest
        + Debug
        + GraphEdgeEditing,
>(
    graph: &G,
    timeout: Duration,
) -> (u32, ContractionSequence, u32) {
    let time_now = std::time::Instant::now();
    let mut full_tree = MonteCarloSearchTree::new(graph, false);

    loop {
        let mut tree = full_tree.new_game();
        tree.make_random_choice(MonteCarloSearchTreeGame::random_choice, &mut full_tree);

        full_tree.add_game(&tree);

        if time_now.elapsed() > timeout {
            break;
        }
    }

    full_tree.into_best_sequence()
}

/// Runs the search tree for the given period of time, after the initial timeout the search tree
/// will start to decrease the depth of the search try by subsequently choosing the best current move and playing games
/// which always start with this move. The two last parameters decide how long each collapsing period takes and how many
/// levels of the tree are collapsed in total (Total Time: ~ timeout+descend_time*max_descends)
pub fn timeout_monte_carlo_search_tree_solver_with_descend<
    G: Clone
        + AdjacencyList
        + GraphEdgeOrder
        + ColoredAdjacencyList
        + ColoredAdjacencyTest
        + Debug
        + GraphEdgeEditing,
>(
    graph: &G,
    timeout: Duration,
    descend_time: Duration,
    mut max_descends: u32,
) -> (u32, ContractionSequence, u32) {
    let mut time_now = std::time::Instant::now();
    let mut full_tree = MonteCarloSearchTree::new(graph, true).aborted_game_penalty(2);

    loop {
        let mut tree = full_tree.new_game();
        tree.make_random_choice(MonteCarloSearchTreeGame::random_choice, &mut full_tree);

        full_tree.add_game(&tree);

        if time_now.elapsed() > timeout {
            break;
        }
    }

    while max_descends > 0 {
        full_tree.permanently_collapse_one_move();
        time_now = std::time::Instant::now();
        loop {
            let mut tree = full_tree.new_game();
            tree.make_random_choice(MonteCarloSearchTreeGame::random_choice, &mut full_tree);

            full_tree.add_game(&tree);

            if time_now.elapsed() > descend_time {
                break;
            }
        }
        max_descends -= 1;
    }

    full_tree.into_best_sequence()
}

pub enum MonteCarloSearchTreeNode {
    // The leaf contains only the score and nothing else
    Leaf(u32),

    // Inner contains the choices made already together with the game tree
    // The second u32 is the average
    Inner {
        choices: std::rc::Rc<std::cell::RefCell<FxHashMap<u32, MonteCarloSearchTreeNode>>>,
        cumulative_score: u32,
        number_of_games: u32,
    },
}

pub struct MonteCarloSearchTree<G> {
    graph: G,
    // First one is the choice, second one the outcome
    games: std::rc::Rc<std::cell::RefCell<FxHashMap<u32, MonteCarloSearchTreeNode>>>,
    best_contraction_sequence: Option<ContractionSequence>,
    best_score: u32,
    num_games: u32,
    preprocessing_sequence: ContractionSequence,
    aborted_game_penalty: u32,

    collapse_sequence: ContractionSequence,
    collapse_score: u32,
    collapsed_graph: G,
}

impl<
        G: Clone
            + AdjacencyList
            + GraphEdgeOrder
            + ColoredAdjacencyList
            + ColoredAdjacencyTest
            + Debug
            + GraphEdgeEditing,
    > MonteCarloSearchTree<G>
{
    /// Creates a new MonteCarloSearchTree based on a implementation of a graph
    pub fn new(g: &G, with_preprocessing: bool) -> MonteCarloSearchTree<G> {
        let mut clone = g.clone();
        let mut preprocessing_sequence = ContractionSequence::new(clone.number_of_nodes());

        if with_preprocessing {
            prune_leaves(&mut clone, &mut preprocessing_sequence);
            prune_twins(&mut clone, &mut preprocessing_sequence);
        }

        let nodes = clone.number_of_nodes();

        MonteCarloSearchTree {
            graph: clone.clone(),
            games: std::rc::Rc::new(std::cell::RefCell::new(FxHashMap::default())),
            best_contraction_sequence: None,
            best_score: g.number_of_nodes(),
            num_games: 0,
            preprocessing_sequence,
            aborted_game_penalty: 0,

            collapse_sequence: ContractionSequence::new(nodes),
            collapse_score: 0,
            collapsed_graph: clone,
        }
    }

    /// Uses the best currently known first move to reduce the height of the search tree by 1
    /// This permanently adds the move to any subsequent games, therefore a warmup period is
    /// recommended
    pub fn permanently_collapse_one_move(&mut self) {
        let mut min_move_id = 0;
        let mut min_score = f64::MAX;

        let mut current = self.games.clone();

        for descend in self.collapse_sequence.merges() {
            let temp = match current
                .borrow()
                .get(&(descend.0 * self.graph.number_of_nodes() + descend.1))
                .unwrap()
            {
                MonteCarloSearchTreeNode::Inner {
                    choices,
                    cumulative_score: _,
                    number_of_games: _,
                } => choices.clone(),
                MonteCarloSearchTreeNode::Leaf(_) => {
                    return;
                }
            };
            current = temp;
        }

        for moves in current.borrow().iter() {
            match moves.1 {
                MonteCarloSearchTreeNode::Inner {
                    choices: _,
                    cumulative_score,
                    number_of_games,
                } => {
                    if min_score > (*cumulative_score as f64 / *number_of_games as f64) {
                        min_score = *cumulative_score as f64 / *number_of_games as f64;
                        min_move_id = *moves.0;
                    }
                }
                MonteCarloSearchTreeNode::Leaf(score) => {
                    if min_score > *score as f64 {
                        min_score = *score as f64;
                        min_move_id = *moves.0;
                    }
                }
            }
        }
        if min_move_id == 0 {
            return;
        }
        let first_node = min_move_id / self.graph.number_of_nodes();
        let second_node = min_move_id % self.graph.number_of_nodes();

        // Add the currently best move to the collapse sequence
        self.collapse_sequence
            .merge_node_into(first_node, second_node);

        // Update collapsed graph and collapsed score
        let mut new_graph = self.collapsed_graph.clone();
        new_graph.merge_node_into(first_node, second_node);

        self.collapse_score = self
            .collapse_score
            .max(new_graph.red_degrees().max().unwrap());
        self.collapsed_graph = new_graph;
    }

    /// This decides what penalty is added to the twin width of aborted games. It should increase
    /// the gap between promising and hopeless games to faster converge to an optimal solution
    pub fn aborted_game_penalty(mut self, penalty: u32) -> Self {
        self.aborted_game_penalty = penalty;
        self
    }

    /// Creates a new game based on the current graph. The current graph is the original graph after all preprocessing steps
    /// and with all collapsed moves already executed.
    pub fn new_game(&self) -> MonteCarloSearchTreeGame<G> {
        MonteCarloSearchTreeGame::new(self.collapsed_graph.clone())
    }

    /// Checks how many games have been played
    pub fn num_games(&self) -> u32 {
        self.num_games
    }

    /// Checks the current best score of any solution
    pub fn best_score(&self) -> u32 {
        self.best_score
    }

    /// Builds the best contraction sequence based on initial preprocessing sequence, the collapse sequence followed by the
    /// current best contraction sequence
    pub fn into_best_sequence(mut self) -> (u32, ContractionSequence, u32) {
        self.preprocessing_sequence
            .append(&self.best_contraction_sequence.unwrap());
        (self.best_score, self.preprocessing_sequence, self.num_games)
    }

    /// Adds a played out game to update the heuristics of the MonteCarlo search tree
    /// Also updates the best score
    pub fn add_game(&mut self, game: &MonteCarloSearchTreeGame<G>) {
        self.num_games += 1;
        let mut current_ptr = self.games.clone();

        let mut graph = self.graph.clone();
        let twin_width = game
            .get_final_twin_width()
            .unwrap_or(self.best_score + self.aborted_game_penalty)
            + self.collapse_score;

        for x in game.contraction_sequence.merges() {
            let mut next_choice = None;

            match current_ptr
                .borrow_mut()
                .entry(x.0 * self.graph.number_of_nodes() + x.1)
            {
                Entry::Occupied(mut value) => {
                    if let MonteCarloSearchTreeNode::Inner {
                        choices,
                        cumulative_score,
                        number_of_games,
                    } = value.get_mut()
                    {
                        graph.merge_node_into(x.0, x.1);

                        *cumulative_score += twin_width;
                        *number_of_games += 1;
                        next_choice = Some(choices.clone());
                    }
                }
                Entry::Vacant(value) => {
                    if graph.len() == 1 {
                        value.insert(MonteCarloSearchTreeNode::Leaf(twin_width));
                    } else {
                        graph.merge_node_into(x.0, x.1);

                        if let MonteCarloSearchTreeNode::Inner {
                            choices,
                            cumulative_score: _,
                            number_of_games: _,
                        } = value.insert(MonteCarloSearchTreeNode::Inner {
                            choices: std::rc::Rc::new(
                                std::cell::RefCell::new(FxHashMap::default()),
                            ),
                            cumulative_score: twin_width,
                            number_of_games: 1,
                        }) {
                            next_choice = Some(choices.clone());
                        }
                    }
                }
            }
            current_ptr = next_choice.unwrap();
        }
        if self.best_contraction_sequence.is_none() || twin_width < self.best_score {
            self.best_contraction_sequence = Some(game.contraction_sequence.clone());
            self.best_score = twin_width;
        }
    }
}

pub struct MonteCarloSearchTreeGame<G> {
    // The pointer to the graph to permutate
    graph: G,

    // Current contraction sequence which have been used to reach this tree
    contraction_sequence: ContractionSequence,

    // Either the final twin width or None in case that the game was aborted due to a low score
    final_twin_width: Option<u32>,
}

impl<
        G: Clone
            + AdjacencyList
            + GraphEdgeOrder
            + ColoredAdjacencyList
            + ColoredAdjacencyTest
            + GraphEdgeEditing
            + Debug,
    > MonteCarloSearchTreeGame<G>
{
    /// Creates a new game from a given graph
    fn new(graph: G) -> MonteCarloSearchTreeGame<G> {
        let seq = ContractionSequence::new(graph.number_of_nodes());
        MonteCarloSearchTreeGame {
            graph,
            final_twin_width: None,
            contraction_sequence: seq,
        }
    }

    /// Return the final twin width if there is any. If the game has been aborted this will return None
    pub fn get_final_twin_width(&self) -> &Option<u32> {
        &self.final_twin_width
    }

    /// The basic function to make a random allowed next move, includes basic optimizations like
    /// only choosing neighbors in a two neighborhood and only considering those neighbors which do have similar degree.
    pub fn random_choice(
        remaining_nodes: &BitSet,
        graph: &mut G,
        max_allowed_red_edges: u32,
    ) -> (u32, u32) {
        let chosen_node =
            (rand::thread_rng().gen::<f64>() * (remaining_nodes.cardinality() - 1) as f64) as usize;

        // Get a random node
        if let Some(first_node) = remaining_nodes.iter().nth(chosen_node) {
            let random_choice_or_neighboorhood = rand::thread_rng().gen::<f64>();

            let mut best_red_edges = std::u32::MAX;

            // TODO: Make this a bitset it may be faster!
            let mut best_partners = Vec::new();

            if random_choice_or_neighboorhood < 0.99 {
                let neighbors = graph.neighbors_of(first_node);

                let mut set = FxHashSet::default();

                // Find the best contraction partner
                for neighbor in neighbors.iter() {
                    if graph
                        .neighbors_of(*neighbor)
                        .len()
                        .abs_diff(neighbors.len())
                        <= max_allowed_red_edges as usize
                    {
                        set.insert(*neighbor);
                    }
                    for neighbors_of_neighbors in graph.neighbors_of(*neighbor).iter() {
                        if graph
                            .neighbors_of(*neighbor)
                            .len()
                            .abs_diff(neighbors.len())
                            <= max_allowed_red_edges as usize
                        {
                            set.insert(*neighbors_of_neighbors);
                        }
                    }
                }
                // Remove ourself since the first node is not available for contraction
                set.remove(&first_node);

                // If we have neighbors try merging with them
                if !set.is_empty() {
                    for partner in set.iter() {
                        let dry_run_merge = graph.red_degree_after_merge(*partner, first_node);
                        match dry_run_merge.cmp(&best_red_edges) {
                            std::cmp::Ordering::Equal => {
                                best_partners.push(*partner);
                            }
                            std::cmp::Ordering::Less => {
                                best_partners.clear();
                                best_partners.push(*partner);
                                best_red_edges = dry_run_merge;
                            }
                            _ => {}
                        }
                    }
                }
                // If we do not have neighbors merge with the best node of all nodes.
                else {
                    for partner in remaining_nodes.iter() {
                        if partner == first_node {
                            continue;
                        }
                        let dry_run_merge = graph.red_degree_after_merge(partner, first_node);

                        match dry_run_merge.cmp(&best_red_edges) {
                            std::cmp::Ordering::Equal => {
                                best_partners.push(partner);
                            }
                            std::cmp::Ordering::Less => {
                                best_partners.clear();
                                best_partners.push(partner);
                                best_red_edges = dry_run_merge;
                            }
                            _ => {}
                        }
                    }
                }
            } else {
                // Chance told us to merge with any node to find possibilities where the nodes are not in the neighborhood of each other
                for partner in remaining_nodes.iter() {
                    if partner == first_node {
                        continue;
                    }
                    let dry_run_merge = graph.red_degree_after_merge(partner, first_node);
                    match dry_run_merge.cmp(&best_red_edges) {
                        std::cmp::Ordering::Equal => {
                            best_partners.push(partner);
                        }
                        std::cmp::Ordering::Less => {
                            best_partners.clear();
                            best_partners.push(partner);
                            best_red_edges = dry_run_merge;
                        }
                        _ => {}
                    }
                }
            }

            let random_choice_best_partner =
                (rand::thread_rng().gen::<f64>() * (best_partners.len() - 1) as f64) as usize;

            (
                first_node,
                *best_partners.get(random_choice_best_partner).unwrap(),
            )
        } else {
            panic!("This should never happen!");
        }
    }

    /// Executes the main loop with a function which decides what nodes to contract next
    pub fn make_random_choice<F: FnMut(&BitSet, &mut G, u32) -> (u32, u32)>(
        &mut self,
        mut decision_function: F,
        full_game_tree: &mut MonteCarloSearchTree<G>,
    ) {
        if self.graph.number_of_edges() <= 1 {
            if let Some(first_edge) = self.graph.edges(true).next() {
                self.contraction_sequence
                    .merge_node_into(first_edge.0, first_edge.1);
            }
            self.final_twin_width = Some(0);
            return;
        }

        let mut twin_width = 0;
        let mut remaining_nodes = self.contraction_sequence.remaining_nodes().unwrap();

        for x in remaining_nodes.clone().iter() {
            if self.graph.degree_of(x) < 1 {
                remaining_nodes.unset_bit(x);
            }
        }

        loop {
            // All nodes merged? Finish...
            if remaining_nodes.cardinality() <= 1 {
                break;
            }

            let choice = decision_function(
                &remaining_nodes,
                &mut self.graph,
                full_game_tree.best_score() - twin_width,
            );
            self.graph.merge_node_into(choice.1, choice.0);

            let max_red_deg = self.graph.red_degrees().max().unwrap();
            twin_width = twin_width.max(max_red_deg);

            self.contraction_sequence
                .merge_node_into(choice.1, choice.0);
            remaining_nodes.unset_bit(choice.1);

            // Abort games which are not better than previous games
            if twin_width >= full_game_tree.best_score() {
                //+2 is an penalty term to prohibit the usage of this branch by decreasing the probability with simulated annealing
                return;
            }
        }

        self.final_twin_width = Some(twin_width);
    }
}

#[cfg(test)]
pub mod tests {
    use std::{fs::File, io::BufReader};

    use crate::{
        graph::{AdjArray, EdgeColor, GraphEdgeEditing, GraphNew},
        heuristic::monte_carlo_search_tree::{
            timeout_monte_carlo_search_tree_solver, MonteCarloSearchTree,
        },
        io::PaceReader,
    };

    use super::MonteCarloSearchTreeGame;

    // Check instances/exact-public/exact_034.gr
    #[test]
    fn tiny() {
        let mut cumulative_score = 0;
        for (i, _) in [1, 2, 0, 0, 3, 0, 2, 4, 1, 2].into_iter().enumerate() {
            //for (i, tww) in [1].into_iter().enumerate() {
            if i == 4 {
                continue; // too slow
            }

            let filename = format!("instances/tiny/tiny{:>03}.gr", i + 1);
            let reader = File::open(filename.clone())
                .unwrap_or_else(|_| panic!("Cannot open file {}", &filename));
            let buf_reader = BufReader::new(reader);

            let pace_reader =
                PaceReader::try_new(buf_reader).expect("Could not construct PaceReader");

            let mut graph = AdjArray::new(pace_reader.number_of_nodes());
            graph.add_edges(pace_reader, EdgeColor::Black);

            let mut full_tree = MonteCarloSearchTree::new(&graph, true);

            for _ in 0..1000 {
                let mut tree = full_tree.new_game();
                tree.make_random_choice(MonteCarloSearchTreeGame::random_choice, &mut full_tree);

                full_tree.add_game(&tree);
            }
            cumulative_score += full_tree.best_score;
            println!("Best score {}", full_tree.best_score);
        }
        println!("Cumulative score {cumulative_score}");
    }

    #[test]
    fn tiny_timeout() {
        let mut cumulative_score = 0;
        for (i, _) in [1, 2, 0, 0, 3, 0, 2, 4, 1, 2].into_iter().enumerate() {
            //for (i, tww) in [1].into_iter().enumerate() {
            if i == 4 {
                continue; // too slow
            }

            let filename = format!("instances/tiny/tiny{:>03}.gr", i + 1);
            let reader = File::open(filename.clone())
                .unwrap_or_else(|_| panic!("Cannot open file {}", &filename));
            let buf_reader = BufReader::new(reader);

            let pace_reader =
                PaceReader::try_new(buf_reader).expect("Could not construct PaceReader");

            let mut graph = AdjArray::new(pace_reader.number_of_nodes());
            graph.add_edges(pace_reader, EdgeColor::Black);

            let solve =
                timeout_monte_carlo_search_tree_solver(&graph, std::time::Duration::from_secs(1));
            cumulative_score += solve.0;
            println!("Best score {}", solve.0);
        }
        println!("Cumulative score {cumulative_score}");
    }

    //Benchmark the following graphs since they take ages!
    // instances/exact-public/exact_146.gr
    // instances/exact-public/exact_148.gr
    // instances/exact-public/exact_198.gr
    // instances/exact-public/exact_142.gr
    // instances/exact-public/exact_176.gr
    // instances/exact-public/exact_174.gr
    // instances/exact-public/exact_134.gr
}
