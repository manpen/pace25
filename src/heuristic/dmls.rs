use std::time::Instant;

use log::info;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

use crate::{
    graph::*,
    prelude::{IterativeAlgorithm, TerminatingIterativeAlgorithm},
    utils::DominatingSet,
};

/// # GreedyReverseSearch
///
/// An iterative algorithm that replaces at least one node in the current DomSet by another. If
/// possible, replace two or more nodes by one.
///
/// The algorithm stores for each node u in the dominating set how many nodes they uniquely cover,
/// i.e. how many nodes are *only* covered by u. If this number is 0, the node is considered
/// redundant and can be safely removed from the DomSet.
///
/// Using an IntersectionForest, it efficiently stores for each dominating node u the set of all
/// nodes v not in the DomSet that are also incident to *every* uniquely covered neighbor of u.
/// Thus, v can replace u in the DomSet.
///
/// For each node v not in the DomSet, we assign a score to v which is equivalent to the number of
/// dominating nodes u that v can replace in that way. In each iteration, we then sample such a
/// replacement node v in proportion to its score. Afterwards, we add v to the DomSet, remove the
/// redundant dominating node u from the DomSet and possibly further nodes in the DomSet that are
/// now redundant (i.e. `score[v] > 1`).
///
/// See descriptions of variables for more information.
/// We have multiple invariants throughout the algorithm that we need to maintain:
/// (I1) Adjacency-Lists are partitioned by appearance in the current DomSet: nodes in the DomSet appear first, then nodes not in the DomSet
/// (I2) Only nodes `u` with `num_covered[u] = 1` are inserted into the IntersectionForest
/// (I3) `scores[u]` is equivalent to the number of IntersectionTrees where u is stored in the root (as an entry, not the node itself): excluded are dominating nodes
/// (I4) if `scores[u] > 0`, u is inserted into the Sampler
///
///
/// As this is a step-based algorithm, different members of this struct have certain BaseStates in
/// which they must be at the beginning/end of each *round*. See BaseStateError down below for a
/// description of these states.
pub struct DMLS<
    G,
    const NUM_SAMPLES: usize = 10,
> where
    G: StaticGraph + SelfLoop,
{
    /// A reference to the graph: mutable access is needed as we need to re-order adjacency lists
    graph: G,

    /// The current solution we operate on
    current_solution: DominatingSet,

    /// The currently best known solution of the algorithm
    best_solution: DominatingSet,

    /// RNG used for sampling
    rng: Pcg64Mcg,

    eligible_nodes: Vec<(u64,Node)>,
    node_score: Vec<NumNodes>,


    /// Number of incident dominating nodes
    num_covered: Vec<NumNodes>,

    /// Number of nodes this (dominating) nodes covers uniquely (no other dominating node covers) (V9)
    uniquely_covered: Vec<NumNodes>,

    /// Nodes u that can possibly be removed from the DomSet as uniquely_covered[u] = 0
    redundant_nodes: Vec<Node>,

    /// Last time a node was added/removed from the DomSet
    age: Vec<u64>,

    /// Current iteration
    round: u64,

    /// Keep track of all applied modifications to current_solution to also apply them to
    /// best_solution when new best solution is found
    domset_modifications: Vec<DomSetModification>,

    /// BitSet indicating all nodes that will never appear in an optimal solution and can be
    /// disregarded
    non_optimal_nodes: BitSet,

    num_uncovered: NumNodes,

    /// BitSet indicating whether a node is permanently covered by a (removed) fixed node
    is_perm_covered: BitSet,

    verbose_logging: bool,
    previous_improvement: u64,
    start_time: Instant,
}

impl<G, const NUM_SAMPLES: usize>
    DMLS<G, NUM_SAMPLES>
where
    G: StaticGraph + SelfLoop,
{
    /// Creates a new instance of the algorithm for a given graph and a starting DomSet which must be valid.
    /// Runs Subset-Reduction beforehand to further reduce the DomSet and removes redundant nodes afterwards.
    pub fn new(
        mut graph: G,
        mut initial_solution: DominatingSet,
        is_perm_covered: BitSet,
        non_optimal_nodes: BitSet,
        seeding_rng: &mut impl Rng,
    ) -> Self {
        assert!(initial_solution.is_valid_given_previous_cover(&graph, &is_perm_covered));
        assert!(graph.len() > 0);

        let n = graph.len();

        // Initialize NumCovered with 2 for permanently covered nodes to prevent unique-checks
        let mut num_covered: Vec<NumNodes> = (0..graph.number_of_nodes())
            .map(|i| is_perm_covered.get_bit(i) as NumNodes * 2)
            .collect();
        let mut age = vec![0; n];

        // (I1) Reorder adjacency lists such that dominating nodes appear first
        for u in initial_solution.iter() {
            age[u as usize] = 1;
            for i in 0..graph.degree_of(u) {
                let v = graph.ith_neighbor(u, i);
                graph.swap_neighbors(
                    v,
                    graph.ith_cross_position(u, i),
                    num_covered[v as usize] - (is_perm_covered.get_bit(v) as NumNodes * 2),
                );
                num_covered[v as usize] += 1;
            }
        }

        let mut uniquely_covered: Vec<NumNodes> = vec![0; graph.len()];
        graph.vertices().for_each(|u| {
            if num_covered[u as usize] == 1 {
                uniquely_covered[graph.ith_neighbor(u, 0) as usize] += 1;
            }
        });

        // Remove redundant nodes from the DomSet and update datastructures
        for i in (0..initial_solution.len()).rev() {
            let u = initial_solution.ith_node(i);
            if uniquely_covered[u as usize] != 0 {
                continue;
            }

            // Possibly breaks (I1)
            initial_solution.remove_node(u);
            age[u as usize] = 0;

            // Restores (I1)
            for j in (0..graph.degree_of(u)).rev() {
                let v = graph.ith_neighbor(u, j);
                num_covered[v as usize] -= 1;
                graph.swap_neighbors(
                    v,
                    graph.ith_cross_position(u, j),
                    num_covered[v as usize] - (is_perm_covered.get_bit(v) as NumNodes * 2),
                );

                if num_covered[v as usize] == 1 {
                    uniquely_covered[graph.ith_neighbor(v, 0) as usize] += 1;
                }
            }
        }

        let current_solution = initial_solution.clone();
        let best_solution = initial_solution;

        let rng = Pcg64Mcg::seed_from_u64(seeding_rng.r#gen());
        Self {
            graph,
            current_solution,
            best_solution,
            rng,

            eligible_nodes: Vec::with_capacity(n),
            node_score: vec![0;n],
            num_covered,
            uniquely_covered,
            redundant_nodes: Vec::with_capacity(n),
            age,
            round: 1,
            domset_modifications: Vec::with_capacity(n>>5),
            is_perm_covered,
            verbose_logging: false,
            previous_improvement: 1,
            start_time: Instant::now(),
            non_optimal_nodes,
            num_uncovered: 0,
        }
    }

    pub fn enable_verbose_logging(&mut self) {
        self.verbose_logging = true;
    }

    pub fn step(&mut self) {
        self.update_best_solution();

        if self.round % 10_000 == 0 {
            info!(
                " Best solution: size={:6}, current solution: {:6}, uncovered: {:6}, round={:9}, gap={:9}, time={:7}ms",
                self.best_solution.len(),
                self.current_solution.len(),
                self.num_uncovered,
                self.round,
                self.round - self.previous_improvement,
                self.start_time.elapsed().as_millis()
            );
        }
        self.round+=1;

        // Solution is feasible
        if self.num_uncovered == 0 {
            let mut min_age:u64 = u64::MAX;
            let mut max_score:NumNodes = NumNodes::MAX;
            let mut node:Node = u32::MAX;
            for x in self.current_solution.iter() {
                let curr_score = self.uniquely_covered[x as usize];
                if curr_score < max_score || (curr_score == max_score && self.age[x as usize] < min_age) {
                    min_age = self.age[x as usize];
                    max_score = curr_score;
                    node = x;
                }
            }
            self.remove_node_from_solution(node);
        }

        let random = self.current_solution.sample_non_fixed(&mut self.rng);
        self.remove_node_from_solution(random);

        if self.rng.r#gen::<f32>() < 0.6 {
            let random_best = self.random_minimum_loss_node::<NUM_SAMPLES>();
            self.remove_node_from_solution(random_best);
        }

        self.greedy_add_node_to_domset();
        if self.best_solution.len() > self.current_solution.len()+1 {
            self.greedy_add_node_to_domset();
        }
    }

    fn greedy_add_node_to_domset(&mut self) {
        if self.num_uncovered == 0 {return;}
        let mut min_age:u64 = u64::MAX;
        let mut max_score:NumNodes = 0;
        let mut node:Node = u32::MAX;

        self.eligible_nodes.retain_mut(|(age,nd)| {
            let curr_score = self.node_score[*nd as usize];
            if curr_score > max_score || (curr_score == max_score && *age < min_age) {
                min_age = *age;
                max_score = curr_score;
                node = *nd;
            }
            curr_score > 0
        });
        if self.current_solution.is_in_domset(node) {
            panic!("Something went wrong here (add) {} and score {}!", node, max_score);
        }

        self.current_solution.add_node(node);
        let _ = self
            .domset_modifications
            .push_within_capacity(DomSetModification::Add(node));

        self.age[node as usize] = self.round;


        for i in 0..self.graph.degree_of(node) {
            let neighbor = self.graph.ith_neighbor(node, i);
            self.graph.swap_neighbors(
                neighbor,
                self.graph.ith_cross_position(node, i),
                self.num_covered[neighbor as usize]
                    - (self.is_perm_covered.get_bit(neighbor) as NumNodes * 2),
            );
            self.num_covered[neighbor as usize] += 1;

            // Was uncovered before
            if self.num_covered[neighbor as usize] == 1 {
                self.uniquely_covered[node as usize] += 1;
                self.num_uncovered -= 1;
                for nb in self.graph.neighbors_of(neighbor).filter(|x| !self.non_optimal_nodes.get_bit(*x)) {
                    self.node_score[nb as usize] -= 1;
                }
            }
            // Not uniquely covering anymore
            else if self.num_covered[neighbor as usize] == 2 {
                let former_unique_covering_node = self.graph.ith_neighbor(neighbor, 0);
                self.uniquely_covered[former_unique_covering_node as usize] -= 1;

                if self.uniquely_covered[former_unique_covering_node as usize] == 0 {
                    self.redundant_nodes.push(former_unique_covering_node);
                }
            }
        }

        self.remove_redundant_nodes();
    }

    fn remove_redundant_nodes(&mut self) {
        self.redundant_nodes.sort_by_key(|x| self.age[*x as usize]);
        for x in 0..self.redundant_nodes.len() {
            if self.uniquely_covered[x] == 0 {
                self.remove_node_from_solution(self.redundant_nodes[x]);
            }
        }
        self.redundant_nodes.clear();
    }

    fn remove_node_from_solution(&mut self, removed_node: Node) {
        if !self.current_solution.is_in_domset(removed_node) {
            panic!("Something went wrong here!");
        }
        self.current_solution.remove_node(removed_node);
        let _ = self
            .domset_modifications
            .push_within_capacity(DomSetModification::Remove(removed_node));

        self.age[removed_node as usize] = self.round;


        // (I1) Re-order neighbors
        for i in (0..self.graph.degree_of(removed_node)).rev() {
            let neighbor = self.graph.ith_neighbor(removed_node, i);
            self.num_covered[neighbor as usize] -= 1;
            self.graph.swap_neighbors(
                neighbor,
                self.graph.ith_cross_position(removed_node, i),
                self.num_covered[neighbor as usize]
                    - (self.is_perm_covered.get_bit(neighbor) as NumNodes * 2),
            );
            // New uncovered node
            if self.num_covered[neighbor as usize] == 0 {
                self.num_uncovered += 1;
                for nb_2 in self.graph.neighbors_of(neighbor).filter(|x| !self.non_optimal_nodes.get_bit(*x)) {
                    self.node_score[nb_2 as usize] += 1;
                    if self.node_score[nb_2 as usize] == 1 {
                        self.eligible_nodes.push((self.age[nb_2 as usize], nb_2));
                    }
                }
            }
            // Now uniquely covering
            else if self.num_covered[neighbor as usize] == 1 {
                let former_unique_covering_node = self.graph.ith_neighbor(neighbor, 0);
                self.uniquely_covered[former_unique_covering_node as usize] += 1;
            }
        }
    }

    /// Updates the best_solution to current_solution if better and valid
    fn update_best_solution(&mut self) {
        if self.num_uncovered > 0 || self.current_solution.len() >= self.best_solution.len() {
            return;
        }

        if self.domset_modifications.len() > self.graph.number_of_nodes() as usize / 64 {
            self.best_solution = self.current_solution.clone();
            self.domset_modifications.clear();
        } else {
            for modification in self.domset_modifications.drain(..) {
                match modification {
                    DomSetModification::Add(node) => self.best_solution.add_node(node),
                    DomSetModification::Remove(node) => self.best_solution.remove_node(node),
                }
            }
        }
    }

    pub fn current_round(&self) -> u64 {
        self.round
    }

    pub fn current_score(&self) -> NumNodes {
        self.best_solution.len() as NumNodes
    }

    /// Samples a constant number of times from the set of non-fixed nodes in self.current_solution
    /// and returns the node with minimum UniqueCov and age
    #[inline(always)]
    fn random_minimum_loss_node<const SAMPLE_TIMES: usize>(&mut self) -> Node {
        self.current_solution
            .sample_many_non_fixed::<_, SAMPLE_TIMES>(&mut self.rng)
            .map(|u| (self.uniquely_covered[u as usize], self.age[u as usize], u))
            .min()
            .unwrap()
            .2
    }
}

/// Helper enum to keep track of DomSet-Changes
enum DomSetModification {
    Add(Node),
    Remove(Node),
}

impl<G, const NUM_SAMPLES: usize>
    IterativeAlgorithm<DominatingSet> for DMLS<G, NUM_SAMPLES>
where
    G: StaticGraph + SelfLoop,
{
    fn execute_step(&mut self) {
        self.step();
    }

    fn is_completed(&self) -> bool {
        false
    }

    fn best_known_solution(&mut self) -> Option<DominatingSet> {
        Some(self.best_solution.clone())
    }
}

impl<G, const NUM_SAMPLES: usize>
    TerminatingIterativeAlgorithm<DominatingSet>
    for DMLS<G, NUM_SAMPLES>
where
    G: StaticGraph + SelfLoop,
{
}
