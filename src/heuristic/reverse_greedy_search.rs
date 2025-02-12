use std::fmt::Debug;

use rand::Rng;
use thiserror::Error;

use crate::{
    errors::InvariantCheck,
    graph::*,
    kernelization::{KernelizationRule, SubsetRule},
    prelude::{IterativeAlgorithm, TerminatingIterativeAlgorithm},
    utils::{
        intersection_forest::{IntersectionForest, IntersectionForestError},
        sampler::{SamplerError, WeightedPow2Sampler},
        DominatingSet,
    },
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
pub struct GreedyReverseSearch<
    'a,
    R,
    G,
    const NUM_SAMPLER_BUCKETS: usize = 8,
    const NUM_SAMPLES: usize = 10,
> where
    R: Rng,
    G: StaticGraph + SelfLoop,
{
    /// A reference to the graph: mutable access is needed as we need to re-order adjacency lists
    graph: &'a mut G,

    /// The current solution we operate on
    current_solution: DominatingSet,
    /// The currently best known solution of the algorithm
    best_solution: DominatingSet,
    /// Is the algorithm stuck in a (local) optimum?
    ///
    /// Will be true if the sampler is empty, i.e. the algorithm can no longer improve the solution.
    is_locally_optimal: bool,

    /// A sampler for sampling nodes with weights that are powers of 2.
    ///
    /// Contains only nodes u with scores[u] > 0.
    sampler: WeightedPow2Sampler<NUM_SAMPLER_BUCKETS>,
    /// RNG used for sampling
    rng: &'a mut R,

    /// List of all nodes that are either currently inserted into an IntersectionTree and need to
    /// be removed to maintain (I2) or need to be added to an IntersectionTree to maintain (I2)
    nodes_to_update: Vec<Node>,
    /// Helper BitSet to easily identify if a node is pushed to `nodes_to_update`
    in_nodes_to_update: BitSet,

    /// Number of incident dominating nodes
    ///
    /// (I1) `Neighbors[u][..num_covered\[u\]]` is a subset of the current DomSet
    num_covered: Vec<NumNodes>,
    /// Number of nodes this (dominating) nodes covers uniquely (no other dominating node covers)
    uniquely_covered: Vec<NumNodes>,

    /// Nodes u that can possibly be removed from the DomSet as uniquely_covered[u] = 0
    redundant_nodes: Vec<Node>,
    /// Number of appearances in entries of root nodes in the IntersectionForest
    scores: Vec<NumNodes>,

    /// Last time a node was added/removed from the DomSet
    age: Vec<u64>,
    /// Current iteration
    round: u64,

    /// IntersectionForest
    ///
    /// Every node u in the DomSet is assigned an IntersectionTree. Nodes that are uniquely covered by this
    /// u are then inserted into the IntersectionTree of `u`. `IntersectionTree[u]` thus stores all nodes in its root
    /// that are incident to *all* uniquely covered nodes of u and can thus replace u in the DomSet.
    ///
    /// v in root of `IntersectionTree[u]` ==> `scores[v] > 0` ==> `v` in sampler ==> `v` can be sampled to replace `u`
    ///
    /// Note that we only *really* consider neighbors that are not subset-dominated and thus can appear in any
    /// optimal DomSet without the possibility of directly replacing them.
    pub intersection_forest: IntersectionForest,

    /// Keep track of all applied modifications to current_solution to also apply them to
    /// best_solution when new best solution is found
    domset_modifications: Vec<DomSetModification>,
}

impl<'a, R, G, const NUM_SAMPLER_BUCKETS: usize, const NUM_SAMPLES: usize>
    GreedyReverseSearch<'a, R, G, NUM_SAMPLER_BUCKETS, NUM_SAMPLES>
where
    R: Rng,
    G: StaticGraph + SelfLoop,
{
    /// Creates a new instance of the algorithm for a given graph and a starting DomSet which must be valid.
    /// Runs Subset-Reduction beforehand to further reduce the DomSet and removes redundant nodes afterwards.
    pub fn new(graph: &'a mut G, mut initial_solution: DominatingSet, rng: &'a mut R) -> Self {
        assert!(initial_solution.is_valid(graph));

        // If only fixed nodes cover the graph, this is optimal.
        // For API-purposes, we create an *empty* instance that holds the optimal solution
        if initial_solution.all_fixed() {
            return Self {
                graph,
                current_solution: initial_solution.clone(),
                best_solution: initial_solution,
                is_locally_optimal: true,
                sampler: WeightedPow2Sampler::new(0),
                rng,
                nodes_to_update: Vec::new(),
                in_nodes_to_update: BitSet::new(1),
                num_covered: Vec::new(),
                uniquely_covered: Vec::new(),
                redundant_nodes: Vec::new(),
                scores: Vec::new(),
                age: Vec::new(),
                intersection_forest: IntersectionForest::default(),
                round: 1,
                domset_modifications: Vec::new(),
            };
        }

        let n = graph.number_of_nodes() as usize;

        // Run Subset-Reduction and create reduced edge set
        let mut csr_repr = graph.extract_csr_repr();
        let non_optimal_nodes = SubsetRule::apply_rule(&mut csr_repr, &mut initial_solution);

        let mut num_covered = vec![0; n];
        let mut age = vec![0; n];

        // Reorder adjacency lists such that dominating nodes appear first
        for u in initial_solution.iter() {
            age[u as usize] = 1;
            for i in 0..graph.degree_of(u) {
                let v = graph.ith_neighbor(u, i);
                graph.swap_neighbors(v, graph.ith_cross_position(u, i), num_covered[v as usize]);
                num_covered[v as usize] += 1;
            }
        }

        // Count number of uniquely covered neighbors
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

            initial_solution.remove_node(u);
            age[u as usize] = 0;

            for j in (0..graph.degree_of(u)).rev() {
                let v = graph.ith_neighbor(u, j);
                num_covered[v as usize] -= 1;
                graph.swap_neighbors(v, graph.ith_cross_position(u, j), num_covered[v as usize]);

                if num_covered[v as usize] == 1 {
                    uniquely_covered[graph.ith_neighbor(v, 0) as usize] += 1;
                }
            }
        }

        // Instantiate sampler and IntersectionForest with reduced neighbor-set
        let mut sampler = WeightedPow2Sampler::new(n);
        let mut scores = vec![0; n];

        // Fixed nodes will never appear in the IntersectionForest as they are fixed in the DomSet,
        // ie. no Tree-Owners, and will thus be never uniquely covered by another DomSet node nor
        // will they be a neighbor to a uniquely covered node (of another DomSet node).
        let fixed_nodes =
            BitSet::new_with_bits_set(graph.number_of_nodes(), initial_solution.iter_fixed());
        let mut intersection_forest =
            IntersectionForest::new_unsorted(csr_repr, fixed_nodes, non_optimal_nodes);

        // Insert uniquely covered neighbors of dominating nodes into IntersectionTrees & Sampler
        for u in initial_solution.iter_non_fixed() {
            for v in graph.neighbors_of(u) {
                debug_assert!(num_covered[v as usize] > 0);
                if num_covered[v as usize] == 1 {
                    intersection_forest.add_entry(u, v);
                }
            }

            for &v in intersection_forest.get_root_nodes(u) {
                if u != v {
                    scores[v as usize] += 1;
                    sampler.set_bucket(v, scores[v as usize] as usize);
                }
            }
        }

        let current_solution = initial_solution.clone();
        let best_solution = initial_solution;

        Self {
            graph,
            current_solution,
            best_solution,
            is_locally_optimal: false,
            sampler,
            rng,
            nodes_to_update: Vec::new(),
            in_nodes_to_update: BitSet::new(n as NumNodes),
            num_covered,
            uniquely_covered,
            redundant_nodes: Vec::new(),
            scores,
            age,
            intersection_forest,
            round: 1,
            domset_modifications: Vec::with_capacity(1 + n / 64),
        }
    }

    /// Run one iteration of the algorithm:
    ///
    /// 1. Sample a node from sampler
    /// 2. Insert the node into the DomSet
    /// 3. Remove all now redundant nodes of the DomSet
    /// 4. Update IntersectionTrees/Scores/Sampler accordingly
    pub fn step(&mut self) {
        #[cfg(debug_assertions)]
        self.is_correct().unwrap();

        // Sample node: if no node can be sampled, current solution is optimal
        let proposed_node = if let Some(node) = self.draw_node() {
            node
        } else {
            self.is_locally_optimal = true;
            return;
        };

        self.round += 1;

        debug_assert!(!self.current_solution.is_in_domset(proposed_node));
        debug_assert!(self.scores[proposed_node as usize] > 0);

        // Add node to DomSet
        self.add_node_to_domset(proposed_node);

        // Prefer nodes that have been unchanged for longer
        self.redundant_nodes.sort_by_key(|u| self.age[*u as usize]);

        // Remove redundant nodes from DomSet
        if !self.redundant_nodes.is_empty() {
            self.remove_redundant_node::<true>(self.redundant_nodes[0], proposed_node);
            for i in 1..self.redundant_nodes.len() {
                self.remove_redundant_node::<false>(self.redundant_nodes[i], proposed_node);
            }
            self.redundant_nodes.clear();
        }

        debug_assert!(self.uniquely_covered[proposed_node as usize] > 0);

        // Update IntersectionForest/Sampler for all remaining nodes_to_update
        self.update_forest_and_sampler();

        // Update the best known solution if needed
        self.update_best_solution();
    }

    /// Sample a node from the sampler
    ///
    /// Returns *None* if the sampler is empty, ie there is no way to replace any node in the
    /// current DomSet.
    fn draw_node(&mut self) -> Option<Node> {
        if self.sampler.is_empty() {
            return None;
        }

        let mut sample_node = None;
        let mut sample_bucket = 0;
        let mut sample_age = 0;

        self.sampler
            .sample_many::<_, NUM_SAMPLES>(&mut self.rng, |bucket, node| {
                if sample_bucket == bucket && sample_age < self.age[node as usize] {
                    return;
                }

                sample_node = Some(node);
                sample_bucket = bucket;
                sample_age = self.age[node as usize];
            });

        sample_node
    }

    /// Adds a node to the DomSet that was not part of it before.
    /// The node must be able to directly replace at least one node in the DomSet, ie. must be part
    /// of Sampler (I4).
    /// Updates the corresponding datastructures to maintain invariants.
    fn add_node_to_domset(&mut self, u: Node) {
        debug_assert!(!self.current_solution.is_in_domset(u));

        self.current_solution.add_node(u);
        // (I3) dominating nodes have no score
        self.scores[u as usize] = 0;
        self.age[u as usize] = self.round;
        // (I4) Node must be part of Sampler
        self.sampler.remove_entry(u);

        let _ = self
            .domset_modifications
            .push_within_capacity(DomSetModification::Add(u));

        // (I1)Update adjacency lists as well as (I2) num_covered/uniquely_covered
        //
        // If a previously uniquely covered node is now not longer uniquely covered,
        // add it to nodes_to_update as we must later update its IntersectionTree-Appearance
        for i in 0..self.graph.degree_of(u) {
            let neighbor = self.graph.ith_neighbor(u, i);
            self.graph.swap_neighbors(
                neighbor,
                self.graph.ith_cross_position(u, i),
                self.num_covered[neighbor as usize],
            );
            self.num_covered[neighbor as usize] += 1;

            if self.num_covered[neighbor as usize] == 2 {
                // (I1) the first neighbor must be a dominating node
                let former_unique_covering_node = self.graph.ith_neighbor(neighbor, 0);
                self.uniquely_covered[former_unique_covering_node as usize] -= 1;
                if !self.in_nodes_to_update.set_bit(neighbor) {
                    self.nodes_to_update.push(neighbor);
                }

                if self.uniquely_covered[former_unique_covering_node as usize] == 0 {
                    self.redundant_nodes.push(former_unique_covering_node);
                }
            }
        }
    }

    /// Removes a redundant node `old_node` from the DomSet after inserting `new_node` into it.
    ///
    /// if MARKER is *true*, this is the first redundant node that gets removed and we can copy the
    /// IntersectionTree from `old_node` to `new_node` instead of deleting it (I2).
    fn remove_redundant_node<const MARKER: bool>(&mut self, old_node: Node, new_node: Node) {
        if self.uniquely_covered[old_node as usize] > 0 {
            return;
        }

        self.current_solution.remove_node(old_node);
        self.age[old_node as usize] = self.round;

        let _ = self
            .domset_modifications
            .push_within_capacity(DomSetModification::Remove(old_node));

        // (I1) Re-order neighbors
        for i in (0..self.graph.degree_of(old_node)).rev() {
            let neighbor = self.graph.ith_neighbor(old_node, i);
            self.num_covered[neighbor as usize] -= 1;
            self.graph.swap_neighbors(
                neighbor,
                self.graph.ith_cross_position(old_node, i),
                self.num_covered[neighbor as usize],
            );

            if self.num_covered[neighbor as usize] == 1 {
                let dominating_node = self.graph.ith_neighbor(neighbor, 0);
                self.uniquely_covered[dominating_node as usize] += 1;

                // Normally, we would have to leave neighbor in nodes_to_update as we removed old_node
                // and need to add neighbor to IntersectionTree[new_node] later.
                // However, since we later copy/transfer IntersectionTree[old_node] to IntersectionTree[new_node]
                // in this iteration, we already have updated IntersectionTree[new_node] correctly
                // and do not need to consider it later again (except when a later old_node changes this again).
                let prev_bit = if MARKER {
                    self.in_nodes_to_update.flip_bit(neighbor)
                } else {
                    self.in_nodes_to_update.set_bit(neighbor)
                };

                if !prev_bit {
                    self.nodes_to_update.push(neighbor);
                }
            }
        }

        // Copy IntersectionTree in the first iteration as it should be a superset of the intended one for
        // new_node by (I3). This will later be further updated and corrected.
        if MARKER {
            self.intersection_forest.transfer_tree(old_node, new_node);
            self.scores[old_node as usize] = 1;
            self.sampler.add_entry(old_node, 1);
        } else {
            // (I4) Update sampler
            for &node in self.intersection_forest.get_root_nodes(old_node) {
                if node != old_node && node != new_node {
                    self.scores[node as usize] -= 1;
                    self.sampler
                        .set_bucket(node, self.scores[node as usize] as usize);
                }
            }
            self.intersection_forest.clear_tree(old_node);
        }
    }

    /// Update the IntersectionForest to maintain (I2) (I3) (I4)
    ///
    /// Remove nodes from the forest that are no longer uniquely covered
    /// or add nodes to the forest that are now uniquely covered
    fn update_forest_and_sampler(&mut self) {
        for candidate in self.nodes_to_update.drain(..) {
            if !self.in_nodes_to_update.clear_bit(candidate) {
                continue;
            }

            let dominating_node = self.graph.ith_neighbor(candidate, 0);
            if self.current_solution.is_fixed_node(dominating_node) {
                continue;
            }

            // Remove entries of IntersectionTree[dominating_node] from sampler
            for &node in self.intersection_forest.get_root_nodes(dominating_node) {
                if node != dominating_node && self.scores[node as usize] != 0 {
                    self.scores[node as usize] -= 1;
                    self.sampler
                        .set_bucket(node, self.scores[node as usize] as usize);
                }
            }

            // Update IntersectionTree[dominating_node]
            if self.num_covered[candidate as usize] == 1 {
                self.intersection_forest
                    .add_entry(dominating_node, candidate);
            } else {
                self.intersection_forest
                    .remove_entry(dominating_node, candidate);
            }

            // Add all entries of IntersectionTree[dominating_node] to sampler (insert later)
            for &node in self.intersection_forest.get_root_nodes(dominating_node) {
                if node != dominating_node {
                    self.scores[node as usize] += 1;
                    self.sampler
                        .set_bucket(node, self.scores[node as usize] as usize);
                }
            }
        }
    }

    /// Updates the best_solution to current_solution if better
    fn update_best_solution(&mut self) {
        if self.current_solution.len() >= self.best_solution.len() {
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

    fn force_removal_dms(&mut self) {
        debug_assert!(self.current_solution.num_of_non_fixed_nodes() > 1);

        let mut uncovered_nodes;

        let min_loss_node = self
            .current_solution
            .iter_non_fixed()
            .map(|u| (self.uniquely_covered[u as usize], self.age[u as usize], u))
            .min()
            .unwrap()
            .2;
        // TODO: Remove redundant nodes
        uncovered_nodes = self.force_remove_node_from_domset(min_loss_node);

        // CHECK: Is there guaranteed to be a non-fixed node here?
        let random_node = self.current_solution.sample_non_fixed(self.rng);
        uncovered_nodes += self.force_remove_node_from_domset(random_node);

        if self.rng.gen_bool(0.5) && self.current_solution.num_of_non_fixed_nodes() > 0 {
            let min_loss_node = self
                .current_solution
                .sample_many_non_fixed::<_, 50>(self.rng)
                .map(|u| (self.uniquely_covered[u as usize], self.age[u as usize], u))
                .min()
                .unwrap()
                .2;
            uncovered_nodes += self.force_remove_node_from_domset(min_loss_node);
        }

        // Repair solution
        while uncovered_nodes > 0 {
            let best_candidate = self.find_best_candidate();
            uncovered_nodes -= self.force_add_node_to_domset(best_candidate);
        }

        // TODO: repair Forest/Sampler

        self.update_best_solution();
    }

    fn force_remove_node_from_domset(&mut self, u: Node) -> usize {
        debug_assert!(self.current_solution.is_in_domset(u));

        let mut num_non_covered_nodes = 0;

        // Remove node from DomSet & update values
        self.current_solution.remove_node(u);
        self.age[u as usize] = self.round;
        self.uniquely_covered[u as usize] = 0;

        for i in (0..self.graph.degree_of(u)).rev() {
            let v = self.graph.ith_neighbor(u, i);
            self.num_covered[v as usize] -= 1;
            self.graph.swap_neighbors(
                v,
                self.graph.ith_cross_position(u, i),
                self.num_covered[v as usize],
            );

            match self.num_covered[v as usize] {
                0 => {
                    num_non_covered_nodes += 1;
                    for w in self.graph.neighbors_of(v) {
                        // TODO: increase ForceRemovalScore[v] by 1
                        if !self.in_nodes_to_update.set_bit(w) {
                            self.nodes_to_update.push(w);
                        }
                    }
                }
                1 => {
                    let dominating_node = self.graph.ith_neighbor(v, 0);
                    self.uniquely_covered[dominating_node as usize] += 1;
                }
                _ => {}
            };
        }

        num_non_covered_nodes
    }

    fn find_best_candidate(&mut self) -> Node {
        let mut max_score = 0;

        // A simulated ReservoirSampler with 1 element.
        // First element is the chosen element, second element is the number of elements inserted.
        let mut reservoir: (Node, NumNodes) = (0, 0);

        for i in (0..self.nodes_to_update.len()).rev() {
            let u = self.nodes_to_update[i];
            // TODO: replace with ForceRemovalScore
            if self.scores[u as usize] == 0 {
                self.in_nodes_to_update.clear_bit(u);
                self.nodes_to_update.swap_remove(i);
                continue;
            }

            // TODO: store SubsetDominatedNodes-Subset in the DS and check if not subset-dominated
            if true {
                // TODO: check reduced neighborhood for NumCovered >= 2 and continue

                // TODO replace with ForceRemovalScore
                if self.scores[u as usize] > max_score {
                    max_score = self.scores[u as usize];
                    // Resets reservoir
                    reservoir = (u, 1);
                } else if self.scores[u as usize] == max_score {
                    // Replaces stored node in reservoir with probability 1/N where N is the number
                    // of previous candidates with the same score
                    reservoir.1 += 1;
                    if self.rng.gen_range(0..reservoir.1) == 0 {
                        reservoir.0 = u;
                    }
                }
            }
        }

        reservoir.0
    }

    fn force_add_node_to_domset(&mut self, u: Node) -> usize {
        let prev_size = self.nodes_to_update.len();

        let mut newly_covered_nodes = 0;

        self.current_solution.add_node(u);
        self.age[u as usize] = self.round;

        for i in 0..self.graph.degree_of(u) {
            let v = self.graph.ith_neighbor(u, i);
            self.graph.swap_neighbors(
                v,
                self.graph.ith_cross_position(u, i),
                self.num_covered[v as usize],
            );
            self.num_covered[v as usize] += 1;

            match self.num_covered[v as usize] {
                1 => {
                    newly_covered_nodes += 1;
                    self.uniquely_covered[u as usize] += 1;

                    // TODO: decrease score of every neighbor of v
                }
                2 => {
                    // (I1) the first neighbor must be a dominating node
                    let former_unique_covering_node = self.graph.ith_neighbor(v, 0);
                    self.uniquely_covered[former_unique_covering_node as usize] -= 1;
                    if self.uniquely_covered[former_unique_covering_node as usize] == 0 {
                        self.nodes_to_update.push(former_unique_covering_node);
                    }
                }
                _ => {}
            };
        }

        // Remove redundant vertices
        // TODO: use self.redundant_nodes instead?
        for i in prev_size..self.nodes_to_update.len() {
            let v = self.nodes_to_update[i];
            if self.uniquely_covered[v as usize] == 0 {
                self.force_remove_node_from_domset(v);
            }
        }
        self.nodes_to_update.truncate(prev_size);

        newly_covered_nodes
    }

    /// Asserts that all current datastructures contain correct values
    #[allow(unused)]
    pub fn assert_correctness(&self) {
        let mut unique = vec![0; self.graph.len()];
        let mut scores = vec![0; self.graph.len()];
        for u in self.graph.vertices() {
            // Check (I1)
            for i in 0..self.num_covered[u as usize] {
                assert!(self
                    .current_solution
                    .is_in_domset(self.graph.ith_neighbor(u, i)));
            }

            if !self.current_solution.is_fixed_node(u) {
                // Check that only non-fixed DomSet-Nodes own trees
                assert_eq!(
                    self.current_solution.is_in_domset(u),
                    self.intersection_forest.owns_tree(u)
                );
            }

            // Check (I2)
            if self.num_covered[u as usize] == 1 {
                let dom = self.graph.ith_neighbor(u, 0);
                unique[dom as usize] += 1;
                if !self.current_solution.is_fixed_node(dom) {
                    assert!(self.intersection_forest.is_in_tree(dom, u));
                }
            }

            if self.current_solution.is_fixed_node(u) {
                continue;
            }

            // Prepare Check (I3)
            if self.current_solution.is_in_domset(u) {
                for &v in self.intersection_forest.get_root_nodes(u) {
                    if v != u {
                        scores[v as usize] += 1;
                    }
                }
            }
        }

        // Check (I3) and UniquelyCovered
        assert_eq!(self.uniquely_covered, unique);
        assert_eq!(self.scores, scores);

        for u in self.graph.vertices() {
            // Check (I4)
            assert_eq!(scores[u as usize] as usize, self.sampler.bucket_of_node(u));
        }

        // Check Sampler-Weight
        self.sampler.assert_positions();
        self.sampler.assert_total_weight();

        println!("Check completed");
    }
}

/// Helper enum to keep track of DomSet-Changes
enum DomSetModification {
    Add(Node),
    Remove(Node),
}

impl<R, G, const NUM_SAMPLER_BUCKETS: usize, const NUM_SAMPLES: usize>
    IterativeAlgorithm<DominatingSet>
    for GreedyReverseSearch<'_, R, G, NUM_SAMPLER_BUCKETS, NUM_SAMPLES>
where
    R: Rng,
    G: StaticGraph + SelfLoop,
{
    fn execute_step(&mut self) {
        self.step();
    }

    fn is_completed(&self) -> bool {
        self.is_locally_optimal
    }

    fn best_known_solution(&mut self) -> Option<DominatingSet> {
        Some(self.best_solution.clone())
    }
}

impl<R, G, const NUM_SAMPLER_BUCKETS: usize, const NUM_SAMPLES: usize>
    TerminatingIterativeAlgorithm<DominatingSet>
    for GreedyReverseSearch<'_, R, G, NUM_SAMPLER_BUCKETS, NUM_SAMPLES>
where
    R: Rng,
    G: StaticGraph + SelfLoop,
{
}

#[derive(Copy, Clone, Error)]
pub enum BaseStateError {
    #[error("nodes_to_update should be empty")]
    NodesToUpdate,
    #[error("in_nodes_to_update should be set to 00...00")]
    InNodesToUpdate,
    #[error("redundant_nodes should be empty")]
    RedundantNodes,
}

impl Debug for BaseStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

#[derive(Copy, Clone, Error)]
pub enum RevGreedyError {
    #[error("SamplerError: {0}")]
    SamplerError(SamplerError),
    #[error("IntersectionForestError: {0}")]
    IntersectionForestError(IntersectionForestError),
    #[error("{0} is not covered by any node")]
    NotCovered(Node),
    #[error("{0} covers no node uniquely")]
    RedundantDomNode(Node),
    #[error(
        "the neighborhood of {0} is not partitioned by appearance in the current dominating set"
    )]
    AdjacencyOrdering(Node),
    #[error("{0} is not inserted into the tree of its uniquely covering node {1}")]
    TreeInsertion(Node, Node),
    #[error("{0} covers {2} uniquely but {1} is stored")]
    FaultyUniquelyCovered(Node, NumNodes, NumNodes),
    #[error("{0} has score {2} but {1} is stored")]
    FaultyScore(Node, NumNodes, NumNodes),
    #[error("{0} is in sampler bucket {2} but should be in bucket {1}")]
    FaultyBucket(Node, NumNodes, NumNodes),
    #[error("the current solution has {0} fixed nodes whereas the best solution has {1}")]
    FixedNodesDifference(NumNodes, NumNodes),
    #[error("the current solution with size {0} is better than the best solution with size {1}")]
    WorseBestSolution(NumNodes, NumNodes),
    #[error("VariableError: {0}")]
    VariableBaseError(BaseStateError),
}

impl Debug for RevGreedyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl<R, G, const NUM_SAMPLER_BUCKETS: usize, const NUM_SAMPLES: usize>
    InvariantCheck<RevGreedyError>
    for GreedyReverseSearch<'_, R, G, NUM_SAMPLER_BUCKETS, NUM_SAMPLES>
where
    R: Rng,
    G: StaticGraph + SelfLoop,
{
    fn is_correct(&self) -> Result<(), RevGreedyError> {
        self.sampler
            .is_correct()
            .map_err(RevGreedyError::SamplerError)?;
        self.intersection_forest
            .is_correct()
            .map_err(RevGreedyError::IntersectionForestError)?;

        if self.current_solution.num_of_fixed_nodes() != self.best_solution.num_of_fixed_nodes() {
            return Err(RevGreedyError::FixedNodesDifference(
                self.current_solution.num_of_fixed_nodes() as NumNodes,
                self.best_solution.num_of_fixed_nodes() as NumNodes,
            ));
        }

        if self.current_solution.len() < self.best_solution.len() {
            return Err(RevGreedyError::WorseBestSolution(
                self.current_solution.len() as NumNodes,
                self.best_solution.len() as NumNodes,
            ));
        }

        if !self.nodes_to_update.is_empty() {
            return Err(RevGreedyError::VariableBaseError(
                BaseStateError::NodesToUpdate,
            ));
        }

        if self.in_nodes_to_update.cardinality() > 0 {
            return Err(RevGreedyError::VariableBaseError(
                BaseStateError::InNodesToUpdate,
            ));
        }

        if !self.redundant_nodes.is_empty() {
            return Err(RevGreedyError::VariableBaseError(
                BaseStateError::RedundantNodes,
            ));
        }

        let mut unique = vec![0; self.graph.len()];
        let mut scores = vec![0; self.graph.len()];

        for u in self.graph.vertices() {
            if self.num_covered[u as usize] == 0 {
                return Err(RevGreedyError::NotCovered(u));
            }

            for i in 0..self.num_covered[u as usize] {
                if !self
                    .current_solution
                    .is_in_domset(self.graph.ith_neighbor(u, i))
                {
                    return Err(RevGreedyError::AdjacencyOrdering(u));
                }
            }

            for i in self.num_covered[u as usize]..self.graph.degree_of(u) {
                if self
                    .current_solution
                    .is_in_domset(self.graph.ith_neighbor(u, i))
                {
                    return Err(RevGreedyError::AdjacencyOrdering(u));
                }
            }

            if self.num_covered[u as usize] == 1 {
                let dom = self.graph.ith_neighbor(u, 0);
                unique[dom as usize] += 1;
                if !self.current_solution.is_fixed_node(dom)
                    && !self.intersection_forest.is_in_tree(dom, u)
                {
                    return Err(RevGreedyError::TreeInsertion(u, dom));
                }
            }

            if self.current_solution.is_non_fixed_node(u) {
                for &v in self.intersection_forest.get_root_nodes(u) {
                    if v != u {
                        scores[v as usize] += 1;
                    }
                }
            }
        }

        for u in self.graph.vertices() {
            if self.uniquely_covered[u as usize] != unique[u as usize] {
                return Err(RevGreedyError::FaultyUniquelyCovered(
                    u,
                    self.uniquely_covered[u as usize],
                    unique[u as usize],
                ));
            }

            if self.scores[u as usize] != scores[u as usize] {
                return Err(RevGreedyError::FaultyScore(
                    u,
                    self.scores[u as usize],
                    scores[u as usize],
                ));
            }

            let bucket = scores[u as usize].min((NUM_SAMPLER_BUCKETS - 2) as NumNodes);
            let real_bucket = self.sampler.bucket_of_node(u) as NumNodes;
            if bucket != real_bucket {
                return Err(RevGreedyError::FaultyBucket(u, real_bucket, bucket));
            }

            if self.current_solution.is_non_fixed_node(u) && unique[u as usize] == 0 {
                return Err(RevGreedyError::RedundantDomNode(u));
            }
        }

        Ok(())
    }
}
