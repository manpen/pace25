use std::{fmt::Debug, string::ParseError, time::Instant};

use log::info;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use thiserror::Error;

use crate::{
    errors::InvariantCheck,
    graph::*,
    prelude::{IterativeAlgorithm, TerminatingIterativeAlgorithm},
    utils::{
        DominatingSet,
        intersection_forest::{IntersectionForest, IntersectionForestError},
        sampler::{SamplerError, WeightedPow2Sampler},
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
///description of these states.
pub struct GreedyReverseSearch<
    G,
    const NUM_SAMPLER_BUCKETS: usize = 5,
    const NUM_SAMPLES: usize = 5,
> where
    G: StaticGraph + SelfLoop,
{
    /// A reference to the graph: mutable access is needed as we need to re-order adjacency lists
    graph: G,

    /// The current solution we operate on
    current_solution: DominatingSet,

    /// The currently best known solution of the algorithm
    best_solution: DominatingSet,

    /// A sampler for sampling nodes with weights that are powers of 2.
    sampler: WeightedPow2Sampler<NUM_SAMPLER_BUCKETS>,

    /// RNG used for sampling
    rng: Pcg64Mcg,

    /// List of all nodes that are either currently inserted into an IntersectionTree and need to
    /// be removed to maintain (I2) or need to be added to an IntersectionTree to maintain (I2)
    nodes_to_update: Vec<(Node,Node)>,

    /// Helper BitSet to easily identify if a node is pushed to `nodes_to_update`
    in_nodes_to_update: BitSet,

    /// Number of (incident dominating nodes, xoring of dominating nodes set)
    num_covered: Vec<(NumNodes, NumNodes)>,

    /// Number of nodes this (dominating) nodes covers uniquely (no other dominating node covers) (V9)
    uniquely_covered: Vec<NumNodes>,

    /// Nodes u that can possibly be removed from the DomSet as uniquely_covered[u] = 0
    redundant_nodes: Vec<Node>,

    /// Number of appearances in entries of root nodes in the IntersectionForest
    scores: Vec<NumNodes>,

    /// Last time a node was added/removed from the DomSet
    age: Vec<u64>,

    /// Current iteration
    round: u64,


    working_set: DominatingSet,

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

    /// BitSet indicating all nodes that will never appear in an optimal solution and can be
    /// disregarded
    non_optimal_nodes: BitSet,

    /// Temp-List of all non-covered nodes in a force_removal subroutine
    non_covered_nodes: Vec<Node>,

    /// Helper BitSet to easily identify if a node is pushed to `non_covered_nodes`
    in_non_covered_nodes: BitSet,

    /// Number of uncovered neighbors of a node. Except inside a forced_removal subroutine, this
    /// should always be [0,...,0].
    ///
    /// TODO: combine with self.scores such that scores = scores + CONSTANT * num_uncovered_neighbors
    num_uncovered_neighbors: Vec<NumNodes>,

    expunge_frequency: Vec<NumNodes>,

    /// A helper BitSet
    helper_bitset: BitSet,

    /// Which forced removal procedure to apply
    forced_rule: ForcedRemovalRuleType,

    /// BitSet indicating whether a node is permanently covered by a (removed) fixed node
    is_perm_covered: BitSet,

    verbose_logging: bool,
    previous_improvement: u64,
    start_time: Instant,


    hitting_score: Vec<NumNodes>
}

impl<G, const NUM_SAMPLER_BUCKETS: usize, const NUM_SAMPLES: usize>
    GreedyReverseSearch<G, NUM_SAMPLER_BUCKETS, NUM_SAMPLES>
where
    G: StaticGraph + SelfLoop,
{
    /// Creates a new instance of the algorithm for a given graph and a starting DomSet which must be valid.
    /// Runs Subset-Reduction beforehand to further reduce the DomSet and removes redundant nodes afterwards.
    pub fn new(
        graph: G,
        mut initial_solution: DominatingSet,
        is_perm_covered: BitSet,
        non_optimal_nodes: BitSet,
        seeding_rng: &mut impl Rng,
        rule: ForcedRemovalRuleType
    ) -> Self {
        assert!(initial_solution.is_valid_given_previous_cover(&graph, &is_perm_covered));
        assert!(graph.len() > 0);

        let n = graph.len();

        // Initialize NumCovered with 2 for permanently covered nodes to prevent unique-checks
        let mut num_covered: Vec<(NumNodes,NumNodes)> = (0..graph.number_of_nodes())
            .map(|i| (is_perm_covered.get_bit(i) as NumNodes * 2,0))
            .collect();
        let mut age = vec![0; n];

        // (I1) Reorder adjacency lists such that dominating nodes appear first
        for u in initial_solution.iter() {
            age[u as usize] = 1;
            for v in graph.neighbors_of(u) {
                num_covered[v as usize].0 += 1;
                num_covered[v as usize].1 ^= u;
            }
        }

        let mut uniquely_covered: Vec<NumNodes> = vec![0; graph.len()];
        graph.vertices().for_each(|u| {
            if num_covered[u as usize].0 == 1 {
                if !initial_solution.is_in_domset(num_covered[u as usize].1) {
                    panic!("Something wrong!");
                }
                uniquely_covered[num_covered[u as usize].1 as usize] += 1;
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
            for v in graph.neighbors_of(u)  {
                num_covered[v as usize].0 -= 1;
                num_covered[v as usize].1 ^= u;

                if num_covered[v as usize].0 == 1 {
                    uniquely_covered[num_covered[v as usize].1 as usize] += 1;
                }
            }
        }

        // Instantiate sampler and IntersectionForest with reduced neighbor-set
        let mut sampler = WeightedPow2Sampler::new(n);
        let mut scores = vec![0; n];

        let mut intersection_forest =
            IntersectionForest::new_unsorted(graph.extract_csr_repr(), non_optimal_nodes.clone());

        // Insert uniquely covered neighbors of dominating nodes into IntersectionTrees & Sampler
        for u in initial_solution.iter() {
            for v in graph.neighbors_of(u) {
                debug_assert!(num_covered[v as usize].0 > 0);
                if num_covered[v as usize].0 == 1 {
                    if num_covered[v as usize].1 != u {
                        panic!("Something wrong!");
                    }
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

        let rng = Pcg64Mcg::seed_from_u64(seeding_rng.r#gen());
        Self {
            graph,
            current_solution,
            best_solution,
            sampler,
            rng,
            nodes_to_update: Vec::with_capacity(n),
            in_nodes_to_update: BitSet::new(n as NumNodes),
            num_covered,
            uniquely_covered,
            redundant_nodes: Vec::with_capacity(n),
            scores,
            age,
            intersection_forest,
            round: 1,
            domset_modifications: Vec::with_capacity(1 + n / 64),
            non_optimal_nodes,
            non_covered_nodes: Vec::with_capacity(n),
            in_non_covered_nodes: BitSet::new(n as NumNodes),
            helper_bitset: BitSet::new(n as NumNodes),
            num_uncovered_neighbors: vec![0; n],
            forced_rule: rule,
            hitting_score: vec![0;n],
            is_perm_covered,
            verbose_logging: false,
            previous_improvement: 0,
            start_time: Instant::now(),
            expunge_frequency: vec![0;n],
            working_set: DominatingSet::new(n as NumNodes),
        }
    }

    pub fn enable_verbose_logging(&mut self) {
        self.verbose_logging = true;
    }

    /// Run one iteration of the algorithm:
    ///
    /// 1. Sample a node from sampler
    /// 2. Insert the node into the DomSet
    /// 3. Remove all now redundant nodes of the DomSet
    /// 4. Update IntersectionTrees/Scores/Sampler accordingly
    pub fn step(&mut self) {
        let rnd_ch = self.rng.r#gen::<f32>();
        let diff = 1.0 / ((self.current_solution.len()+1) - self.best_solution.len()) as f32;
        // Try to escape local minima every 1000 steps
        //
        // TODO: find better threshold
        if (((self.round-self.previous_improvement) % 10_000 == 0) && (rnd_ch < diff)) || (self.round < self.previous_improvement) {
            self.working_set.clear();
            match self.forced_rule {
                ForcedRemovalRuleType::FRDR => {
                    let rnd_choice = self.rng.r#gen::<f32>();
                    let removable = if rnd_choice < 0.33 {
                        Some(self.current_solution.sample_non_fixed(&mut self.rng))
                    }
                    else if rnd_choice < 0.66 {
                        (0..NUM_SAMPLES).map(|_| self.current_solution.sample_non_fixed(&mut self.rng)).filter(|x| {
                            self.intersection_forest.get_root_nodes(*x).len() == 1
                        })
                        .min_by_key(|a| (self.expunge_frequency[*a as usize], self.age[*a as usize]))
                    }
                    else {
                        (0..NUM_SAMPLES).map(|_| self.current_solution.sample_non_fixed(&mut self.rng)).filter(|x| {
                            self.intersection_forest.get_root_nodes(*x).len() == 1
                        })
                        .min_by_key(|a| (self.uniquely_covered[*a as usize], self.age[*a as usize]))
                    };
                    if let Some(non_removable_node) = removable {
                        self.expunge_frequency[non_removable_node as usize] += 1;
                        debug_assert!(self.intersection_forest.get_root_nodes(non_removable_node).len() == 1);
                        debug_assert!(self.redundant_nodes.len() == 0);
                        for nb in self.graph.neighbors_of(non_removable_node) {
                            if self.num_covered[nb as usize].0 == 1 {
                                self.in_nodes_to_update.set_bit(nb);

                                for j in self.graph.neighbors_of(nb).filter(|x| !self.non_optimal_nodes.get_bit(*x))  {
                                    self.hitting_score[j as usize] += 1;
                                    if self.hitting_score[j as usize] == 1 {
                                        self.redundant_nodes.push(j);
                                    }
                                }
                            }
                        }

                        let mut added_nodes_len = 0;
                        let mut uncovered_nodes = self.uniquely_covered[non_removable_node as usize];
                        'outer: while uncovered_nodes > 0 {
                            let mut max_score = 0;
                            let mut max_pos = usize::MAX;
                            let mut min_age = u64::MAX;

                            for (idx,nd) in self.redundant_nodes[added_nodes_len..].iter().enumerate() {
                                if *nd == non_removable_node {continue;}
                                if self.hitting_score[*nd as usize] > max_score || (self.hitting_score[*nd as usize] == max_score && self.age[*nd as usize] < min_age) {
                                    max_score = self.hitting_score[*nd as usize];
                                    max_pos = idx+added_nodes_len;
                                    min_age = self.age[*nd as usize];
                                }
                            }

                            if max_pos >= self.redundant_nodes.len() {
                                self.redundant_nodes.iter().for_each(|x| self.hitting_score[*x as usize] = 0);
                                for nb in self.graph.neighbors_of(non_removable_node) {
                                    self.in_nodes_to_update.clear_bit(nb);
                                }
                                self.redundant_nodes.clear();
                                added_nodes_len = 0;
                                break 'outer;
                            }

                            let nd = self.redundant_nodes[max_pos];
                            self.redundant_nodes.swap(max_pos, added_nodes_len);
                            added_nodes_len+=1;

                            for nb in self.graph.neighbors_of(nd) {
                                if self.in_nodes_to_update.get_bit(nb) {
                                    uncovered_nodes -= 1;
                                    self.in_nodes_to_update.clear_bit(nb);

                                    for j in self.graph.neighbors_of(nb).filter(|x| !self.non_optimal_nodes.get_bit(*x))  {
                                        if self.hitting_score[j as usize] > 0 {
                                            self.hitting_score[j as usize] -= 1;
                                        }
                                    }
                                }
                            }
                        }

                        if added_nodes_len > 0 {
                            let mut res: Vec<Node> = self.redundant_nodes[..added_nodes_len].to_vec();
                            res.sort_by_key(|u| self.age[*u as usize]);
                            self.redundant_nodes.clear();
                            for x in res.iter() {
                                self.add_node_to_domset(*x);

                                // Prefer nodes that have been unchanged for longer
                                self.redundant_nodes.sort_by_key(|u| self.age[*u as usize]);

                                // Remove redundant nodes from DomSet
                                if !self.redundant_nodes.is_empty() {
                                    self.remove_redundant_node::<true>(self.redundant_nodes[0], *x);
                                    for i in 1..self.redundant_nodes.len() {
                                        self.remove_redundant_node::<false>(self.redundant_nodes[i], *x);
                                    }
                                    self.redundant_nodes.clear();
                                }
                                // Update IntersectionForest/Sampler for all remaining nodes_to_update
                                self.update_forest_and_sampler();
                            }
                            for x in res.into_iter() {
                                if self.current_solution.is_in_domset(x) {
                                    self.remove_redundant_node::<false>(x, u32::MAX);
                                }
                            }
                            self.update_forest_and_sampler();
                            self.update_best_solution();
                        }
                    }
                },
                ForcedRemovalRuleType::None => {},
                _ => unimplemented!("Rule not implemented!"),
            };

            self.round += 1;
            return;
        }

        // Sample node: if no node can be sampled, current solution is optimal
        let proposed_node = if let Some(node) = self.draw_node() {
            node
        } else {
            self.previous_improvement = self.round+1;
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
        let rand_choice = self.rng.r#gen::<f32>();
        if rand_choice < 0.8 && self.working_set.len() != 0 {
            let mut sample_node: Option<Node> = None;
            let mut sample_bucket = 0;
            let mut sample_age = u64::MAX;

            for _ in 0..NUM_SAMPLES {
                let nd = self.working_set.sample_non_fixed(&mut self.rng);
                let score = self.sampler.bucket_of_node(nd);
                if score == 0 {
                    self.working_set.remove_node(nd);
                    if self.working_set.len() == 0 {
                        break;
                    }
                    continue;
                }

                if score > sample_bucket || (score == sample_bucket && sample_age > self.age[nd as usize]) {
                    sample_node = Some(nd);
                    sample_bucket = score;
                    sample_age = self.age[nd as usize];
                }
            }
            sample_node
        }
        else {
            if self.sampler.is_empty() {
                return None;
            }

            let mut sample_node = None;
            let mut sample_bucket = 0;
            let mut sample_age = 0;

            self.sampler
                .sample_many::<_, NUM_SAMPLES>(&mut self.rng, |bucket, node| {
                    if sample_bucket == bucket && (sample_age < self.age[node as usize]) {
                        return;
                    }

                    sample_node = Some(node);
                    sample_bucket = bucket;
                    sample_age = self.age[node as usize];
                });

            sample_node
        }
    }

    /// Adds a node to the DomSet that was not part of it before.
    /// The node must be able to directly replace at least one node in the DomSet, ie. must be part
    /// of Sampler (I4).
    /// Updates the corresponding datastructures to maintain invariants.
    fn add_node_to_domset(&mut self, u: Node) {
        debug_assert!(!self.current_solution.is_in_domset(u));

        // Breaks (I1)
        self.current_solution.add_node(u);
        if self.working_set.is_in_domset(u) {
            self.working_set.remove_node(u);
        }
        // (I3) dominating nodes have no score
        self.scores[u as usize] = 0;
        self.age[u as usize] = self.round;
        if self.sampler.is_in_sampler(u) {
            // (I4) Node must be part of Sampler
            self.sampler.remove_entry(u);
        }

        let _ = self
            .domset_modifications
            .push_within_capacity(DomSetModification::Add(u));

        // (I1) Update adjacency lists as well as (I2) num_covered/uniquely_covered
        //
        // If a previously uniquely covered node is now not longer uniquely covered,
        // add it to nodes_to_update as we must later update its IntersectionTree-Appearance
        for neighbor in self.graph.neighbors_of(u)  {
            self.num_covered[neighbor as usize].0 += 1;

            // If self.is_perm_covered is true, NumCovered must now be at least 3
            if self.num_covered[neighbor as usize].0 == 2 {
                // (I1) the first neighbor must be a dominating node
                let former_unique_covering_node = self.num_covered[neighbor as usize].1;
                if !self.current_solution.is_in_domset(former_unique_covering_node) {
                    panic!("Error!");
                }
                self.uniquely_covered[former_unique_covering_node as usize] -= 1;
                if !self.in_nodes_to_update.set_bit(neighbor) {
                    self.nodes_to_update.push((former_unique_covering_node,neighbor));
                }

                if self.uniquely_covered[former_unique_covering_node as usize] == 0 {
                    self.redundant_nodes.push(former_unique_covering_node);
                }
            }
            self.num_covered[neighbor as usize].1 ^= u;
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
        if !MARKER {
            self.previous_improvement = self.round-1;
        }
        self.expunge_frequency[old_node as usize] += 1;

        // Breaks (I1)
        self.current_solution.remove_node(old_node);
        self.age[old_node as usize] = self.round;

        let _ = self
            .domset_modifications
            .push_within_capacity(DomSetModification::Remove(old_node));

        // (I1) Re-order neighbors
        for neighbor in self.graph.neighbors_of(old_node) {
            self.num_covered[neighbor as usize].0 -= 1;
            self.num_covered[neighbor as usize].1 ^= old_node;

            // If self.is_perm_covered is true, NumCovered is at least 2
            if self.num_covered[neighbor as usize].0 == 1 {
                let dominating_node = self.num_covered[neighbor as usize].1;
                if !self.current_solution.is_in_domset(dominating_node) {
                    panic!("Something is wrong here!");
                }
                self.uniquely_covered[dominating_node as usize] += 1;

                // Normally, we would have to leave neighbor in nodes_to_update as we removed old_node
                // and need to add neighbor to IntersectionTree[new_node] later.
                // However, since we later copy/transfer IntersectionTree[old_node] to IntersectionTree[new_node]
                // in this iteration, we already have updated IntersectionTree[new_node] correctly
                // and do not need to consider it later again (except when a later old_node changes this again).
                if MARKER {
                    self.in_nodes_to_update.flip_bit(neighbor);
                } else {
                    self.in_nodes_to_update.set_bit(neighbor);
                }

                self.nodes_to_update.push((dominating_node, neighbor));
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
        for (dominating_node,candidate) in self.nodes_to_update.iter().rev() {
            if !self.in_nodes_to_update.clear_bit(*candidate) {
                if self.in_nodes_to_update.cardinality() == 0 {
                    break;
                }
                continue;
            }

            // Remove entries of IntersectionTree[dominating_node] from sampler
            for &node in self.intersection_forest.get_root_nodes(*dominating_node) {
                if node != *dominating_node && self.scores[node as usize] != 0 {
                    self.scores[node as usize] -= 1;
                    self.sampler
                        .set_bucket(node, self.scores[node as usize] as usize);
                }
            }
            // Update IntersectionTree[dominating_node]
            if self.num_covered[*candidate as usize].0 == 1 {
                self.intersection_forest
                    .add_entry(*dominating_node, *candidate);
            } else {
                self.intersection_forest
                    .remove_entry(*dominating_node, *candidate);
            }

            // Add all entries of IntersectionTree[dominating_node] to sampler
            if self.num_covered[*candidate as usize].0 > 1 {
                for &node in self.intersection_forest.get_root_nodes(*dominating_node) {
                    if node != *dominating_node {
                        self.scores[node as usize] += 1;
                        self.sampler
                            .set_bucket(node, self.scores[node as usize] as usize);
                        if !self.current_solution.is_in_domset(node) {
                            if !self.working_set.is_in_domset(node) {
                                self.working_set.add_node(node);
                            }
                        }
                    }
                }
            }
            else {
                for &node in self.intersection_forest.get_root_nodes(*dominating_node) {
                    if node != *dominating_node {
                        self.scores[node as usize] += 1;
                        self.sampler
                            .set_bucket(node, self.scores[node as usize] as usize);
                    }
                }
            }

            if self.in_nodes_to_update.cardinality() == 0 {
                break;
            }
        }
        self.nodes_to_update.clear();
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

        if self.verbose_logging {
            info!(
                " Better solution: size={:6}, round={:9}, gap={:9}, time={:7}ms",
                self.best_solution.len(),
                self.round,
                self.round - self.previous_improvement,
                self.start_time.elapsed().as_millis()
            );
            self.previous_improvement = self.round;
        }
    }

    pub fn current_round(&self) -> u64 {
        self.round
    }

    pub fn current_score(&self) -> NumNodes {
        self.best_solution.len() as NumNodes
    }
}

/// Helper enum to keep track of DomSet-Changes
enum DomSetModification {
    Add(Node),
    Remove(Node),
}

/// Helper enum to generalize type of forced node removals
#[allow(unused)]
#[derive(Debug, Copy, Clone)]
enum ForcedRemovalNodeType {
    MinLoss,
    Random,
    RandomMinLoss,
    Fixed(Node),
}

impl<G, const NUM_SAMPLER_BUCKETS: usize, const NUM_SAMPLES: usize>
    IterativeAlgorithm<DominatingSet> for GreedyReverseSearch<G, NUM_SAMPLER_BUCKETS, NUM_SAMPLES>
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

impl<G, const NUM_SAMPLER_BUCKETS: usize, const NUM_SAMPLES: usize>
    TerminatingIterativeAlgorithm<DominatingSet>
    for GreedyReverseSearch<G, NUM_SAMPLER_BUCKETS, NUM_SAMPLES>
where
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
    #[error("non_covered_nodes should be empty")]
    NonCoveredNodes,
    #[error("in_non_covered_nodes should be set to 00...00")]
    InNonCoveredNodes,
    #[error("num_uncovered_neighbors should be set to 0,0,...,0,0")]
    NumUncoveredNeighbors,
    #[error("helper_bitset should be set to 00...00")]
    HelperBitset,
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

impl<G, const NUM_SAMPLER_BUCKETS: usize, const NUM_SAMPLES: usize> InvariantCheck<RevGreedyError>
    for GreedyReverseSearch<G, NUM_SAMPLER_BUCKETS, NUM_SAMPLES>
where
    G: StaticGraph + SelfLoop,
{
    fn is_correct(&self) -> Result<(), RevGreedyError> {
        self.sampler
            .is_correct()
            .map_err(RevGreedyError::SamplerError)?;
        self.intersection_forest
            .is_correct()
            .map_err(RevGreedyError::IntersectionForestError)?;

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

        if !self.non_covered_nodes.is_empty() {
            return Err(RevGreedyError::VariableBaseError(
                BaseStateError::NonCoveredNodes,
            ));
        }

        if self.in_non_covered_nodes.cardinality() > 0 {
            return Err(RevGreedyError::VariableBaseError(
                BaseStateError::InNonCoveredNodes,
            ));
        }

        for u in self.graph.vertices() {
            if self.num_uncovered_neighbors[u as usize] > 0 {
                return Err(RevGreedyError::VariableBaseError(
                    BaseStateError::NumUncoveredNeighbors,
                ));
            }
        }

        if self.helper_bitset.cardinality() > 0 {
            return Err(RevGreedyError::VariableBaseError(
                BaseStateError::HelperBitset,
            ));
        }

        let mut unique = vec![0; self.graph.len()];
        let mut scores = vec![0; self.graph.len()];

        for u in self.graph.vertices() {
            if self.num_covered[u as usize].0 == 0 {
                return Err(RevGreedyError::NotCovered(u));
            }

            if self.num_covered[u as usize].0 == 1 && !self.is_perm_covered.get_bit(u) {
                let dom = self.num_covered[u as usize].1;
                unique[dom as usize] += 1;
                if !self.intersection_forest.is_in_tree(dom, u) {
                    return Err(RevGreedyError::TreeInsertion(u, dom));
                }
            }

            if self.current_solution.is_in_domset(u) {
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

            if self.current_solution.is_in_domset(u) && unique[u as usize] == 0 {
                return Err(RevGreedyError::RedundantDomNode(u));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ForcedRemovalRuleType {
    DMS = 0,
    BFS2 = 1,
    BFS3 = 2,
    BFS4 = 3,
    BFSP2 = 4,
    BFSP3 = 5,
    BFSP4 = 6,
    FRDR = 7,
    None = 8,
}

impl std::str::FromStr for ForcedRemovalRuleType {
    type Err = ParseError;
    fn from_str(rule: &str) -> Result<Self, ParseError> {
        match rule {
            "0" => Ok(ForcedRemovalRuleType::DMS),
            "1" => Ok(ForcedRemovalRuleType::BFS2),
            "2" => Ok(ForcedRemovalRuleType::BFS3),
            "3" => Ok(ForcedRemovalRuleType::BFS4),
            "4" => Ok(ForcedRemovalRuleType::BFSP2),
            "5" => Ok(ForcedRemovalRuleType::BFSP3),
            "6" => Ok(ForcedRemovalRuleType::BFSP4),
            "7" => Ok(ForcedRemovalRuleType::FRDR),
            "8" => Ok(ForcedRemovalRuleType::None),
            //TODO: Fix this
            _ => panic!("No such rule!"),
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::{Rng, SeedableRng, seq::SliceRandom};
    use rand_pcg::Pcg64Mcg;

    use super::*;

    fn random_initial_solution(rng: &mut impl Rng, graph: &impl AdjacencyList) -> DominatingSet {
        let mut covered = graph.vertex_bitset_unset();
        let mut node_order = graph.vertices_range().collect_vec();
        node_order.shuffle(rng);
        let mut domset = DominatingSet::new(graph.number_of_nodes());
        while let Some(u) = node_order.pop() {
            if graph.closed_neighbors_of(u).any(|v| !covered.get_bit(v)) {
                domset.add_node(u);
                covered.set_bits(graph.closed_neighbors_of(u));
                if covered.are_all_set() {
                    break;
                }
            }
        }
        domset
    }

    #[test]
    /// Randomly generate G(n,p) graphs and check that the algorithm produces a feasible solution
    fn full_graph() {
        let mut rng = Pcg64Mcg::seed_from_u64(123456);
        for _ in 0..500 {
            let graph = AdjArray::random_gnp(&mut rng, 100, 0.03);
            let initial_domset = random_initial_solution(&mut rng, &graph);

            let domset = {
                let csr_graph = CsrGraph::from_edges(graph.number_of_nodes(), graph.edges(true));
                let mut algo = GreedyReverseSearch::<_, 10, 10>::new(
                    csr_graph,
                    initial_domset,
                    graph.vertex_bitset_unset(),
                    graph.vertex_bitset_unset(),
                    &mut rng,
                );

                for _ in 0..50 {
                    algo.step()
                }

                assert!(algo.is_correct().is_ok());
                algo.best_known_solution().unwrap()
            };

            assert!(domset.is_valid(&graph));
        }
    }

    #[test]
    /// Randomly generate G(n,p) graphs and check that the algorithm produces a feasible solution,
    /// even if we delete some nodes selected by the initial solution
    fn with_deleted_nodes() {
        let mut rng = Pcg64Mcg::seed_from_u64(123456);
        for _ in 0..500 {
            let org_graph = AdjArray::random_gnp(&mut rng, 100, 0.03);
            let mut graph = org_graph.clone();
            let initial_domset = random_initial_solution(&mut rng, &graph);

            let mut perm_covered = graph.vertex_bitset_unset();
            {
                let mut in_sol = initial_domset.iter().collect_vec();
                let (to_be_fixed, _) = in_sol.partial_shuffle(&mut rng, 5);
                for u in to_be_fixed {
                    perm_covered.set_bits(graph.closed_neighbors_of(*u));
                    graph.remove_edges_at_node(*u);
                }
            }

            // select first 10 nodes which are perm covered (since its a gnp graph that sufficiently random)
            let mut non_opt_nodes = BitSet::new_with_bits_set(
                graph.number_of_nodes(),
                perm_covered.iter_set_bits().take(10),
            );
            non_opt_nodes -= &perm_covered;

            let domset = {
                let csr_graph = CsrGraph::from_edges(graph.number_of_nodes(), graph.edges(true));
                let mut algo = GreedyReverseSearch::<_, 10, 10>::new(
                    csr_graph,
                    initial_domset,
                    perm_covered.clone(),
                    non_opt_nodes.clone(),
                    &mut rng,
                );

                for _ in 0..50 {
                    algo.step()
                }

                assert!(algo.is_correct().is_ok());
                algo.best_known_solution().unwrap()
            };

            assert!(domset.is_valid_given_previous_cover(&graph, &perm_covered));
            for non_opt in non_opt_nodes.iter_set_bits() {
                assert!(!domset.is_in_domset(non_opt));
            }
        }
    }

    #[test]
    /// Randomly generate G(n,p) graphs and check that the algorithm produces a feasible solution,
    /// without selecting `forbidden` (= non_optimal) nodes
    fn non_opt_nodes() {
        let mut rng = Pcg64Mcg::seed_from_u64(123456);
        for _ in 0..500 {
            let org_graph = AdjArray::random_gnp(&mut rng, 100, 0.03);
            let graph = org_graph.clone();
            let initial_domset = random_initial_solution(&mut rng, &graph);

            let mut non_opt_nodes = graph.vertex_bitset_unset();
            while non_opt_nodes.cardinality() < 5 {
                let u = rng.gen_range(graph.vertices_range());
                if initial_domset.is_in_domset(u) {
                    continue;
                }
                non_opt_nodes.set_bit(u);
            }

            let domset = {
                let csr_graph = CsrGraph::from_edges(graph.number_of_nodes(), graph.edges(true));
                let mut algo = GreedyReverseSearch::<_, 10, 10>::new(
                    csr_graph,
                    initial_domset,
                    graph.vertex_bitset_unset(),
                    non_opt_nodes.clone(),
                    &mut rng,
                );

                for _ in 0..50 {
                    algo.step()
                }

                assert!(algo.is_correct().is_ok());
                algo.best_known_solution().unwrap()
            };

            assert!(domset.is_valid(&graph));

            for non_opt in non_opt_nodes.iter_set_bits() {
                assert!(!domset.is_in_domset(non_opt));
            }
        }
    }
}
