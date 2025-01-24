use std::cmp::Reverse;

use rand::Rng;
use rand_distr::Distribution;

use crate::{
    graph::*,
    kernelization::subset_reduction,
    prelude::{IterativeAlgorithm, TerminatingIterativeAlgorithm},
    utils::{intersection_forest::IntersectionForest, sampler::WeightedPow2Sampler, DominatingSet},
};

/// # GreedyReverseSearch
///
/// An iterative algorithm that replaces at least one node in the current DomSet by another. If
/// possible, replace two or more nodes by one.
///
/// The algorithm stores for each node u in the dominating set how many nodes they uniquely cover,
/// ie. how many nodes are *only* covered by u. If this number is 0, the node is considered
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
/// now redundant (ie. score[v] > 1).
///
/// See descriptions of variables for more information.
/// We have multiple invariants throughout the algorithm that we need to maintain:
/// (I1) Adjacency-Lists are partitioned by appearance in the current DomSet: nodes in the DomSet appear first, then nodes not in the DomSet
/// (I2) Only nodes u with num_covered[u] = 1 are inserted into the IntersectionForest
/// (I3) scores[u] is equivalent to the number of IntersectionTrees where u is stored in the root (as an entry, not the node itself): excluded are dominating nodes
/// (I4) if scores[u] > 0, u is inserted into the Sampler
pub struct GreedyReverseSearch<
    'a,
    R,
    G,
    const NUM_SAMPLER_BUCKETS: usize = 8,
    const NUM_SAMPLES: usize = 10,
> where
    R: Rng,
    G: StaticGraph + SelfLoop,
    <G as ToSliceRepresentation>::SliceRepresentation: ReduceGraphNodes + SelfLoop + Default,
{
    /// A reference to the graph: mutable access is needed as we need to re-order adjacency lists
    graph: &'a mut G,

    /// The current solution we operate on
    current_solution: DominatingSet,
    /// The currently best known solution of the algorithm
    best_solution: DominatingSet,
    /// Is the algorithm stuck in a (local) optimum?
    ///
    /// Will be true if the sampler is empty, ie. the algorithm can no longer improve the solution.
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
    /// Additional supporting vector used in `self.update_trees_and_sampler()` to temporarily store
    /// nodes that need to be re-inserted into the sampler.
    ///
    /// To prevent re-allocations, this is initialized inside the algorithm itself.
    temp_nodes: Vec<Node>,

    /// Number of incident dominating nodes
    ///
    /// (I1) Neighbors[u][..num_covered[u]] is a subset of the current DomSet
    num_covered: Vec<NumNodes>,
    /// Number of nodes this (dominating) nodes covers uniquely (no other dominating node covers)
    uniquely_covered: Vec<NumNodes>,

    /// Nodes u that can possibly be removed from the DomSet as uniquely_covered[u] = 0
    redundant_nodes: Vec<Node>,
    /// Number of appearences in entries of root nodes in the IntersectionForest
    scores: Vec<NumNodes>,

    /// Last time a node was added/removed from the DomSet
    age: Vec<u64>,
    /// Current iteration
    round: u64,

    /// IntersectionForest
    ///
    /// Every node u in the DomSet is assigned an IntersectionTree. Nodes that are uniquely covered by this
    /// u are then inserted into the IntersectionTree of u. IntersectionTree[u] thus stores all nodes in its root
    /// that are incident to *all* uniquely covered nodes of u and can thus replace u in the DomSet.
    ///
    /// v in root of IntersectionTree[u] ==> scores[v] > 0 ==> v in sampler ==> v can be sampled to replace u
    ///
    /// Note that we only *really* consider neighbors that are not subset-dominated and thus can appear in any
    /// optimal DomSet without the possibility of directly replacing them.
    intersection_forest: IntersectionForest<<G as ToSliceRepresentation>::SliceRepresentation>,

    /// Keep track of all applied modifications to current_solution to also apply them to
    /// best_solution when new best solution is found
    domset_modifications: Vec<DomSetModification>,
}

impl<'a, R, G, const NUM_SAMPLER_BUCKETS: usize, const NUM_SAMPLES: usize>
    GreedyReverseSearch<'a, R, G, NUM_SAMPLER_BUCKETS, NUM_SAMPLES>
where
    R: Rng,
    G: StaticGraph + SelfLoop,
    <G as ToSliceRepresentation>::SliceRepresentation: ReduceGraphNodes + SelfLoop + Default,
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
                temp_nodes: Vec::new(),
                num_covered: Vec::new(),
                uniquely_covered: Vec::new(),
                redundant_nodes: Vec::new(),
                scores: Vec::new(),
                age: Vec::new(),
                intersection_forest: IntersectionForest::new(
                    <G as ToSliceRepresentation>::SliceRepresentation::default(),
                ),
                round: 1,
                domset_modifications: Vec::new(),
            };
        }

        let n = graph.number_of_nodes() as usize;

        // Run Subset-Reduction and create reduced edge set
        let mut neighborhoods = graph.to_slice_representation();
        let non_optimal_nodes = subset_reduction(&mut neighborhoods, &mut initial_solution);
        neighborhoods.filter_out_nodes(&non_optimal_nodes);

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

        // Instantiate sampler and merge trees with reduced neighbor-set
        let mut sampler = WeightedPow2Sampler::new(n);
        let mut scores = vec![0; n];
        let mut intersection_forest = IntersectionForest::new_sorted(neighborhoods);

        // Insert uniquely covered neighbors of dominating nodes into MergeTrees & Sampler
        for u in initial_solution.iter_non_fixed() {
            for v in graph.neighbors_of(u) {
                if num_covered[v as usize] <= 1 {
                    intersection_forest.add_entry(u, v);
                }
            }

            for &v in intersection_forest.get_root_nodes(u) {
                if u != v {
                    scores[v as usize] += 1;
                    if sampler.is_in_sampler(v) {
                        sampler.remove_entry(v);
                    }
                    sampler.add_entry(v, scores[v as usize] as usize - 1);
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
            temp_nodes: Vec::new(),
            num_covered,
            uniquely_covered,
            redundant_nodes: Vec::new(),
            scores,
            age,
            intersection_forest,
            round: 1,
            domset_modifications: Vec::new(),
        }
    }

    /// Run one iteration of the algorithm:
    ///
    /// 1. Sample a node from sampler
    /// 2. Insert the node into the DomSet
    /// 3. Remove all now redundant nodes of the DomSet
    /// 4. Update MergeTrees/Scores/Sampler accordingly
    pub fn step(&mut self) {
        // Sample node: if no node can be sampled, current solution is optimal
        let proposed_node = if let Some(node) = self.draw_node() {
            node
        } else {
            self.is_locally_optimal = true;
            return;
        };

        self.round += 1;

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

        (0..NUM_SAMPLES)
            .map(|_| {
                let node = self.sampler.sample(&mut self.rng);
                (
                    self.scores[node as usize],
                    Reverse(self.age[node as usize]),
                    node,
                )
            })
            .max()
            .map(|(_, _, x)| x)
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

        self.domset_modifications.push(DomSetModification::Add(u));

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

        self.domset_modifications
            .push(DomSetModification::Remove(old_node));

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
            self.intersection_forest.transfer(old_node, new_node);
            self.scores[old_node as usize] = 1;
            self.sampler.add_entry(old_node, 0);
        } else {
            // (I4) Update sampler
            for &node in self.intersection_forest.get_root_nodes(old_node) {
                if node != old_node && node != new_node {
                    self.scores[node as usize] -= 1;
                    self.sampler.remove_entry(node);
                    if self.scores[node as usize] > 0 {
                        self.sampler
                            .add_entry(node, self.scores[node as usize] as usize - 1);
                    }
                }
            }
            self.scores[old_node as usize] = 0;
            self.intersection_forest.clear(old_node);
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
                    self.sampler.remove_entry(node);
                    if self.scores[node as usize] > 0 {
                        self.temp_nodes.push(node);
                    }
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
                    if self.scores[node as usize] == 1 {
                        self.temp_nodes.push(node);
                    }
                }
            }

            // Add all nodes for which an update occured to the sampler again
            for node in self.temp_nodes.drain(..) {
                self.sampler
                    .add_entry(node, self.scores[node as usize] as usize - 1);
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
    <G as ToSliceRepresentation>::SliceRepresentation: ReduceGraphNodes + SelfLoop + Default,
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
    <G as ToSliceRepresentation>::SliceRepresentation: ReduceGraphNodes + SelfLoop + Default,
{
}
