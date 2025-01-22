use rand::Rng;
use rand_distr::Distribution;

use crate::{
    graph::*,
    prelude::{IterativeAlgorithm, TerminatingIterativeAlgorithm},
    utils::{merge_tree::MergeTrees, sampler::WeightedPow2Sampler, DominatingSet},
};

use super::subsets::subset_reduction;

/// # GreedyReverseSearch
///
/// An iterative algorithm that samples from a set of candidates to replace some other node in the
/// DomSet. Sampling is biased to nodes that replace more nodes.
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

    /// The current solution
    current_solution: DominatingSet,
    /// Currently best known solution
    best_solution: DominatingSet,
    /// Has an optimal solution been found?
    is_optimal: bool,

    /// A sampler for sampling nodes with weights that are powers of 2.
    ///
    /// Contains only nodes u with scores[u] > 0.
    sampler: WeightedPow2Sampler<NUM_SAMPLER_BUCKETS>,
    /// RNG used for sampling
    rng: &'a mut R,

    /// Candidates are nodes for which we have to update datastructures at the end of a round
    candidates: Vec<Node>,
    /// Is a node currently a candidate?
    in_candidates: BitSet,
    /// Additional vector for temporarily storing nodes for which we need to update the datastructures
    temp_nodes: Vec<Node>,

    /// Number of incident dominating nodes
    num_covered: Vec<NumNodes>,
    /// Number of nodes this (dominating) nodes covers uniquely (no other dominating node covers)
    uniquely_covered: Vec<NumNodes>,

    /// Nodes that can possibly be removed from the DomSet
    redundant_nodes: Vec<Node>,
    /// A score for each node == Number of occurences in roots of MergeTrees
    scores: Vec<NumNodes>,

    /// Last time a node was added/removed from the DomSet
    age: Vec<u64>,
    /// Current iteration
    round: u64,

    /// Collection of MergeTrees:
    ///
    /// Every node u in the DomSet is assigned a MergeTree. Nodes that are uniquely covered by this
    /// u are then inserted into the MergeTree of u. MergeTree[u] thus stores all nodes in its root
    /// are incident to *all* uniquely covered nodes of u and can thus replace u in the DomSet.
    ///
    /// v in root of MergeTree[u] ==> scores[v] > 0 ==> v in sampler ==> v can be sampled to replace u
    ///
    /// Note that we only *really* consider neighbors that are not subset-dominated and thus can appear in any
    /// optimal DomSet without the possibility of directly replacing them.
    merge_trees: MergeTrees<<G as ToSliceRepresentation>::SliceRepresentation>,

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
    /// Creates a new instance of the algorithm
    ///
    /// Runs the subset-reduction beforehand
    pub fn new(graph: &'a mut G, mut initial_solution: DominatingSet, rng: &'a mut R) -> Self {
        // If only fixed nodes cover the graph, this is optimal.
        // For API-purposes, we create an *empty* instance that holds the optimal solution
        if initial_solution.all_fixed() {
            return Self {
                graph,
                current_solution: initial_solution.clone(),
                best_solution: initial_solution,
                is_optimal: true,
                sampler: WeightedPow2Sampler::new(0),
                rng,
                candidates: Vec::new(),
                in_candidates: BitSet::new(1),
                temp_nodes: Vec::new(),
                num_covered: Vec::new(),
                uniquely_covered: Vec::new(),
                redundant_nodes: Vec::new(),
                scores: Vec::new(),
                age: Vec::new(),
                merge_trees: MergeTrees::new(
                    <G as ToSliceRepresentation>::SliceRepresentation::default(),
                    true,
                ),
                round: 1,
                domset_modifications: Vec::new(),
            };
        }

        assert!(initial_solution.is_valid(graph));

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
        let mut uniquely_covered: Vec<NumNodes> = graph
            .vertices()
            .map(|u| {
                if !initial_solution.is_in_domset(u) {
                    return 0;
                }

                graph
                    .neighbors_of(u)
                    .map(|v| (num_covered[v as usize] <= 1) as NumNodes)
                    .sum()
            })
            .collect();

        // Remove redundant nodes from the DomSet and update datastructures
        for i in (0..initial_solution.len()).rev() {
            let u = initial_solution.ith_node(i);
            if uniquely_covered[u as usize] == 0 {
                initial_solution.remove_node(u);
                age[u as usize] = 0;

                for j in (0..graph.degree_of(u)).rev() {
                    let v = graph.ith_neighbor(u, j);
                    num_covered[v as usize] -= 1;
                    graph.swap_neighbors(
                        v,
                        graph.ith_cross_position(u, j),
                        num_covered[v as usize],
                    );

                    if num_covered[v as usize] == 1 {
                        uniquely_covered[graph.ith_neighbor(v, 0) as usize] += 1;
                    }
                }
            }
        }

        // Instantiate sampler and merge trees with reduced neighbor-set
        let mut sampler = WeightedPow2Sampler::new(n);
        let mut scores = vec![0; n];
        let mut merge_trees = MergeTrees::new(neighborhoods, true);

        // Insert uniquely covered neighbors of dominating nodes into MergeTrees & Sampler
        for u in initial_solution.iter_non_fixed() {
            for v in graph.neighbors_of(u) {
                if num_covered[v as usize] <= 1 {
                    merge_trees.add_entry(u, v);
                }
            }

            for v in merge_trees.get_root_nodes(u) {
                if u != *v {
                    scores[*v as usize] += 1;
                    sampler.update_entry(*v, scores[*v as usize] as usize - 1);
                }
            }
        }

        let current_solution = initial_solution.clone();
        let best_solution = initial_solution;

        Self {
            graph,
            current_solution,
            best_solution,
            is_optimal: false,
            sampler,
            rng,
            candidates: Vec::new(),
            in_candidates: BitSet::new(n as NumNodes),
            temp_nodes: Vec::new(),
            num_covered,
            uniquely_covered,
            redundant_nodes: Vec::new(),
            scores,
            age,
            merge_trees,
            round: 1,
            domset_modifications: Vec::new(),
        }
    }

    /// Sample a node from the sampler
    ///
    /// Returns *None* if the sampler is empty, ie there is no way to replace any node in the
    /// current DomSet.
    fn draw_node(&mut self) -> Option<Node> {
        if self.sampler.is_empty() {
            return None;
        }

        let mut best_node = 0;
        (0..NUM_SAMPLES).for_each(|_| {
            let node = self.sampler.sample(&mut self.rng);
            if self.scores[node as usize] > self.scores[best_node as usize]
                || (self.scores[node as usize] == self.scores[best_node as usize]
                    && self.age[node as usize] < self.age[best_node as usize])
            {
                best_node = node;
            }
        });

        Some(best_node)
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
            self.is_optimal = true;
            return;
        };

        self.round += 1;

        // Add node to DomSet
        self.current_solution.add_node(proposed_node);
        self.scores[proposed_node as usize] = 0;
        self.age[proposed_node as usize] = self.round;
        self.sampler.remove_entry(proposed_node);

        self.domset_modifications
            .push(DomSetModification::Add(proposed_node));

        // Update adjacency lists as well as num_covered/uniquely_covered
        //
        // If a previously uniquely covered node is now not longer uniquely covered,
        // add it to candidates as we must later update its MergeTree-Appearance
        for i in 0..self.graph.degree_of(proposed_node) {
            let neighbor = self.graph.ith_neighbor(proposed_node, i);
            self.graph.swap_neighbors(
                neighbor,
                self.graph.ith_cross_position(proposed_node, i),
                self.num_covered[neighbor as usize],
            );
            self.num_covered[neighbor as usize] += 1;

            if self.num_covered[neighbor as usize] == 2 {
                let former_unique_covering_node = self.graph.ith_neighbor(neighbor, 0);
                self.uniquely_covered[former_unique_covering_node as usize] -= 1;
                if !self.in_candidates.get_bit(neighbor) {
                    self.candidates.push(neighbor);
                    self.in_candidates.set_bit(neighbor);
                }

                if self.uniquely_covered[former_unique_covering_node as usize] == 0 {
                    self.redundant_nodes.push(former_unique_covering_node);
                }
            }
        }

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

        // Update MergeTrees/Sampler for all remaining Candidates = nodes that have to be
        // added/removed/updated in MergeTrees/Sampler
        for candidate in self.candidates.drain(..) {
            if !self.in_candidates.get_bit(candidate) {
                continue;
            }
            self.in_candidates.clear_bit(candidate);

            let dominating_node = self.graph.ith_neighbor(candidate, 0);
            if self.current_solution.is_fixed_node(dominating_node) {
                continue;
            }

            // Remove entries of MergeTree[dominating_node] from sampler
            for node in self.merge_trees.get_root_nodes(dominating_node) {
                if *node != dominating_node && self.scores[*node as usize] != 0 {
                    self.scores[*node as usize] -= 1;
                    self.sampler.remove_entry(*node);
                    if self.scores[*node as usize] > 0 {
                        self.temp_nodes.push(*node);
                    }
                }
            }

            // Update MergeTree[dominating_node]: this correctly resolves entries in MergeTree[dominating_node]
            if self.num_covered[candidate as usize] == 1 {
                self.merge_trees.add_entry(dominating_node, candidate);
            } else {
                self.merge_trees.remove_entry(dominating_node, candidate);
            }

            // Add all entries of MergeTree[dominating_node] to sampler (insert later)
            for node in self.merge_trees.get_root_nodes(dominating_node) {
                if *node != dominating_node {
                    self.scores[*node as usize] += 1;
                    if self.scores[*node as usize] == 1 {
                        self.temp_nodes.push(*node);
                    }
                }
            }

            // Add all nodes for which an update occured to the sampler again
            for node in self.temp_nodes.drain(..) {
                self.sampler
                    .add_entry(node, self.scores[node as usize] as usize - 1);
            }
        }

        // Update the best known solution if needed
        self.update_best_solution();
    }

    /// Removes a redundant node from the DomSet after inserting proposed_node
    ///
    /// MARKER marks whether this is the first redundant node that is considered (see comments in
    /// code).
    fn remove_redundant_node<const MARKER: bool>(&mut self, red_node: Node, proposed_node: Node) {
        if self.uniquely_covered[red_node as usize] > 0 {
            return;
        }

        self.current_solution.remove_node(red_node);
        self.age[red_node as usize] = self.round;

        self.domset_modifications
            .push(DomSetModification::Remove(red_node));

        for i in (0..self.graph.degree_of(red_node)).rev() {
            let neighbor = self.graph.ith_neighbor(red_node, i);
            self.num_covered[neighbor as usize] -= 1;
            self.graph.swap_neighbors(
                neighbor,
                self.graph.ith_cross_position(red_node, i),
                self.num_covered[neighbor as usize],
            );

            if self.num_covered[neighbor as usize] == 1 {
                let dominating_node = self.graph.ith_neighbor(neighbor, 0);
                self.uniquely_covered[dominating_node as usize] += 1;

                // Normally, we would have to leave neighbor in Candidates as we removed red_node
                // and need to add neighbor to MergeTree[proposed_node] later.
                // However, since we later copy/transfer MergeTree[red_node] to MergeTree[proposed_node]
                // in this iteration, we already have updated MergeTree[proposed_node] correctly
                // and do not need to consider it later again (except when a later red_node changes
                // this again).
                let prev_bit = self.in_candidates.get_bit(neighbor);
                self.in_candidates
                    .assign_bit(neighbor, !(MARKER && prev_bit));
                if !prev_bit {
                    self.candidates.push(neighbor);
                }
            }
        }

        // Copy MergeTree in the first iteration as it should be a superset of the intended one for
        // proposed_node. This will later be further updated and corrected.
        if MARKER {
            self.merge_trees.transfer(red_node, proposed_node);
            self.scores[red_node as usize] = 1;
            self.sampler.add_entry(red_node, 0);
        } else {
            // Update sampler
            for node in self.merge_trees.get_root_nodes(red_node) {
                if *node != red_node && *node != proposed_node {
                    self.scores[*node as usize] -= 1;
                    self.sampler.remove_entry(*node);
                    if self.scores[*node as usize] > 0 {
                        self.sampler
                            .add_entry(*node, self.scores[*node as usize] as usize - 1);
                    }
                }
            }
            self.scores[red_node as usize] = 0;
            self.merge_trees.clear(red_node);
        }
    }

    /// Updates the best_solution to current_solution if better
    fn update_best_solution(&mut self) {
        if self.current_solution.len() < self.best_solution.len() {
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
        self.is_optimal
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
