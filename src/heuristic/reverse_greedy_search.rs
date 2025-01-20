use rand::Rng;
use rand_distr::Distribution;

use crate::{
    graph::*,
    prelude::{IterativeAlgorithm, TerminatingIterativeAlgorithm},
    utils::{merge_tree::MergeTreeVec, sampler::WeightedPow2Sampler, ExtDominatingSet},
};

use super::subsets::subset_reduction;

pub struct GreedyReverseSearch<
    'a,
    R: Rng,
    G: StaticGraph + SelfLoop,
    const NUM_SAMPLER_BUCKETS: usize = 8,
    const NUM_SAMPLES: usize = 10,
> {
    graph: &'a mut G,

    current_solution: ExtDominatingSet,
    best_solution: ExtDominatingSet,
    is_optimal: bool,

    sampler: WeightedPow2Sampler<NUM_SAMPLER_BUCKETS>,
    rng: &'a mut R,

    candidates: Vec<Node>,
    in_candidates: BitSet,
    temp_nodes: Vec<Node>,

    num_covered: Vec<NumNodes>,
    uniquely_covered: Vec<NumNodes>,

    redundant_nodes: Vec<Node>,
    scores: Vec<NumNodes>,

    age: Vec<u64>,
    round: u64,

    merge_trees: MergeTreeVec,

    domset_modifications: Vec<DomSetModification>,
}

impl<
        'a,
        R: Rng,
        G: StaticGraph + SelfLoop,
        const NUM_SAMPLER_BUCKETS: usize,
        const NUM_SAMPLES: usize,
    > GreedyReverseSearch<'a, R, G, NUM_SAMPLER_BUCKETS, NUM_SAMPLES>
{
    pub fn new(graph: &'a mut G, mut initial_solution: ExtDominatingSet, rng: &'a mut R) -> Self {
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
                merge_trees: MergeTreeVec::new(Vec::new(), vec![0]),
                round: 1,
                domset_modifications: Vec::new(),
            };
        }

        assert!(initial_solution.is_valid(graph));

        let n = graph.number_of_nodes() as usize;

        let (reduced_edges, reduced_offsets) = subset_reduction(graph, &mut initial_solution);

        let mut num_covered = vec![0; n];
        let mut age = vec![0; n];

        // Reorder adjacency lists such that dominating nodes appear first
        for u in initial_solution.iter() {
            age[*u as usize] = 1;
            for i in 0..graph.degree_of(*u) {
                let v = graph.ith_neighbor(*u, i);
                graph.swap_neighbors(v, graph.ith_cross_position(*u, i), num_covered[v as usize]);
                num_covered[v as usize] += 1;
            }
        }

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
                initial_solution.remove(u);
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

        let mut sampler = WeightedPow2Sampler::new(n);

        let mut scores = vec![0; n];
        let mut merge_trees = MergeTreeVec::new(reduced_edges, reduced_offsets);

        // Insert uniquely covered neighbors of dominating nodes into MergeTrees & Sampler
        for u in initial_solution.iter_non_fixed() {
            for v in graph.neighbors_of(*u) {
                if num_covered[v as usize] <= 1 {
                    merge_trees.add_entry(*u, v);
                }
            }

            for v in merge_trees.get_root_nodes(*u) {
                if u != v {
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

    pub fn step(&mut self) {
        let proposed_node = if let Some(node) = self.draw_node() {
            node
        } else {
            self.is_optimal = true;
            return;
        };

        self.round += 1;

        // Add node to DomSet
        self.current_solution.push(proposed_node);
        self.scores[proposed_node as usize] = 0;
        self.age[proposed_node as usize] = self.round;
        self.sampler.remove_entry(proposed_node);

        self.domset_modifications
            .push(DomSetModification::Add(proposed_node));

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

        self.redundant_nodes.sort_by_key(|u| self.age[*u as usize]);

        // Remove redundant nodes from DomSet
        if !self.redundant_nodes.is_empty() {
            self.remove_redundant_node::<true>(self.redundant_nodes[0], proposed_node);
            for i in 1..self.redundant_nodes.len() {
                self.remove_redundant_node::<false>(self.redundant_nodes[i], proposed_node);
            }
            self.redundant_nodes.clear();
        }

        for candidate in self.candidates.drain(..) {
            if !self.in_candidates.get_bit(candidate) {
                continue;
            }
            self.in_candidates.clear_bit(candidate);

            let dominating_node = self.graph.ith_neighbor(candidate, 0);
            if self.current_solution.is_fixed_node(dominating_node) {
                continue;
            }

            for node in self.merge_trees.get_root_nodes(dominating_node) {
                if *node != dominating_node && self.scores[*node as usize] != 0 {
                    self.scores[*node as usize] -= 1;
                    self.sampler.remove_entry(*node);
                    if self.scores[*node as usize] > 0 {
                        self.temp_nodes.push(*node);
                    }
                }
            }

            if self.num_covered[candidate as usize] == 1 {
                self.merge_trees.add_entry(dominating_node, candidate);
            } else {
                self.merge_trees.remove_entry(dominating_node, candidate);
            }

            for node in self.merge_trees.get_root_nodes(dominating_node) {
                if *node != dominating_node {
                    self.scores[*node as usize] += 1;
                    if self.scores[*node as usize] == 1 {
                        self.temp_nodes.push(*node);
                    }
                }
            }

            for node in self.temp_nodes.drain(..) {
                self.sampler
                    .add_entry(node, self.scores[node as usize] as usize - 1);
            }
        }

        self.update_best_solution();
    }

    fn remove_redundant_node<const MARKER: bool>(&mut self, red_node: Node, proposed_node: Node) {
        if self.uniquely_covered[red_node as usize] > 0 {
            return;
        }

        self.current_solution.remove(red_node);
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

                // If neighbor was a previous candidate, remove it in the first iteration as it was
                // possible only covered by the first red_node. If not, later red_nodes will
                // correct this and re-push neighbor into candidates.
                let prev_bit = self.in_candidates.get_bit(neighbor);
                self.in_candidates
                    .assign_bit(neighbor, !(MARKER && prev_bit));
                if !prev_bit {
                    self.candidates.push(neighbor);
                }
            }
        }

        // Copy MergeTree in the first iteration as it should be a superset of the intended one for
        // proposed_node. this will later be updated and corrected.
        if MARKER {
            self.merge_trees.transfer(red_node, proposed_node);
            self.scores[red_node as usize] = 1;
            self.sampler.add_entry(red_node, 0);
        } else {
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
        }

        self.merge_trees.clear(red_node);
    }

    fn update_best_solution(&mut self) {
        if self.current_solution.len() < self.best_solution.len() {
            if self.domset_modifications.len() > self.graph.number_of_nodes() as usize / 64 {
                self.best_solution = self.current_solution.clone();
                self.domset_modifications.clear();
            } else {
                for modification in self.domset_modifications.drain(..) {
                    match modification {
                        DomSetModification::Add(node) => self.best_solution.push(node),
                        DomSetModification::Remove(node) => self.best_solution.remove(node),
                    }
                }
            }
        }
    }
}

enum DomSetModification {
    Add(Node),
    Remove(Node),
}

impl<
        R: Rng,
        G: StaticGraph + SelfLoop,
        const NUM_SAMPLER_BUCKETS: usize,
        const NUM_SAMPLES: usize,
    > IterativeAlgorithm<ExtDominatingSet>
    for GreedyReverseSearch<'_, R, G, NUM_SAMPLER_BUCKETS, NUM_SAMPLES>
{
    fn execute_step(&mut self) {
        self.step();
    }

    fn is_completed(&self) -> bool {
        self.is_optimal
    }

    fn best_known_solution(&mut self) -> Option<ExtDominatingSet> {
        Some(self.best_solution.clone())
    }
}

impl<
        R: Rng,
        G: StaticGraph + SelfLoop,
        const NUM_SAMPLER_BUCKETS: usize,
        const NUM_SAMPLES: usize,
    > TerminatingIterativeAlgorithm<ExtDominatingSet>
    for GreedyReverseSearch<'_, R, G, NUM_SAMPLER_BUCKETS, NUM_SAMPLES>
{
}
