use std::{cell::RefCell, mem::take, rc::Rc};

use itertools::Itertools;
#[allow(unused_imports)]
use log::{info, trace};

use crate::{kernelization::KernelRules, prelude::*};

type SolverResultCache = crate::utils::ResultCache<digest::Output<sha2::Sha256>>;
type BestSolution = (Option<(Node, Node)>, NumNodes, ContractionSequence);
type OptSolution = Option<(NumNodes, ContractionSequence)>;

#[derive(Clone, Copy, Debug)]
pub struct FeatureConfiguration {
    pub try_split_ccs: bool,
    pub try_complement: bool,
    pub try_remove_isolated: bool,
    pub red_bridges: bool,
    pub kernelize: bool,
    pub use_cache: bool,
    pub mergable_heuristic: bool,
    pub test_bipartite: bool,
    pub atmost_distance_three: bool,
    pub kernel_rules: KernelRules,
    pub paranoid: bool,
}

impl Default for FeatureConfiguration {
    fn default() -> Self {
        Self {
            try_split_ccs: true,
            try_complement: true,
            try_remove_isolated: false,
            red_bridges: false,
            kernelize: true,
            use_cache: true,
            mergable_heuristic: false,
            test_bipartite: false,
            paranoid: false,
            atmost_distance_three: true,
            kernel_rules: Default::default(),
        }
    }
}

impl FeatureConfiguration {
    pub fn pessimitic() -> Self {
        Self {
            try_split_ccs: false,
            try_complement: false,
            try_remove_isolated: false,
            red_bridges: false,
            kernelize: false,
            use_cache: true,
            mergable_heuristic: false,
            test_bipartite: false,
            paranoid: true,
            atmost_distance_three: false,
            kernel_rules: Default::default(),
        }
    }
}

pub struct BranchAndBound<G: FullfledgedGraph> {
    cache: Rc<RefCell<SolverResultCache>>,
    graph: G,
    sequence: ContractionSequence,
    protected: BitSet,
    slack: NumNodes,
    not_above: NumNodes,
    know_connected: bool,
    features: FeatureConfiguration,
}

macro_rules! return_if_some {
    ($e : expr) => {
        if let Some(x) = $e {
            return x;
        }
    };
}

macro_rules! return_none_if {
    ($e : expr) => {
        if $e {
            return None;
        }
    };
}

impl<G: FullfledgedGraph> BranchAndBound<G> {
    pub fn new(graph: G) -> Self {
        let n = graph.number_of_nodes();
        Self::new_with_bounds(graph, 0, n)
    }

    pub fn new_with_bounds(graph: G, slack: NumNodes, not_above: NumNodes) -> Self {
        let cache = Rc::new(RefCell::new(SolverResultCache::default()));
        let sequence = ContractionSequence::new(graph.number_of_nodes());
        let protected = BitSet::new(graph.number_of_nodes());
        Self {
            cache,
            graph,
            sequence,
            protected,
            slack,
            not_above,
            know_connected: false,
            features: Default::default(),
        }
    }

    pub fn configure_features(&mut self, features: FeatureConfiguration) {
        self.features = features;
    }

    fn fork(&self, graph: G) -> Self {
        self.fork_with_bounds(graph, self.slack, self.not_above)
    }

    fn fork_with_bounds(&self, graph: G, slack: NumNodes, not_above: NumNodes) -> Self {
        let sequence = ContractionSequence::new(graph.number_of_nodes());
        Self {
            cache: self.cache.clone(),
            graph,
            sequence,
            protected: self.protected.clone(),
            not_above,
            slack,
            know_connected: self.know_connected,
            features: self.features,
        }
    }

    pub fn solve(&mut self) -> Option<(NumNodes, ContractionSequence)> {
        trace!(
            "Recurse n={:>5} m={:>5} slack={:>5} not_above={:>5}",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self.slack,
            self.not_above,
        );

        let edges_on_protected = self
            .protected
            .iter()
            .map(|u| self.graph.degree_of(u) as NumEdges)
            .sum::<NumEdges>();

        if self.graph.number_of_nodes() == 0 || self.graph.number_of_edges() == edges_on_protected {
            return Some((0, take(&mut self.sequence)));
        }

        return_if_some!(self.try_split_into_ccs());
        return_if_some!(self.try_remove_singletons());
        return_if_some!(self.try_complement());

        let org_graph = self.features.paranoid.then(|| self.graph.clone());

        let result = if self.features.use_cache && self.protected.cardinality() == 0 {
            self.try_cache_first()
        } else {
            self.solver_impl()
        };

        if let Some(org_graph) = org_graph {
            if let Some((tww, seq)) = result.as_ref() {
                let comp_tww = seq.compute_twin_width(org_graph).unwrap();
                assert!(*tww >= comp_tww);
            }
        }

        result
    }

    fn try_complement(&mut self) -> Option<OptSolution> {
        return_none_if!(!self.features.try_complement);
        assert!(self.sequence.is_empty());

        return_none_if!(self.graph.number_of_nodes() < 8);
        return_none_if!(
            self.graph.number_of_edges() < 4 * (self.graph.number_of_nodes() as NumEdges)
        );

        let edges_in_complement = self.graph.number_of_edges_in_trigraph_complement(true);
        return_none_if!(edges_in_complement > self.graph.number_of_edges());

        let complement = self.graph.trigraph_complement(true);
        assert!(complement.number_of_edges() < self.graph.number_of_edges());

        Some(self.fork(complement.clone()).solve().map(|(tww, mut seq)| {
            seq.add_unmerged_singletons(&complement).unwrap();
            (tww, seq)
        }))
    }

    fn try_remove_singletons(&mut self) -> Option<OptSolution> {
        return_none_if!(!self.features.try_remove_isolated);
        return_none_if!(
            self.graph
                .number_of_nodes_with_neighbors()
                .next_power_of_two()
                == self.graph.number_of_nodes().next_power_of_two()
        );

        assert!(self.sequence.is_empty());

        let (graph, mapper) = self.graph.remove_disconnected_verts();

        let protected = BitSet::new_all_unset_but(
            graph.number_of_nodes(),
            mapper.get_filtered_new_ids(self.protected.iter()),
        );

        let mut child = self.fork(graph);
        child.protected = protected;
        let (tww, mut seq) = child.solve()?;

        seq.apply_mapper(&mapper);
        Some(Some((tww, seq)))
    }

    fn try_cache_first(&mut self) -> Option<(u32, ContractionSequence)> {
        assert!(self.sequence.is_empty());

        let hash = self.graph.binary_digest_sha256();

        if let Some(solution) = self
            .cache
            .borrow_mut()
            .get(&hash, self.slack, self.not_above)
        {
            return solution.cloned();
        }

        let result = self.solver_impl();

        if self.protected.cardinality() == 0 {
            self.cache
                .borrow_mut()
                .add_to_cache(hash, result.clone(), self.slack, self.not_above);
        }
        result
    }

    fn try_split_into_ccs(&mut self) -> Option<OptSolution> {
        return_none_if!(!self.features.try_split_ccs);
        if self.know_connected {
            return None;
        }

        let part = self.graph.partition_into_connected_components(true);
        if part.number_of_classes() == 1 && part.number_of_unassigned() == 0 {
            self.know_connected = true;
            return None;
        }

        let mut max_tww = 0;

        for (subgraph, mapper) in part.split_into_subgraphs(&self.graph) {
            let protected = BitSet::new_all_unset_but(
                subgraph.number_of_nodes(),
                mapper.get_filtered_new_ids(self.protected.iter()),
            );

            let mut child = self.fork(subgraph);
            child.protected = protected;
            child.know_connected = true;

            if let Some((size, sol)) = child.solve() {
                self.slack = self.slack.max(size);
                max_tww = max_tww.max(size);
                self.sequence.append_mapped(&sol, &mapper);
            } else {
                return Some(None);
            }
        }

        Some(Some((max_tww, take(&mut self.sequence))))
    }

    fn solver_impl(&mut self) -> Option<(NumNodes, ContractionSequence)> {
        return_none_if!(self.slack > self.not_above);

        if self.graph.number_of_edges() == 0 {
            return Some((0, ContractionSequence::new(0)));
        }

        self.kernelize();

        let edges_on_protected = self
            .protected
            .iter()
            .map(|u| self.graph.degree_of(u) as NumEdges)
            .sum::<NumEdges>();

        if self.graph.number_of_edges() == edges_on_protected {
            trace!("Left with empty kernel: {:?}", self.sequence.merges());
            return Some((self.slack, take(&mut self.sequence)));
        }

        return_if_some!(self.try_red_bridges());

        assert!(self.graph.red_degrees().max().unwrap() <= self.not_above);

        let mergable = self.compute_mergable_nodes();

        let pairs = self.contraction_candidates(&mergable);

        trace!(
            " mergeable: {:>5} pairs: {:>8} | {pairs:?} {:?}",
            mergable.cardinality(),
            pairs.len(),
            self.graph
        );

        if self.graph.number_of_edges() == 0 {
            // contract_candidates may prune, so we've to check again
            return Some((self.slack, take(&mut self.sequence)));
        }

        if pairs.is_empty() {
            return None;
        };

        let mut best_solution: Option<BestSolution> = None;
        'outer: for &(r, (u, v)) in &pairs {
            if self.protected[u] || self.protected[v] {
                continue;
            }

            assert!(self.graph.degree_of(u) > 0);
            if self.graph.degree_of(v) == 0 {
                continue;
            }

            assert_ne!(u, v);

            if r > self.not_above {
                break;
            }

            if !(mergable[u] && mergable[v]) {
                continue;
            }

            let mut local_graph = self.graph.clone();
            local_graph.merge_node_into(u, v);

            let max_red_degree = local_graph.red_degrees().max().unwrap();
            if max_red_degree > self.not_above {
                continue;
            }

            if let Some((sol_size, seq)) = self
                .fork_with_bounds(local_graph, self.slack.max(max_red_degree), self.not_above)
                .solve()
            {
                let sol_size = sol_size.max(max_red_degree);
                assert!(sol_size <= self.not_above);
                best_solution = Some((Some((u, v)), sol_size, seq));

                if sol_size <= self.slack {
                    break 'outer;
                }

                self.not_above = sol_size.checked_sub(1).unwrap();
            }
        }
        let (pair, tww, seq) = best_solution?;

        if let Some((u, v)) = pair {
            self.sequence.merge_node_into(u, v);
        }
        self.sequence.append(&seq);
        Some((tww, take(&mut self.sequence)))
    }

    fn compute_mergable_nodes(&self) -> BitSet {
        let mut mergable = BitSet::new(self.graph.number_of_nodes());

        if !self.features.mergable_heuristic || self.graph.degrees().all(|d| d == 0 || d == 2) {
            mergable.set_all();
        } else {
            for u in self.graph.vertices() {
                if self.graph.degree_of(u) == 2 && self.graph.red_degree_of(u) == 0 {
                    continue;
                }
                mergable.set_bit(u);
                mergable.set_bits(self.graph.neighbors_of(u));
            }
        }

        mergable.and_not(&self.protected);

        mergable
    }

    fn kernelize(&mut self) {
        if !self.features.kernelize {
            return;
        }

        let mut kernel = Kernelization::new_with_protected(
            &mut self.graph,
            &mut self.sequence,
            self.slack,
            self.protected.clone(),
        );
        kernel.configure_rules(self.features.kernel_rules);

        kernel.run_recursion_defaults();
        self.slack = self.slack.max(kernel.slack());
    }

    fn try_red_bridges(&mut self) -> Option<OptSolution> {
        return_none_if!(!self.features.red_bridges);

        let red_bridges: Vec<_> = self
            .graph
            .compute_colored_bridges()
            .into_iter()
            .filter(|&ColoredEdge(u, v, c)| {
                c.is_red()
                    && self.graph.degree_of(u) > 1
                    && self.graph.degree_of(v) > 1
                    && !self.protected[u]
                    && !self.protected[v]
            })
            .collect();

        for ColoredEdge(u, v, _) in red_bridges {
            self.graph.remove_edge(u, v);
            let part = self.graph.partition_into_connected_components(true);
            self.graph.add_edge(u, v, EdgeColor::Red);

            if part.number_of_classes() != 2 {
                println!("Num Classes: {}", part.number_of_classes());
                continue;
            }

            let (small_node, large_node) = if part.number_in_class(part.class_of_node(u).unwrap())
                <= part.number_in_class(part.class_of_node(v).unwrap())
            {
                (u, v)
            } else {
                (v, u)
            };

            let small_class = part.class_of_node(small_node).unwrap();
            let _large_class = part.class_of_node(large_node).unwrap();

            if part.number_in_class(small_class) < 3 {
                continue;
            }

            let extract_subgraph = |class_idx, other_node| -> G {
                let mut nodes = BitSet::new_all_unset_but(
                    self.graph.number_of_nodes(),
                    part.members_of_class(class_idx),
                );
                nodes.set_bit(other_node);
                self.graph.sub_graph(&nodes)
            };

            let small_graph = extract_subgraph(small_class, large_node);
            let mut small_protected = self.protected.clone();
            small_protected.set_bit(large_node);

            let mut child = self.fork(small_graph.clone());
            child.protected = small_protected.clone();
            let sol = child.solve();

            if sol.is_none() {
                continue;
            }

            let (tww_with_prot, seq_with_prot) = sol.unwrap();

            assert!(seq_with_prot
                .merges()
                .iter()
                .all(|&(u, v)| !small_protected[u] && !small_protected[v]));

            let mut child = self.fork_with_bounds(
                small_graph.clone(),
                self.slack,
                tww_with_prot.saturating_sub(1),
            );
            child.protected = small_protected.clone();
            let sol1 = child.solve();

            if tww_with_prot > self.slack && sol1.is_some() {
                continue;
            }

            assert!(!seq_with_prot.is_empty());

            let mut large_graph = self.graph.clone();
            for &(u, v) in seq_with_prot.merges() {
                large_graph.merge_node_into(u, v);
            }

            let (tww, seq) = self.fork(large_graph).solve()?;

            self.sequence.append(&seq_with_prot);
            self.sequence.append(&seq);

            return Some(Some((
                tww_with_prot.max(tww).max(self.slack),
                take(&mut self.sequence),
            )));
        }

        None
    }

    fn contraction_candidates(&mut self, org_mergeable: &BitSet) -> Vec<(u32, (u32, u32))> {
        let mut pairs = Vec::new();
        let mut mergeable = org_mergeable.clone();

        let red_degs_of_black_neighbors = self
            .graph
            .vertices()
            .map(|u| {
                if !mergeable[u] {
                    return self.not_above + 1;
                }

                self.graph
                    .black_neighbors_of(u)
                    .map(|v| self.graph.red_degree_of(v) + 1)
                    .max()
                    .unwrap_or(0)
            })
            .collect_vec();

        let is_bipartite = self.features.test_bipartite && self.graph.is_bipartite();
        for u in self.graph.vertices_range() {
            if !mergeable.unset_bit(u) {
                continue;
            }
            let degree_u = self.graph.degree_of(u);
            if degree_u == 0 {
                continue;
            }

            let org_two_neighbors = self.graph.closed_two_neighborhood_of(u);
            let mut two_neighbors = org_two_neighbors.clone();
            two_neighbors.and(&mergeable);

            if is_bipartite && self.graph.degree_of(u) > 1 {
                for x in self.graph.neighbors_of(u) {
                    if self.graph.degree_of(x) > 2 {
                        two_neighbors.unset_bit(x);
                    }
                }
            }

            for v in two_neighbors.iter() {
                assert!(v > u);
                let mut red_neighs = self.graph.red_neighbors_after_merge(u, v, false);
                let mut red_card = red_neighs.cardinality();

                //trace!("{u} {v} {red_card} {:?}", red_neighs.iter().collect_vec());

                if red_neighs.cardinality() == 0 {
                    // this should not happend in practice, but is a short-cut if
                    // the kernelization feature is disabled
                    trace!("Forcefully merging {u} into {v}");
                    self.sequence.merge_node_into(v, u);
                    self.graph.merge_node_into(v, u);
                    assert_eq!(self.graph.red_degree_of(v), 0);
                    return self.contraction_candidates(org_mergeable);
                }

                if red_card > self.not_above {
                    continue;
                }

                red_neighs.unset_bits(self.graph.red_neighbors_of(u));
                red_neighs.unset_bits(self.graph.red_neighbors_of(v));

                for new_red in red_neighs.iter() {
                    red_card = red_card.max(self.graph.red_degree_of(new_red) + 1);
                }

                if red_card <= self.not_above {
                    pairs.push((red_card, (u, v)));
                }
            }

            if degree_u > self.not_above {
                continue;
            }

            let distant_nodes = {
                let mut dist_nodes = BitSet::new(self.graph.number_of_nodes());
                if self.features.atmost_distance_three {
                    //let mut two_neighbors = org_two_neighbors.clone();
                    //two_neighbors.unset_bits(self.graph.neighbors_of(u).iter().copied());

                    let three = self.graph.closed_three_neighborhood_of(u);
                    dist_nodes.or(&three);
                    for x in three.iter() {
                        dist_nodes.set_bits(self.graph.neighbors_of(x));
                    }
                } else {
                    dist_nodes.set_all();
                }
                dist_nodes.and_not(&two_neighbors);
                dist_nodes.and(&mergeable);
                dist_nodes
            };

            let red_deg_of_black_neighbors = self
                .graph
                .black_neighbors_of(u)
                .map(|v| self.graph.red_degree_of(v) + 1)
                .max()
                .unwrap_or(0);

            if red_degs_of_black_neighbors[u as usize] > self.not_above {
                continue;
            }

            for v in distant_nodes.iter() {
                assert!(v > u);
                let degree_v = self.graph.degree_of(v);
                let red_degree = red_deg_of_black_neighbors.max(degree_u + degree_v);
                if degree_v > 0
                    && red_degree <= self.not_above
                    && (!is_bipartite || degree_v < 3 || degree_u < 3)
                    && self
                        .graph
                        .black_neighbors_of(v)
                        .all(|w| self.graph.red_degree_of(w) < self.not_above)
                {
                    pairs.push((red_degree, (u, v)));
                }
            }
        }
        pairs.sort();
        pairs
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[allow(unused_imports)]
    use crate::{log::build_pace_logger_for_level, testing::*};
    #[allow(unused_imports)]
    use log::LevelFilter;
    use paste::paste;
    #[allow(unused_imports)]
    use rayon::prelude::*;
    use std::{fs::File, io::BufReader};

    #[test]
    fn tiny() {
        for (i, tww) in [1, 2, 0, 0, 3, 0, 2, 4, 1, 2].into_iter().enumerate() {
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

            let (size, _sol) = BranchAndBound::new(graph).solve().unwrap();
            assert_eq!(size, tww, "file: {filename}");
        }
    }

    #[test]
    fn small_random() {
        //build_pace_logger_for_level(LevelFilter::Trace);

        get_test_graphs_with_tww("instances/small-random/*38d4.gr").for_each(
            |(filename, graph, presolved_tww)| {
                let (tww, mut seq) = BranchAndBound::new(graph.clone()).solve().unwrap();
                seq.add_unmerged_singletons(&graph).unwrap();
                let verified = seq.compute_twin_width(graph.clone());
                assert_eq!(
                    Some(tww),
                    verified,
                    "does not match computed tww. file: {filename}"
                );
                assert_eq!(
                    tww, presolved_tww,
                    "does not match presolved. file: {filename}"
                );
                println!("Done with file: {filename}");
            },
        );
    }

    macro_rules! impl_test_feature {
        ($feature:ident) => {
            paste! {
                #[test]
                fn [< non_default_setting_for_$feature >]() {
                    for (filename, graph, presolved_tww) in
                        get_test_graphs_with_tww("instances/small-random/*.gr").step_by(3)
                    {
                        if graph.number_of_nodes() > 15 {
                            continue;
                        }

                        let mut algo = BranchAndBound::new(graph);
                        algo.features.$feature = !algo.features.$feature;

                        //println!(" Test {filename}");
                        let (tww, _seq) = algo.solve().unwrap();
                        assert_eq!(tww, presolved_tww, "file: {filename}");
                    }
                }
            }
        };
    }

    impl_test_feature!(try_split_ccs);
    impl_test_feature!(try_complement);
    impl_test_feature!(try_remove_isolated);
    impl_test_feature!(red_bridges);
    impl_test_feature!(use_cache);
    impl_test_feature!(mergable_heuristic);
    impl_test_feature!(test_bipartite);
}
