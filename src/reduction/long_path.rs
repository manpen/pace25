use super::*;
use crate::graph::*;
use itertools::Itertools;
use smallvec::SmallVec;
use std::{borrow::Borrow, marker::PhantomData};

#[allow(unused_imports)]
use log::{debug, info};

type SmallNodeBuffer = SmallVec<[Node; 8]>;

pub struct LongPathReduction {
    in_post: BitSet,
}

#[must_use] // this rule has a post-processing step and may cause invalid results if not applied
pub struct LongPathPostProcessor<G> {
    removed_paths: Vec<Vec<Node>>,
    _graph: PhantomData<G>,
}

impl LongPathReduction {
    pub fn new(n: NumNodes) -> Self {
        Self {
            in_post: BitSet::new(n),
        }
    }
}

/// This reduction rule shortens paths of lengths at least 5 by removing groups of three nodes
/// until as long as at least 2 nodes remain. It is a two-staged rule that requires post-processing
/// once a domset for the modified graph was computed.
impl<G: AdjacencyList + AdjacencyTest + GraphEdgeEditing + 'static> ReductionRule<G>
    for LongPathReduction
{
    const NAME: &str = "LongPathReduction";

    fn apply_rule(
        &mut self,
        graph: &mut G,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<G>>>) {
        let long_paths = graph.path_iter_with_atleast_path_nodes(3).collect_vec();
        if long_paths.is_empty() {
            return (false, None);
        }

        let mut post_process_paths = Vec::new();

        let mut num_cycle = 0;
        let mut num_path = 0;
        let mut num_path_with_pp = 0;
        for mut path in long_paths {
            // If two adjacent nodes on the path are covered, the edge between them can be deleted
            // and we do not have a proper path any more. Skip and let the reducer split the path.
            if path
                .iter()
                .map(|&u| covered.get_bit(u))
                .tuple_windows()
                .any(|(x, y)| x && y)
            {
                continue;
            }

            // Same for redundant path
            if path
                .iter()
                .map(|&u| never_select.get_bit(u))
                .tuple_windows()
                .any(|(x, y)| x && y)
            {
                continue;
            }

            let mut rule_impl = RuleImpl {
                graph,
                domset,
                covered,
                never_select,
            };

            if rule_impl.process_circle(&path) {
                num_cycle += 1;
                continue;
            }

            if rule_impl.process_path_without_postprocess(&mut path) {
                num_path += 1;
                continue;
            }

            if rule_impl.process_path_with_postprocess(&mut path, &mut self.in_post) {
                num_path_with_pp += 1;
                post_process_paths.push(path);
            }
        }

        let modified = (num_cycle + num_path + num_path_with_pp) > 0;

        if modified {
            debug!(
                "{} Cycle: {num_cycle:4} Path (w/o pp): {num_path:4} Path (w pp): {num_path_with_pp:4}",
                <Self as ReductionRule<G>>::NAME
            );
        }

        (
            modified,
            (!post_process_paths.is_empty()).then(|| -> Box<dyn Postprocessor<G>> {
                Box::new(LongPathPostProcessor {
                    removed_paths: post_process_paths,
                    _graph: Default::default(),
                })
            }),
        )
    }
}

struct RuleImpl<'a, G> {
    graph: &'a mut G,
    domset: &'a mut DominatingSet,
    covered: &'a mut BitSet,
    never_select: &'a BitSet,
}

impl<G: AdjacencyList + GraphEdgeEditing + AdjacencyTest> RuleImpl<'_, G> {
    fn add_to_solution(&mut self, u: Node) {
        assert!(!self.never_select.get_bit(u));
        self.covered.set_bits(self.graph.closed_neighbors_of(u));
        self.domset.add_node(u);
    }

    fn greedy_cover_into<T: Borrow<Node>>(
        &self,
        path: impl Iterator<Item = T>,
        mut is_covered: bool,
    ) -> Option<(SmallNodeBuffer, bool)> {
        let mut result: SmallNodeBuffer = Default::default();
        let mut needs_covering = None;
        let mut previous_element = None;
        for u in path {
            let u = *u.borrow();
            if needs_covering.is_some() {
                if self.never_select.get_bit(u) {
                    result.push(previous_element.unwrap());
                } else {
                    result.push(u);
                    is_covered = true;
                }
                needs_covering = None;
            } else if is_covered {
                is_covered = false;
            } else if !self.covered.get_bit(u) {
                needs_covering = Some((previous_element, u));
            }
            previous_element = Some(u);
        }
        if let Some((prev_prev, prev)) = needs_covering {
            if !self.never_select.get_bit(prev) {
                result.push(prev);
            } else {
                result.push(prev_prev?);
            }
        }

        let incl_last = result.last() == previous_element.as_ref();
        Some((result, incl_last))
    }

    fn process_circle(
        &mut self,
        path: &[Node], // first and last are identical and may have degree > 2
    ) -> bool {
        if path.first() != path.last() {
            return false;
        }

        let len = path.len();
        let mut best_result: Option<(SmallNodeBuffer, bool)> = None;

        macro_rules! use_if_better {
            ($path : expr, $covers_outer : expr) => {{
                let tmp = self.greedy_cover_into($path, $covers_outer);
                if let Some((tmp, mut outer)) = tmp {
                    outer |= $covers_outer;
                    if best_result.as_ref().is_none_or(|(b, co)| {
                        b.len() > tmp.len() || (b.len() == tmp.len() && !co && outer)
                    }) {
                        best_result = Some((tmp, outer));
                    }
                }
            }};
        }

        // first assume that the center is put into the solution
        if !self.never_select.get_bit(path[0]) {
            use_if_better!(path[1..len - 2].iter(), true);
            use_if_better!(path[2..len - 1].iter().rev(), true);
            if let Some(res) = best_result.as_mut() {
                res.0.push(path[0]);
            }
        }

        // then make the center optional
        use_if_better!(path[1..len - 1].iter(), false);
        use_if_better!(path[1..len - 1].iter().rev(), false);

        for u in best_result.unwrap().0 {
            self.add_to_solution(u);
        }

        true
    }

    fn process_leaf_path(
        &mut self,
        path: &[Node], // first and last are identical and may have degree > 2
    ) -> bool {
        debug_assert_eq!(self.graph.degree_of(path[0]), 1);

        // option 1 requires all nodes to be covered
        let opt1 = self.greedy_cover_into(path.iter(), false).unwrap().0;

        // option 2 allows the last node to remain uncovered (thereby leaving to up to the solver covered it)
        // this makes only sense if the node not a leaf itself
        let nodes_to_add = if self.graph.degree_of(*path.last().unwrap()) > 1 {
            let opt2 = self.greedy_cover_into(path.iter().rev(), true).unwrap().0;
            if opt1.len() > opt2.len() { opt2 } else { opt1 }
        } else {
            opt1
        };

        for u in nodes_to_add {
            self.add_to_solution(u);
        }

        true
    }

    fn process_path_with_uncovered_triples(
        &mut self,
        path: &[Node], // first and last are identical and may have degree > 2
    ) -> bool {
        let mut modified = false;
        let mut i = 0;
        'outer: while i + 4 < path.len() {
            if !self.covered.get_bit(path[i + 4]) {
                i += 5;
                continue;
            }

            for j in [3, 2, 1] {
                if self.covered.get_bit(path[i + j]) {
                    i += j;
                    continue 'outer;
                }
            }

            if self.covered.get_bit(path[i]) && !self.never_select.get_bit(path[i + 2]) {
                self.add_to_solution(path[i + 2]);
                modified = true;
                i += 4;
                continue;
            }

            i += 1;
        }

        modified
    }

    fn process_path_without_postprocess(
        &mut self,
        path: &mut [Node], // first and last are identical and may have degree > 2
    ) -> bool {
        let len = path.len();

        if self.graph.degree_of(path[0]) == 1 {
            return self.process_leaf_path(path);
        }

        if self.graph.degree_of(path[len - 1]) == 1 {
            path.reverse();
            return self.process_leaf_path(path);
        }

        self.process_path_with_uncovered_triples(path)
    }

    #[allow(unreachable_code, unused_variables)]
    fn process_path_with_postprocess(&mut self, path: &mut [Node], in_post: &mut BitSet) -> bool {
        if path.iter().any(|&u| in_post.get_bit(u)) {
            return false;
        }

        // we can remove groups of three as long as at least four nodes remain
        let nodes_to_remove = ((path.len() - 4) / 3) * 3;
        if nodes_to_remove == 0 {
            return false;
        }

        if path.iter().any(|&u| self.covered.get_bit(u))
            || path.iter().any(|&u| self.never_select.get_bit(u))
        {
            return false;
        }

        for &u in &path[2..2 + nodes_to_remove] {
            self.graph.remove_edges_at_node(u);
            self.covered.set_bit(u);
        }

        in_post.set_bits(path[1..2 + nodes_to_remove].iter().copied());

        self.graph.add_edge(path[1], path[2 + nodes_to_remove]);

        true
    }

    fn post_process_path(&mut self, path: &[Node]) {
        for (i, &u) in path.iter().enumerate() {
            if self.covered.set_bit(u) {
                continue;
            }

            let add = path[(i + 1).min(path.len() - 1)];
            self.domset.add_node(add);
            self.covered.set_bit(add);
            if i + 2 < path.len() {
                self.covered.set_bit(path[i + 2]);
            }
        }
    }
}

impl<G: AdjacencyList + AdjacencyTest + GraphEdgeEditing> Postprocessor<G>
    for LongPathPostProcessor<G>
{
    fn post_process(
        &mut self,
        graph: &mut G,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) {
        let mut rule_impl = RuleImpl {
            graph,
            domset,
            covered,
            never_select,
        };

        while let Some(path) = self.removed_paths.pop() {
            info!("PP: {path:?}");
            rule_impl.post_process_path(&path);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use itertools::Itertools;
    use rand::{Rng, SeedableRng, seq::SliceRandom};
    use rand_pcg::Pcg64Mcg;

    use crate::{
        exact::naive::naive_solver,
        graph::{AdjArray, AdjacencyList},
    };

    fn sample_non_adjacent_bits(
        rng: &mut impl Rng,
        n_attempts: Node,
        graph: &AdjArray,
        forbidden: Option<&BitSet>,
    ) -> BitSet {
        let mut bitset = graph.vertex_bitset_unset();
        for _ in 0..n_attempts {
            let idx = rng.gen_range(graph.vertices_range());

            // do not chose a forbidden node
            if let Some(f) = forbidden
                && f.get_bit(idx)
            {
                continue;
            }

            // reject if a neighbor was selected
            if graph.closed_neighbors_of(idx).any(|v| bitset.get_bit(v)) {
                continue;
            }

            bitset.set_bit(idx);
        }
        bitset
    }

    /// build a lolli path with a common center, a path
    /// connected to the center of n_stem new nodes,
    /// and a cycle connected to the center of n_cycle nodes
    fn build_lolli(
        rng: &mut impl Rng,
        n_stem: Node,
        n_cycle: Node,
    ) -> (AdjArray, Vec<Node>, Vec<Node>) {
        let nodes: Node = 1 + n_stem + n_cycle;
        let mut perm = (0..nodes).collect_vec();
        perm.shuffle(rng);
        let mut graph = AdjArray::new(nodes);

        for i in 0..(nodes as usize - 1) {
            graph.add_edge(perm[i], perm[i + 1]);
        }

        // center is perm[n_cycle + 1]
        let center = perm[n_cycle as usize];
        if n_cycle > 0 {
            graph.add_edge(perm[0], center);
        }

        let cycle = std::iter::once(center)
            .chain(perm[..=n_cycle as usize].iter().copied())
            .collect_vec();

        let path = perm.into_iter().skip(n_cycle as usize).collect_vec();

        (graph, path, cycle)
    }

    #[test]
    fn cycles() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x123456);
        for (n_stem, n_cycle) in (0..3).cartesian_product(3..8) {
            // repeats
            for i in 0..2000 {
                let (mut graph, _, cycle) = build_lolli(&mut rng, n_stem, n_cycle);
                let num_edges = graph.number_of_edges();
                assert_eq!(num_edges, graph.number_of_nodes());

                let mut solution = DominatingSet::new(graph.number_of_nodes());
                let mut covered = sample_non_adjacent_bits(&mut rng, i / 16, &graph, None);
                let redundant = sample_non_adjacent_bits(&mut rng, i % 3, &graph, Some(&covered));

                let dom_number = naive_solver(&graph, &covered, &redundant, None, None)
                    .unwrap()
                    .len() as NumNodes;

                let mut rule_impl = RuleImpl {
                    graph: &mut graph,
                    domset: &mut solution,
                    covered: &mut covered,
                    never_select: &redundant,
                };

                rule_impl.process_circle(&cycle);
                assert!(solution.len() > 0);

                let dom_number_after_reduction = solution.len() as NumNodes
                    + naive_solver(&graph, &covered, &redundant, None, None)
                        .unwrap()
                        .len() as NumNodes;

                assert_eq!(
                    dom_number, dom_number_after_reduction,
                    "n_stem: {n_stem}, n_cycle: {n_cycle}"
                );
            }
        }
    }

    #[test]
    fn isolated_paths() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x123456);
        for (n_stem, n_cycle) in (3..10).cartesian_product([0, 2, 3]) {
            // repeats
            for i in 0..2000 {
                let (mut graph, mut path, _) = build_lolli(&mut rng, n_stem, n_cycle);

                let mut solution = DominatingSet::new(graph.number_of_nodes());
                let mut covered = sample_non_adjacent_bits(&mut rng, i / 16, &graph, None);
                let redundant = sample_non_adjacent_bits(&mut rng, i % 3, &graph, Some(&covered));

                let clean_solution =
                    naive_solver(&graph, &covered, &redundant, None, None).unwrap();

                let mut rule_impl = RuleImpl {
                    graph: &mut graph,
                    domset: &mut solution,
                    covered: &mut covered,
                    never_select: &redundant,
                };

                let modified = rule_impl.process_path_without_postprocess(&mut path);
                assert!(!solution.is_empty());
                assert_eq!(solution.is_empty(), !modified);

                {
                    let tmp = naive_solver(&graph, &covered, &redundant, None, None).unwrap();
                    solution.add_nodes(tmp.iter());
                }

                assert_eq!(
                    clean_solution.len(),
                    solution.len(),
                    "before: {clean_solution:?}, after: {solution:?}"
                );
            }
        }
    }

    fn build_gnp_path(rng: &mut impl Rng, n_path: Node, n_random: Node) -> (AdjArray, Vec<Node>) {
        let nodes = n_path + n_random + 2;
        let mut graph = AdjArray::random_gnp(rng, nodes, 0.5);

        let mut path = (0..nodes).collect_vec();
        path.shuffle(rng);
        path.truncate(n_path as usize + 2);

        for u in path.iter().skip(1).take(n_path as usize) {
            graph.remove_edges_at_node(*u);
        }

        for i in 0..(n_path as usize + 1) {
            graph.add_edge(path[i], path[i + 1]);
        }

        (graph, path)
    }

    #[test]
    fn gnp_paths() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x123456);
        for (n_path, n_random) in (3..10).cartesian_product([8]) {
            // repeats
            for i in 0..1000 {
                let (mut graph, mut path) = build_gnp_path(&mut rng, n_path, n_random);

                let mut solution = DominatingSet::new(graph.number_of_nodes());
                let mut covered = sample_non_adjacent_bits(&mut rng, i / 16, &graph, None);
                let redundant = sample_non_adjacent_bits(&mut rng, i % 3, &graph, Some(&covered));

                let clean_solution =
                    naive_solver(&graph, &covered, &redundant, None, None).unwrap();

                let mut rule_impl = RuleImpl {
                    graph: &mut graph,
                    domset: &mut solution,
                    covered: &mut covered,
                    never_select: &redundant,
                };

                let modified = rule_impl.process_path_without_postprocess(&mut path);
                assert_eq!(solution.is_empty(), !modified);

                {
                    let tmp = naive_solver(&graph, &covered, &redundant, None, None).unwrap();
                    solution.add_nodes(tmp.iter());
                }

                assert_eq!(
                    clean_solution.len(),
                    solution.len(),
                    "before: {clean_solution:?}, after: {solution:?}"
                );
            }
        }
    }

    #[test]
    fn gnp_paths_with_triples() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x123456);
        let mut num_infeasible = 0;
        for (n_path, n_random) in (3..10).cartesian_product([8]) {
            // repeats
            for i in 0..1000 {
                let (mut graph, mut path) = build_gnp_path(&mut rng, n_path, n_random);
                let org_graph = graph.clone();

                let mut solution = DominatingSet::new(graph.number_of_nodes());

                // cover/redundant only on non-path nodes!
                let mut covered = sample_non_adjacent_bits(&mut rng, i % 8, &graph, None);
                let redundant = sample_non_adjacent_bits(&mut rng, i % 3, &graph, Some(&covered));

                let clean_solution =
                    if let Ok(x) = naive_solver(&graph, &covered, &redundant, None, None) {
                        x
                    } else {
                        num_infeasible += 1;
                        continue;
                    };

                let modified = {
                    let mut rule_impl = RuleImpl {
                        graph: &mut graph,
                        domset: &mut solution,
                        covered: &mut covered,
                        never_select: &redundant,
                    };

                    rule_impl.process_path_with_uncovered_triples(&mut path)
                };
                assert_eq!(!solution.is_empty(), modified);

                {
                    let tmp = naive_solver(&graph, &covered, &redundant, None, None).unwrap();
                    solution.add_nodes(tmp.iter());
                }

                assert_eq!(
                    clean_solution.len(),
                    solution.len(),
                    "before: {clean_solution:?}, after: {solution:?}, {org_graph:?}"
                );
            }
        }
        assert!(num_infeasible < 100);
    }

    #[test]
    fn gnp_paths_with_post_process() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x123456);
        for (n_path, n_random) in (3..10).cartesian_product([8]) {
            // repeats
            for i in 0..1000 {
                let (mut graph, mut path) = build_gnp_path(&mut rng, n_path, n_random);
                let org_graph = graph.clone();

                let mut solution = DominatingSet::new(graph.number_of_nodes());

                // cover/redundant only on non-path nodes!
                let mut forbidden =
                    BitSet::new_with_bits_set(graph.number_of_nodes(), path.iter().copied());
                let mut covered =
                    sample_non_adjacent_bits(&mut rng, i / 16, &graph, Some(&forbidden));
                forbidden |= &covered;
                let redundant = sample_non_adjacent_bits(&mut rng, i % 3, &graph, Some(&forbidden));

                let clean_solution =
                    naive_solver(&graph, &covered, &redundant, None, None).unwrap();

                let mut in_post = graph.vertex_bitset_unset();
                let modified = {
                    let mut rule_impl = RuleImpl {
                        graph: &mut graph,
                        domset: &mut solution,
                        covered: &mut covered,
                        never_select: &redundant,
                    };

                    rule_impl.process_path_with_postprocess(&mut path, &mut in_post)
                };
                assert!(solution.is_empty()); // rule does only add nodes in post

                {
                    let tmp = naive_solver(&graph, &covered, &redundant, None, None).unwrap();
                    solution.add_nodes(tmp.iter());
                }

                if modified {
                    let mut covered = solution.compute_covered(&graph);
                    let mut rule_impl = RuleImpl {
                        graph: &mut graph,
                        domset: &mut solution,
                        covered: &mut covered,
                        never_select: &redundant,
                    };
                    rule_impl.post_process_path(&path);
                }

                assert_eq!(
                    clean_solution.len(),
                    solution.len(),
                    "before: {clean_solution:?}, after: {solution:?}, {org_graph:?}"
                );
            }
        }
    }
}
