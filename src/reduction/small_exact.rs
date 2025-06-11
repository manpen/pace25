use itertools::Itertools;
use smallvec::SmallVec;

#[allow(unused_imports)]
use log::{debug, info};

use super::*;
use crate::{
    exact::{
        self,
        highs_advanced::{HighsCache, HighsDominatingSetSolver, SolverResult, unit_weight},
    },
    graph::*,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

pub struct RuleSmallExactReduction {
    #[allow(unused)]
    highs_cache: Option<Arc<HighsCache>>,
}

impl RuleSmallExactReduction {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self { highs_cache: None }
    }

    pub fn new_with_cache(cache: Arc<HighsCache>) -> Self {
        Self {
            highs_cache: Some(cache),
        }
    }
}

struct ConnectedComponentWalker {
    visited: BitSet,
    start_at: Node,
    stack: Vec<Node>,
    max_size: Node,
}

type CC = SmallVec<[Node; 8]>;
impl ConnectedComponentWalker {
    pub fn new(n: NumNodes, max_size: Option<NumNodes>) -> Self {
        Self {
            visited: BitSet::new(n),
            start_at: 0,
            stack: Vec::with_capacity(128),
            max_size: max_size.unwrap_or(n),
        }
    }

    fn visit_at<G: AdjacencyList>(&mut self, graph: &G, u: Node) -> Option<CC> {
        let mut cc: CC = Default::default();

        self.stack.clear();
        self.visited.set_bit(u);
        self.stack.push(u);
        cc.push(u);

        while let Some(u) = self.stack.pop() {
            for v in graph.neighbors_of(u) {
                if !self.visited.set_bit(v) {
                    self.stack.push(v);

                    if cc.len() <= self.max_size as usize {
                        // If the CC is too large, we will put self.max_size+1 nodes in the list
                        // to detect the overshoot
                        cc.push(v);
                    }
                }
            }
        }

        assert!(cc.len() > 1);

        (cc.len() <= self.max_size as usize).then_some(cc)
    }

    pub fn next_cc<G: AdjacencyList>(&mut self, graph: &G) -> Option<CC> {
        loop {
            let start = (self.start_at..graph.number_of_nodes())
                .find(|&u| !self.visited.get_bit(u) && graph.degree_of(u) > 0);

            self.start_at = start.unwrap_or(graph.number_of_nodes()) + 1;

            if let Some(cc) = self.visit_at(graph, start?) {
                return Some(cc);
            }
        }
    }
}

const MAX_CC_SIZE: Node = if exact::DEFAULT_SOLVER_IS_FAST {
    1500
} else {
    100
};
const MAX_UNCOVERED_SIZE: Node = if exact::DEFAULT_SOLVER_IS_FAST {
    MAX_CC_SIZE
} else {
    30
};

const COMBINE_CCS_UPTO: Node = 100;

const MAX_DURATION: Duration = Duration::from_secs(30);

impl<Graph: Clone + AdjacencyList + AdjacencyTest + GraphEdgeEditing + 'static> ReductionRule<Graph>
    for RuleSmallExactReduction
{
    const NAME: &str = "SmallExact";

    /// We expect to run this ReductionRule only once; hence, we do not cache any variables in
    /// RuleSmallExactReduction itself
    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        let mut small_ccs = Vec::with_capacity(128);
        let mut walker = ConnectedComponentWalker::new(graph.number_of_nodes(), Some(MAX_CC_SIZE));

        let mut uncovered: Vec<Node> = Vec::with_capacity(1 + MAX_UNCOVERED_SIZE as usize);

        let ds_size_before = domset.len();

        while let Some(nodes) = walker.next_cc(graph) {
            match nodes.len() {
                // we have special cases for really small ccs
                3 => {
                    Self::process_3nodes(graph, domset, covered, never_select, &nodes);
                }

                4 => {
                    Self::process_4nodes(graph, domset, covered, never_select, &nodes);
                }

                _ => {
                    // the next special cases are based on the number of uncovered nodes -- so let's find them
                    uncovered.clear();
                    uncovered.extend(
                        nodes
                            .iter()
                            .copied()
                            .filter(|&x| !covered.get_bit(x))
                            .take(1 + MAX_UNCOVERED_SIZE as usize),
                    );
                    if uncovered.is_empty() || uncovered.len() > MAX_UNCOVERED_SIZE as usize {
                        continue;
                    }

                    if uncovered.len() == 1 {
                        // if there's only one uncovered node in a cc, it can be safely add to the solution
                        domset.add_node(uncovered[0]);
                        covered.set_bits(nodes.iter().copied());
                    } else if uncovered.len() == 2 {
                        // if there are two, there are two options: either there's one node u that can cover both:
                        // then we add u; otherwise we've established a lower bound of 2, and can safely add both uncovered nodes
                        if let Some(u) = nodes
                            .iter()
                            .copied()
                            .filter(|&u| !never_select.get_bit(u))
                            .find(|&u| {
                                graph
                                    .closed_neighbors_of(u)
                                    .filter(|&v| v == uncovered[0] || v == uncovered[1])
                                    .count()
                                    == 2
                            })
                        {
                            domset.add_node(u);
                        } else {
                            domset.add_nodes(nodes.iter().copied());
                        }
                        covered.set_bits(uncovered.iter().copied());
                    } else {
                        // no special case: then we queue it for the MIP run
                        small_ccs.push((uncovered.len() as NumNodes, nodes))
                    }
                }
            }
        }

        small_ccs.sort_by_key(|(u, cc)| (*u, cc.len()));

        if small_ccs.len() > 1 {
            let mut doner = 0;
            let mut receiver = small_ccs.len() - 1;
            while doner < receiver {
                if small_ccs[doner].1.len() + small_ccs[receiver].1.len()
                    < COMBINE_CCS_UPTO as usize
                {
                    let d = std::mem::take(&mut small_ccs[doner].1);
                    small_ccs[receiver].1.extend(d.into_iter());
                    doner += 1
                } else {
                    receiver -= 1;
                }
            }
        }

        self.process_small_ccs(
            graph,
            domset,
            covered,
            never_select,
            small_ccs.into_iter().map(|(_, cc)| cc),
        );

        (
            ds_size_before != domset.len(),
            None::<Box<dyn Postprocessor<Graph>>>,
        )
    }
}

impl RuleSmallExactReduction {
    fn process_3nodes<Graph: Clone + AdjacencyList + GraphEdgeEditing + 'static>(
        graph: &Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &BitSet,
        nodes: &[Node],
    ) {
        assert_eq!(nodes.len(), 3);

        if nodes.iter().all(|&u| covered.get_bit(u)) {
            return;
        }

        if let Some(&deg2) = nodes
            .iter()
            .filter(|&&u| !never_select.get_bit(u))
            .find(|&&u| graph.degree_of(u) == 2)
        {
            domset.add_node(deg2);
            covered.set_bits(nodes.iter().copied());
        } else {
            for &u in nodes {
                if !covered.get_bit(u) {
                    domset.add_node(u);
                    covered.set_bits(graph.closed_neighbors_of(u));
                }
            }
        }
    }

    fn process_4nodes<Graph: Clone + AdjacencyList + GraphEdgeEditing + 'static>(
        graph: &Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        _never_select: &BitSet,
        nodes: &[Node],
    ) {
        assert_eq!(nodes.len(), 4);

        let num_uncovered = nodes.iter().filter(|&&u| !covered.get_bit(u)).count();
        if num_uncovered > 0 {
            if let Some(u) = nodes.iter().find(|&&u| graph.degree_of(u) == 3) {
                domset.add_node(*u);
            } else if num_uncovered == 1 {
                domset.add_node(
                    nodes
                        .iter()
                        .copied()
                        .find(|&u| !covered.get_bit(u))
                        .unwrap(),
                );
            } else if let Some(u) = nodes.iter().find(|&&u| {
                graph
                    .closed_neighbors_of(u)
                    .filter(|&v| !covered.get_bit(v))
                    .count()
                    == num_uncovered
            }) {
                domset.add_node(*u);
            } else {
                let (a, b) = nodes
                    .iter()
                    .copied()
                    .filter(|&u| graph.degree_of(u) > 1)
                    .take(2)
                    .collect_tuple()
                    .unwrap();

                domset.add_node(a);
                domset.add_node(b);
            }

            covered.set_bits(nodes.iter().copied());
        }
    }

    fn process_small_ccs<
        Graph: Clone + AdjacencyList + AdjacencyTest + GraphEdgeEditing + 'static,
    >(
        &self,
        graph: &Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &BitSet,
        mut ccs: impl Iterator<Item = CC>,
    ) where
        Self: ReductionRule<Graph>,
    {
        let mut num_timeout = 0;
        let mut num_solved = 0;

        let mut solver = HighsDominatingSetSolver::new(graph.number_of_nodes());
        if let Some(cache) = &self.highs_cache {
            solver.register_cache(cache.clone());
        }

        let start = Instant::now();
        #[allow(clippy::while_let_on_iterator)]
        while let Some(nodes) = ccs.next() {
            let n = nodes.len() as NumNodes;
            if n == 0 {
                continue;
            }

            if start.elapsed() > MAX_DURATION {
                break;
            }

            let problem =
                solver.build_problem_of_subgraph(graph, covered, never_select, &nodes, unit_weight);

            if let SolverResult::Optimal(solved) = problem.solve_exact(Some(Duration::from_secs(1)))
            {
                domset.add_nodes(solved.into_iter());
                covered.set_bits(nodes.into_iter());
                num_solved += 1;
            } else {
                num_timeout += 1;
            }
        }

        let num_unconsidered = ccs.count();
        let num_found = num_solved + num_timeout + num_unconsidered;

        info!(
            "{} Found {num_found:6} large small ccs. Solved {num_solved:6}. Timeout {num_timeout:6}. Unconsidered: {num_unconsidered:6}",
            Self::NAME,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::NumNodes;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    #[test]
    fn generic_before_and_after() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x1235342);
        const NODES: NumNodes = 20;
        crate::testing::test_before_and_after_rule(
            &mut rng,
            |_| RuleSmallExactReduction::new(),
            false,
            NODES,
            400,
        );
    }

    #[test]
    fn generic_before_and_after_exhaust() {
        // this test does not make a terrible lot of sense (as cc are either completely removed or remain untouched).
        // but let's be sure that we did not miss anything
        let mut rng = Pcg64Mcg::seed_from_u64(0x43538092);
        const NODES: NumNodes = 20;
        crate::testing::test_before_and_after_rule(
            &mut rng,
            |_| RuleSmallExactReduction::new(),
            true,
            NODES,
            400,
        );
    }
}
