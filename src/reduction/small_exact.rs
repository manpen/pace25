use itertools::Itertools;
use smallvec::SmallVec;

#[allow(unused_imports)]
use log::{debug, info};

use super::*;
use crate::{exact, graph::*};
use std::{
    marker::PhantomData,
    time::{Duration, Instant},
};

const NOT_SET: Node = Node::MAX;

pub struct RuleSmallExactReduction<G> {
    _graph: PhantomData<G>,
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
    1000
} else {
    100
};
const MAX_UNCOVERED_SIZE: Node = if exact::DEFAULT_SOLVER_IS_FAST {
    MAX_CC_SIZE
} else {
    30
};
const MAX_DURATION: Duration = Duration::from_secs(30);

impl<Graph: AdjacencyList + GraphEdgeEditing + 'static> ReductionRule<Graph>
    for RuleSmallExactReduction<Graph>
{
    const NAME: &str = "SmallExact";

    fn apply_rule(
        graph: &mut Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        let mut small_ccs = Vec::with_capacity(128);
        let mut walker = ConnectedComponentWalker::new(graph.number_of_nodes(), Some(MAX_CC_SIZE));

        let mut uncovered = Vec::with_capacity(1 + MAX_UNCOVERED_SIZE as usize);

        let ds_size_before = solution.len();
        while let Some(nodes) = walker.next_cc(graph) {
            match nodes.len() {
                // we have special cases for really small ccs
                3 => {
                    Self::process_3nodes(graph, solution, covered, redundant, &nodes);
                }

                4 => {
                    Self::process_4nodes(graph, solution, covered, redundant, &nodes);
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
                        solution.fix_node(uncovered[0]);
                        covered.set_bits(nodes.iter().copied());
                    } else if uncovered.len() == 2 {
                        // if there are two, there are two options: either there's one node u that can cover both:
                        // then we add u; otherwise we've established a lower bound of 2, and can safely add both uncovered nodes
                        if let Some(u) = nodes.iter().copied().find(|&u| {
                            graph
                                .closed_neighbors_of(u)
                                .filter(|&v| v == uncovered[0] || v == uncovered[1])
                                .count()
                                == 2
                        }) {
                            solution.fix_node(u);
                        } else {
                            solution.fix_nodes(nodes.iter().copied());
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
        Self::process_small_ccs(
            graph,
            solution,
            covered,
            redundant,
            small_ccs.into_iter().map(|(_, cc)| cc),
        );

        (
            ds_size_before != solution.len(),
            None::<Box<dyn Postprocessor<Graph>>>,
        )
    }
}

impl<Graph: AdjacencyList + GraphEdgeEditing + 'static> RuleSmallExactReduction<Graph> {
    fn process_3nodes(
        graph: &Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        _redundant: &BitSet,
        nodes: &[Node],
    ) {
        assert_eq!(nodes.len(), 3);

        if !nodes.iter().all(|&u| covered.get_bit(u)) {
            let deg2 = nodes.iter().find(|&&u| graph.degree_of(u) == 2).unwrap();
            solution.fix_node(*deg2);
            covered.set_bits(nodes.iter().copied());
        }
    }

    fn process_4nodes(
        graph: &Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        _redundant: &BitSet,
        nodes: &[Node],
    ) {
        assert_eq!(nodes.len(), 4);

        let num_uncovered = nodes.iter().filter(|&&u| !covered.get_bit(u)).count();
        if num_uncovered > 0 {
            if let Some(u) = nodes.iter().find(|&&u| graph.degree_of(u) == 3) {
                solution.fix_node(*u);
            } else if num_uncovered == 1 {
                solution.fix_node(
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
                solution.fix_node(*u);
            } else {
                let (a, b) = nodes
                    .iter()
                    .copied()
                    .filter(|&u| graph.degree_of(u) > 1)
                    .take(2)
                    .collect_tuple()
                    .unwrap();

                solution.fix_node(a);
                solution.fix_node(b);
            }

            covered.set_bits(nodes.iter().copied());
        }
    }

    fn process_small_ccs(
        graph: &Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &BitSet,
        mut ccs: impl Iterator<Item = CC>,
    ) {
        let mut num_timeout = 0;
        let mut num_solved = 0;

        let mut org_to_small = vec![NOT_SET; graph.len()];

        let start = Instant::now();
        #[allow(clippy::while_let_on_iterator)]
        while let Some(nodes) = ccs.next() {
            if start.elapsed() > MAX_DURATION {
                break;
            }
            let n = nodes.len() as NumNodes;

            for (i, &u) in nodes.iter().enumerate() {
                debug_assert_eq!(org_to_small[u as usize], NOT_SET);
                org_to_small[u as usize] = i as Node;
            }

            let mut graph_mapped = AdjArray::new(n);
            let mut covered_mapped = graph_mapped.vertex_bitset_unset();
            let mut redundant_mapped = graph_mapped.vertex_bitset_unset();

            for (newu, &oldu) in nodes.iter().enumerate() {
                let newu = newu as Node;

                let ucovered = covered.get_bit(oldu);
                if ucovered {
                    covered_mapped.set_bit(newu);
                }
                if redundant.get_bit(oldu) {
                    redundant_mapped.set_bit(newu);
                }

                for oldv in graph.neighbors_of(oldu) {
                    if oldv >= oldu {
                        continue;
                    }

                    if !ucovered || !covered.get_bit(oldv) {
                        let newv = org_to_small[oldv as usize];
                        debug_assert_ne!(oldv, NOT_SET);
                        graph_mapped.add_edge(newu, newv);
                    }
                }
            }

            let solution_mapped = match exact::default_exact_solver(
                &graph_mapped,
                &covered_mapped,
                &redundant_mapped,
                None,
                Some(Duration::from_secs(1)),
            ) {
                Ok(x) => {
                    num_solved += 1;
                    x
                }
                Err(_) => {
                    info!(
                        "No solution for n={} covered={}",
                        graph_mapped.number_of_nodes(),
                        covered.cardinality()
                    );
                    num_timeout += 1;
                    continue;
                }
            };

            for newu in solution_mapped.iter() {
                let oldu = nodes[newu as usize];
                solution.fix_node(oldu);
                covered.set_bits(graph.closed_neighbors_of(oldu));
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
