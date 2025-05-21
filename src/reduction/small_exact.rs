use itertools::Itertools;
use log::{debug, info};
use smallvec::SmallVec;

use super::*;
use crate::{exact::naive::naive_solver, graph::*};
use std::{
    collections::HashMap,
    marker::PhantomData,
    time::{Duration, Instant},
};

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

impl<Graph: AdjacencyList + GraphEdgeEditing + 'static> ReductionRule<Graph>
    for RuleSmallExactReduction<Graph>
{
    const NAME: &str = "SmallExact";

    fn apply_rule(
        graph: &mut Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        _redundant: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        const MAX_CC_SIZE: Node = 80;
        const MAX_UNCOVERED_SIZE: Node = 10;
        const MAX_DURATION: Duration = Duration::from_secs(30);

        let mut small_ccs = Vec::with_capacity(128);
        let mut walker = ConnectedComponentWalker::new(graph.number_of_nodes(), Some(MAX_CC_SIZE));

        let mut modified = false;
        let mut num_cc3 = 0;
        let mut num_cc4 = 0;

        let mut uncovered = Vec::with_capacity(1 + MAX_UNCOVERED_SIZE as usize);

        while let Some(nodes) = walker.next_cc(graph) {
            match nodes.len() {
                3 => {
                    num_cc3 += 1;
                    if !nodes.iter().all(|&u| covered.get_bit(u)) {
                        let deg2 = nodes.iter().find(|&&u| graph.degree_of(u) == 2).unwrap();
                        solution.fix_node(*deg2);
                        covered.set_bits(nodes.iter().copied());
                    }
                    for u in nodes {
                        graph.remove_edges_at_node(u);
                    }

                    modified = true;
                    continue;
                }

                4 => {
                    num_cc4 += 1;
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
                    for u in nodes {
                        graph.remove_edges_at_node(u);
                    }

                    modified = true;
                    continue;
                }

                _ => {
                    uncovered.clear();
                    uncovered.extend(
                        nodes
                            .iter()
                            .copied()
                            .filter(|&x| !covered.get_bit(x))
                            .take(1 + MAX_UNCOVERED_SIZE as usize),
                    );

                    if uncovered.is_empty() {
                        // should not happen if we have proper reduction rules in place
                        continue;
                    }

                    if uncovered.len() == 1 {
                        solution.fix_node(uncovered[0]);
                        covered.set_bits(nodes.iter().copied());
                        for &u in &nodes {
                            graph.remove_edges_at_node(u);
                        }
                    } else if uncovered.len() == 2 {
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

                        covered.set_bits(nodes.iter().copied());
                        for &u in &nodes {
                            graph.remove_edges_at_node(u);
                        }
                    } else if uncovered.len() <= MAX_UNCOVERED_SIZE as usize {
                        small_ccs.push((uncovered.len() as NumNodes, nodes))
                    }
                }
            }
        }

        info!(
            "{} Found {num_cc3} CC3, {num_cc4} CC4, and {} large ccs",
            Self::NAME,
            small_ccs.len()
        );

        small_ccs.sort_by_key(|(u, cc)| (*u, cc.len()));

        let mut mapping: HashMap<Node, Node> = HashMap::with_capacity(MAX_CC_SIZE as usize * 2);
        let start = Instant::now();
        for (_, nodes) in small_ccs {
            if start.elapsed() > MAX_DURATION {
                break;
            }
            let n = nodes.len() as NumNodes;

            mapping.clear();
            mapping.extend(nodes.iter().enumerate().map(|(i, &u)| (u, i as Node)));

            let mut graph_mapped = AdjArray::new(n);
            let mut covered_mapped = graph_mapped.vertex_bitset_unset();
            for (&oldu, &newu) in mapping.iter() {
                let ucovered = covered.get_bit(oldu);
                if ucovered {
                    covered_mapped.set_bit(newu);
                }

                for oldv in graph.neighbors_of(oldu) {
                    let newv = *mapping.get(&oldv).unwrap();
                    if newv >= newu {
                        continue;
                    }

                    if !ucovered || !covered.get_bit(oldv) {
                        graph_mapped.add_edge(newu, newv, EdgeColor::Black);
                    }
                }
            }

            debug!(
                "RuleSmallExact: Process CC with {n:2} nodes and {:3}. Max degree: {} Covered: {}; {}",
                graph_mapped.number_of_edges(),
                graph_mapped.degrees().max().unwrap(),
                covered_mapped.cardinality(),
                if graph_mapped.number_of_edges() + 1 == n {
                    " is tree"
                } else {
                    ""
                }
            );

            let solution_mapped = match naive_solver(
                &graph_mapped,
                &covered_mapped,
                &graph_mapped.vertex_bitset_unset(),
                None,
                Some(Duration::from_secs(1)),
            ) {
                Ok(x) => x,
                Err(_) => {
                    info!(
                        "No solution for n={} covered={}",
                        graph.number_of_nodes(),
                        covered.cardinality()
                    );
                    continue;
                }
            };

            for (&oldv, &newv) in mapping.iter() {
                if solution_mapped.is_in_domset(newv) {
                    solution.fix_node(oldv);
                }
                covered.set_bit(oldv);
            }

            for u in nodes {
                graph.remove_edges_at_node(u);
            }

            modified = true;
        }

        (modified, None::<Box<dyn Postprocessor<Graph>>>)
    }
}
