use itertools::Itertools;
use log::info;

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

macro_rules! unwrap_or_continue {
    ($x:expr) => {
        match $x {
            Some(x) => x,
            None => continue,
        }
    };
}

fn nodes_in_small_partitions(partition: &Partition, n: NumNodes, max_size: u32) -> Vec<Vec<Node>> {
    let mut partition_mapping: Vec<Option<u32>> =
        vec![None; partition.number_of_classes() as usize];
    let mut number_of_small_partitions = 0u32;
    {
        for i in 0..partition.number_of_classes() {
            if partition.number_in_class(i) < max_size {
                partition_mapping[i as usize] = Some(number_of_small_partitions);
                number_of_small_partitions += 1;
            }
        }
    }
    let mut small_partitions = partition_mapping
        .iter()
        .enumerate()
        .filter(|(_, x)| x.is_some())
        .map(|(i, _)| Vec::with_capacity(partition.number_in_class(i as u32) as usize))
        .collect_vec();

    for u in 0..n {
        let class = unwrap_or_continue!(partition.class_of_node(u));
        let new_class = unwrap_or_continue!(partition_mapping[class as usize]);
        small_partitions[new_class as usize].push(u);
    }

    small_partitions.sort_unstable_by_key(|x| x.len());
    small_partitions
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
        const MAX_CC_SIZE: Node = 32;
        const MAX_DURATION: Duration = Duration::from_secs(30);

        let partition = graph.partition_into_connected_components(true);
        let mut modified = false;

        let small_partitions =
            nodes_in_small_partitions(&partition, graph.number_of_nodes(), MAX_CC_SIZE);

        let mut mapping: HashMap<Node, Node> = HashMap::with_capacity(MAX_CC_SIZE as usize * 2);
        let start = Instant::now();
        for nodes in small_partitions {
            let n = nodes.len() as NumNodes;

            if n == 2 {
                // if there's an uncovered node, select it (and thereby it's neighbor)
                if !covered.get_bit(nodes[0]) {
                    solution.fix_node(nodes[0]);
                } else if !covered.get_bit(nodes[1]) {
                    solution.fix_node(nodes[1]);
                }
                covered.set_bit(nodes[0]);
                covered.set_bit(nodes[1]);
                graph.remove_edges_at_node(nodes[0]);
                modified = true;
                continue;
            }

            if n == 3 {
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

            if n == 4 {
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

            if start.elapsed() > MAX_DURATION {
                break;
            }

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

            info!(
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

            let solution_mapped = naive_solver(
                &graph_mapped,
                &covered_mapped,
                &graph_mapped.vertex_bitset_unset(),
                None,
            )
            .unwrap();

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
