use super::*;
use crate::{exact::naive::naive_solver, graph::*};
use std::{collections::HashMap, marker::PhantomData};

pub struct RuleSmallExactReduction<G> {
    _graph: PhantomData<G>,
}

impl<Graph: AdjacencyList + GraphEdgeEditing + 'static> ReductionRule<Graph>
    for RuleSmallExactReduction<Graph>
{
    const NAME: &str = "SmallExact";

    fn apply_rule(
        graph: &mut Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        const MAX_CC_SIZE: Node = 15;

        let partition = graph.partition_into_connected_components(true);
        let mut modified = false;

        let mut mapping: HashMap<Node, Node> = HashMap::with_capacity(MAX_CC_SIZE as usize * 2);
        for class in 0..partition.number_of_classes() {
            let n = partition.number_in_class(class);
            if n > 15 {
                continue;
            }

            mapping.clear();
            mapping.extend(
                partition
                    .members_of_class(class)
                    .enumerate()
                    .map(|(new, old)| (old, new as Node)),
            );

            let mut graph_mapped = AdjArray::new(n);
            let mut covered_mapped = graph_mapped.vertex_bitset_unset();
            for (&oldu, &newu) in mapping.iter() {
                if covered.get_bit(oldu) {
                    covered_mapped.set_bit(newu);
                }

                for oldv in graph.neighbors_of(oldu) {
                    let newv = *mapping.get(&oldv).unwrap();
                    if newv >= newu {
                        continue;
                    }

                    graph_mapped.add_edge(newu, newv, EdgeColor::Black);
                }
            }

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
                    covered.set_bits(graph.closed_neighbors_of(oldv));
                    graph.remove_edges_at_node(oldv);
                }
            }

            modified = true;
        }

        (modified, None::<Box<dyn Postprocessor<Graph>>>)
    }
}
