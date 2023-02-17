use itertools::Itertools;

use crate::graph::*;

pub fn lower_bound<G: ColoredAdjacencyList + GraphEdgeEditing>(
    graph: &G,
    known_lower: NumNodes,
) -> NumNodes {
    let mut order = graph.vertices().zip(graph.degrees()).collect_vec();
    order.sort_unstable_by_key(|(_, d)| *d);

    let mut lower = graph.number_of_nodes();

    let mut order = order.as_slice();
    while let Some((&(u, du), remaining_order)) = order.split_first() {
        for &(v, dv) in remaining_order {
            assert!(dv >= du);
            if du + lower <= dv {
                break;
            }

            lower = lower.min(graph.red_degree_after_merge(u, v));

            if lower <= known_lower {
                return known_lower;
            }
        }
        order = remaining_order;
    }

    lower
}
