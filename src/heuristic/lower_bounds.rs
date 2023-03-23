use itertools::Itertools;
use rand::{seq::IteratorRandom, Rng};

use crate::prelude::{naive::naive_solver_with_bounds, *};

pub fn lower_bound_lb1<G: ColoredAdjacencyList + GraphEdgeEditing>(
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

pub fn lower_bound_subgraph<G: FullfledgedGraph>(
    graph: &G,
    mut known_lower: NumNodes,
    mut known_upper: NumNodes,
) -> NumNodes {
    let mut rng = rand::thread_rng();

    let max_nodes = 30;
    if known_lower + 1 >= max_nodes {
        return known_lower;
    }
    known_upper = known_upper.min(max_nodes as NumNodes - 1);

    for _ in 0..100 {
        if known_lower >= known_upper {
            break;
        }

        let seed_node = loop {
            let seed = rng.gen_range(graph.vertices_range());
            if graph.degree_of(seed) > 0 {
                break seed;
            }
        };

        let bfs = graph.bfs(seed_node).take(max_nodes as usize);

        let marked =
            BitSet::new_all_unset_but(graph.number_of_nodes(), bfs.filter(|_| rng.gen_bool(0.9)));

        let subgraph = graph.vertex_induced(&marked).0;

        let tww = naive_solver_with_bounds(&subgraph, known_lower, known_upper)
            .map_or(known_upper + 1, |(tww, _)| tww);

        known_lower = known_lower.max(tww);
    }

    known_lower
}

pub fn lower_bound_art_cut<G: FullfledgedGraph>(
    graph: &G,
    mut known_lower: NumNodes,
    known_upper: NumNodes,
) -> NumNodes {
    if graph.number_of_nodes() < 10 + 2 * known_upper {
        return naive_solver_with_bounds(graph, known_lower, known_upper)
            .map_or(known_lower, |(tww, _seq)| tww);
    }

    let part = graph.partition_into_connected_components(true);
    if part.number_of_classes() > 1 {
        let mut subgraphs = part.split_into_subgraphs(graph);
        subgraphs.sort_by_key(|x| x.0.number_of_nodes());

        for (subgraph, _) in subgraphs {
            if known_lower >= known_upper {
                break;
            }

            if subgraph.number_of_nodes() <= known_lower + 1 {
                continue;
            }

            known_lower = known_lower.max(lower_bound_art_cut(&subgraph, known_lower, known_upper));
        }

        return known_lower;
    }

    for u in graph
        .compute_articulation_points()
        .iter()
        .choose_multiple(&mut rand::thread_rng(), 3)
    {
        let mut graph = graph.clone();
        graph.remove_edges_at_node(u);
        known_lower = known_lower.max(lower_bound_art_cut(&graph, known_lower, known_upper));
    }

    known_lower
}
