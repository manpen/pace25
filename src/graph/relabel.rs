use itertools::Itertools;

use crate::utils::DominatingSet;

use super::{AdjacencyList, IndexMapper, Node};

pub fn cuthill_mckee<G: AdjacencyList>(graph: &G) -> IndexMapper {
    let mut mapping = vec![Node::MAX; graph.len()];
    let mut queue = Vec::with_capacity(graph.len());

    // We reuse the set-functionality of DominatingSet as we need a fast iterator
    let mut candidates = DominatingSet::complete_set(graph.number_of_nodes());

    let start_node = graph
        .vertices()
        .map(|u| (graph.degree_of(u), u))
        .min()
        .unwrap()
        .1;
    queue.push(start_node);
    mapping[start_node as usize] = 0;

    let mut i = 0usize;
    loop {
        if queue.len() >= graph.len() {
            break;
        }

        if i >= queue.len() {
            let new_min_node = candidates
                .iter()
                .filter_map(|u| {
                    if mapping[u as usize] == Node::MAX {
                        Some((graph.degree_of(u), u))
                    } else {
                        None
                    }
                })
                .min()
                .unwrap()
                .1;

            candidates.remove_node(new_min_node);
            mapping[new_min_node as usize] = queue.len() as Node;
            queue.push(new_min_node);
        }

        let mut adj = graph
            .neighbors_of(queue[i])
            .filter(|&u| mapping[u as usize] == Node::MAX)
            .collect_vec();
        adj.sort_unstable_by(|&u, &v| {
            (mapping[u as usize], graph.degree_of(u))
                .cmp(&(mapping[v as usize], graph.degree_of(v)))
        });

        for u in adj {
            mapping[u as usize] = queue.len() as Node;
            queue.push(u);
            if candidates.is_in_domset(u) {
                candidates.remove_node(u);
            }
        }

        i += 1;
    }

    IndexMapper::from_vecs(mapping, queue)
}
