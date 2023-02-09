use super::*;
use crate::graph::*;
use std::fmt::Debug;

pub fn naive_solver<
    G: Clone
        + AdjacencyList
        + GraphEdgeOrder
        + ColoredAdjacencyList
        + ColoredAdjacencyTest
        + GraphEdgeEditing
        + Debug,
>(
    input_graph: &G,
) -> (NumNodes, ContractionSequence) {
    let mut graph = input_graph.clone();
    let nodes = graph.number_of_nodes();

    let (twin_width, mut contract_seq) = recurse(&mut graph, 0, nodes).unwrap();

    contract_seq.add_unmerged_singletons(input_graph).unwrap();

    assert_eq!(
        twin_width,
        contract_seq
            .compute_twin_width(input_graph.clone())
            .unwrap()
    );

    (twin_width, contract_seq)
}

fn recurse<
    G: Clone
        + AdjacencyList
        + GraphEdgeOrder
        + ColoredAdjacencyList
        + ColoredAdjacencyTest
        + GraphEdgeEditing
        + Debug,
>(
    graph: &mut G,
    slack: Node,
    not_above: Node,
) -> Option<(NumNodes, ContractionSequence)> {
    let mut preprocessing_contract_seq = ContractionSequence::new(graph.number_of_nodes());

    prune_leaves(graph, &mut preprocessing_contract_seq);
    prune_twins(graph, &mut preprocessing_contract_seq);

    if graph.number_of_edges() <= 1 {
        if let Some(first_edge) = graph.edges(true).next() {
            preprocessing_contract_seq.merge_node_into(first_edge.0, first_edge.1);
        }
        return Some((0, preprocessing_contract_seq));
    }

    assert!(graph.red_degrees().max().unwrap() <= not_above);

    let mut pairs: Vec<_> = graph
        .distance_two_pairs()
        .map(|(u, v)| {
            let red_neighs = graph.red_neighbors_after_merge(u, v, false);
            let mut red_deg = red_neighs.cardinality() as NumNodes;

            for w in red_neighs.iter().map(|x| x as Node) {
                if graph.red_degree_of(w) + 1 > red_deg {
                    red_deg = red_deg.max(
                        graph.red_degree_of(w) + 1
                            - graph.has_red_edge(u, w) as NumNodes
                            - graph.has_red_edge(v, w) as NumNodes,
                    );
                }
            }

            (red_deg, (u, v))
        })
        .filter(|&(deg, _)| deg <= not_above)
        .collect();

    pairs.sort_by_key(|(deg, _)| *deg);

    let mut best_contract_sequence = None;
    let mut best_merge = None;
    let mut best_twin_with = not_above + 1;

    for (r, (u, v)) in pairs {
        if r >= best_twin_with {
            break;
        }

        let mut local_graph = graph.clone();
        local_graph.merge_node_into(u, v);
        assert!(local_graph.red_degree_of(v) <= r);

        debug_assert!(local_graph.red_degrees().max().unwrap() < best_twin_with,);

        if let Some((sol_size, contract_seq)) =
            recurse(&mut local_graph, slack.max(r), best_twin_with - 1)
        {
            let sol_size = sol_size.max(r);
            assert!(best_twin_with > sol_size);

            best_twin_with = sol_size;
            best_contract_sequence = Some(contract_seq);
            best_merge = Some((u, v));

            if best_twin_with <= slack {
                break;
            }
        }
    }

    preprocessing_contract_seq.merge_node_into(best_merge?.0, best_merge?.1);
    preprocessing_contract_seq.append(&best_contract_sequence?);
    (best_twin_with <= not_above).then_some((best_twin_with, preprocessing_contract_seq))
}

#[cfg(test)]
mod test {
    use crate::{graph::*, io::*};
    use std::{fs::File, io::BufReader};

    use super::naive_solver;

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

            let (size, _sol) = naive_solver(&graph);
            assert_eq!(size, tww, "file: {filename}");
        }
    }
}
