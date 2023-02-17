use super::*;
use crate::graph::{connectivity::Connectivity, *};
use std::fmt::Debug;

pub trait NaiveSolvableGraph:
    Clone
    + AdjacencyList
    + GraphEdgeOrder
    + ColoredAdjacencyList
    + ColoredAdjacencyTest
    + GraphEdgeEditing
    + Debug
{
}

impl<G> NaiveSolvableGraph for G where
    G: Clone
        + AdjacencyList
        + GraphEdgeOrder
        + ColoredAdjacencyList
        + ColoredAdjacencyTest
        + GraphEdgeEditing
        + Debug
{
}

pub fn naive_solver<G: NaiveSolvableGraph>(input_graph: &G) -> (NumNodes, ContractionSequence) {
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

fn recurse<G: NaiveSolvableGraph>(
    graph: &mut G,
    slack: Node,
    mut not_above: Node,
) -> Option<(NumNodes, ContractionSequence)> {
    if slack > not_above {
        return None;
    }

    let red_deg_before = graph.red_degrees().max().unwrap();
    assert!(red_deg_before <= slack);

    let mut contract_seq = ContractionSequence::new(graph.number_of_nodes());
    prune_leaves(graph, &mut contract_seq);
    prune_twins(graph, &mut contract_seq);

    assert!(red_deg_before >= graph.red_degrees().max().unwrap());

    if graph.number_of_edges() <= 1 {
        let mut tww = 0;
        if let Some(first_edge) = graph.colored_edges(true).next() {
            contract_seq.merge_node_into(first_edge.0, first_edge.1);
            tww = first_edge.2.is_red() as Node;
        }
        return Some((tww, contract_seq));
    }

    if let Some(res) = try_split_into_cc(graph, slack, not_above) {
        return res.map(|(tww, seq)| {
            (tww, {
                contract_seq.append(&seq);
                contract_seq
            })
        });
    }

    assert!(graph.red_degrees().max().unwrap() <= not_above);

    let mut pairs: Vec<_> = graph
        .distance_two_pairs()
        .filter_map(|(u, v)| {
            let red_neighs = graph.red_neighbors_after_merge(u, v, false);
            (red_neighs.cardinality() <= not_above).then_some((red_neighs.cardinality(), (u, v)))
        })
        .collect();

    pairs.sort();

    let mut best_solution = None;

    for (r, (u, v)) in pairs {
        if r > not_above {
            break;
        }

        let mut local_graph = graph.clone();
        local_graph.merge_node_into(u, v);

        let max_red_degree = local_graph.red_degrees().max().unwrap();
        if max_red_degree > not_above {
            continue;
        }

        if let Some((sol_size, contract_seq)) =
            recurse(&mut local_graph, slack.max(max_red_degree), not_above)
        {
            let sol_size = sol_size.max(max_red_degree);
            best_solution = Some(((u, v), sol_size, contract_seq));

            if sol_size <= slack {
                break;
            }

            not_above = sol_size.checked_sub(1).unwrap();
        }
    }

    let ((u, v), tww, seq) = best_solution?;

    contract_seq.merge_node_into(u, v);
    contract_seq.append(&seq);
    Some((tww, contract_seq))
}

fn try_split_into_cc<G: NaiveSolvableGraph>(
    graph: &mut G,
    slack: u32,
    not_above: u32,
) -> Option<Option<(NumNodes, ContractionSequence)>> {
    let part = graph.partition_into_connected_components(true);
    if part.number_of_classes() == 1 || 10 * part.number_of_unassigned() < graph.number_of_nodes() {
        return None;
    }

    let mut max_tww = 0;
    let mut slack = slack;

    let mut contract_seq = ContractionSequence::new(graph.number_of_nodes());

    for (mut subgraph, mapper) in part.split_into_subgraphs(graph) {
        let (size, sol) = recurse(&mut subgraph, slack, not_above)?;
        slack = slack.max(size);
        max_tww = max_tww.max(size);
        for &(rem, sur) in sol.merges() {
            contract_seq.merge_node_into(
                mapper.old_id_of(rem).unwrap(),
                mapper.old_id_of(sur).unwrap(),
            );
        }
    }

    Some(Some((max_tww, contract_seq)))
}

#[cfg(test)]
mod test {
    use crate::{graph::*, io::*, testing::get_test_graphs_with_tww};
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

    #[test]
    fn small_random() {
        for (filename, graph, presolved_tww) in
            get_test_graphs_with_tww("instances/small-random/*.gr").step_by(3)
        {
            if graph.number_of_nodes() > 10 {
                continue;
            }
            println!(" Test {filename}");
            let (tww, _seq) = naive_solver(&graph);
            assert_eq!(tww, presolved_tww, "file: {filename}");
        }
    }
}
