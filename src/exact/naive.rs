use super::*;
use crate::{
    graph::{connectivity::Connectivity, *},
    prelude::monte_carlo_search_tree::timeout_monte_carlo_search_tree_solver_preprocessed,
};
use std::{fmt::Debug, time::Duration};

pub trait NaiveSolvableGraph:
    Clone
    + AdjacencyList
    + GraphEdgeOrder
    + ColoredAdjacencyList
    + ColoredAdjacencyTest
    + GraphEdgeEditing
    + ColorFilter
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
        + ColorFilter
        + Debug
{
}

pub fn naive_solver<G: NaiveSolvableGraph>(input_graph: &G) -> (NumNodes, ContractionSequence) {
    naive_solver_with_bounds(input_graph, 0, input_graph.number_of_nodes())
        .expect("Could not produce feasable solution")
}

pub fn naive_solver_two_staged<G: NaiveSolvableGraph>(
    input_graph: &G,
    timeout: Duration,
) -> (NumNodes, ContractionSequence) {
    let mut graph = input_graph.clone();

    let mut contract_seq = ContractionSequence::new(graph.number_of_nodes());
    initial_pruning(&mut graph, &mut contract_seq);

    let (heuristic_tww, heuristic_seq, _) =
        timeout_monte_carlo_search_tree_solver_preprocessed(&graph, timeout);

    if heuristic_tww == 0 {
        return (0, heuristic_seq);
    }

    if let Some(sol) = naive_solver_with_bounds(input_graph, 0, heuristic_tww - 1) {
        return sol;
    }

    (heuristic_tww, heuristic_seq)
}

pub fn naive_solver_with_bounds<G: NaiveSolvableGraph>(
    input_graph: &G,
    slack: Node,
    not_above: Node,
) -> Option<(NumNodes, ContractionSequence)> {
    let mut graph = input_graph.clone();

    let (twin_width, mut contract_seq) = recurse(&mut graph, slack, not_above)?;

    contract_seq.add_unmerged_singletons(input_graph).unwrap();

    assert_eq!(
        twin_width,
        contract_seq
            .compute_twin_width(input_graph.clone())
            .unwrap()
    );

    Some((twin_width, contract_seq))
}

fn recurse<G: NaiveSolvableGraph>(
    graph: &mut G,
    slack: Node,
    mut not_above: Node,
) -> Option<(NumNodes, ContractionSequence)> {
    if slack > not_above {
        return None;
    }

    if graph.number_of_edges() == 0 {
        return Some((0, ContractionSequence::new(0)));
    }

    let red_deg_before = graph.red_degrees().max().unwrap();
    assert!(red_deg_before <= slack);

    let mut contract_seq = ContractionSequence::new(graph.number_of_nodes());

    prune_tiny_graph(graph, slack, &mut contract_seq);
    prune_leaves(graph, &mut contract_seq);
    prune_twins(graph, &mut contract_seq);
    //prune_black_ccs(graph, slack, &mut contract_seq);

    assert!(red_deg_before >= graph.red_degrees().max().unwrap());

    if graph.number_of_edges() == 0 {
        return Some((slack, contract_seq));
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

    let pairs = contraction_candidates(graph, not_above);
    if pairs.is_empty() {
        return None;
    };

    let mut best_solution = None;
    let mergable = {
        let mut mergable = BitSet::new(graph.number_of_nodes());

        if graph.degrees().all(|d| d == 0 || d == 2) {
            mergable.set_all();
        } else {
            for u in graph.vertices() {
                if graph.degree_of(u) == 2 && graph.red_degree_of(u) == 0 {
                    continue;
                }
                mergable.set_bit(u);
                mergable.set_bits(graph.neighbors_of(u).iter().copied());
            }
        }

        mergable
    };

    'outer: for &(r, (u, v)) in &pairs {
        assert!(graph.degree_of(u) > 0);
        assert!(graph.degree_of(v) > 0);
        assert_ne!(u, v);

        if r > not_above {
            break;
        }

        if !(mergable[u] && mergable[v]) {
            continue;
        }

        let mut local_graph = graph.clone();
        local_graph.merge_node_into(u, v);

        let max_red_degree = local_graph.red_degrees().max().unwrap();
        if max_red_degree > not_above {
            continue;
        }

        if let Some((sol_size, seq)) =
            recurse(&mut local_graph, slack.max(max_red_degree), not_above)
        {
            let sol_size = sol_size.max(max_red_degree);
            assert!(sol_size <= not_above);
            best_solution = Some(((u, v), sol_size, seq));

            if sol_size <= slack {
                break 'outer;
            }

            not_above = sol_size.checked_sub(1).unwrap();
        }
    }
    let ((u, v), tww, seq) = best_solution?;

    contract_seq.merge_node_into(u, v);
    contract_seq.append(&seq);
    Some((tww, contract_seq))
}

fn contraction_candidates<G: NaiveSolvableGraph>(
    graph: &G,
    not_above: u32,
) -> Vec<(u32, (u32, u32))> {
    let mut pairs = Vec::new();
    for u in graph.vertices() {
        let degree_u = graph.degree_of(u);
        if degree_u == 0 {
            continue;
        }

        let mut two_neighbors = graph.closed_two_neighborhood_of(u);
        for v in two_neighbors.iter().filter(|&v| v > u) {
            let mut red_neighs = graph.red_neighbors_after_merge(u, v, false);
            let mut red_card = red_neighs.cardinality();

            if red_card > not_above {
                continue;
            }

            for &x in graph.red_neighbors_of(u) {
                red_neighs.unset_bit(x);
            }

            for &x in graph.red_neighbors_of(v) {
                red_neighs.unset_bit(x);
            }

            for new_red in red_neighs.iter() {
                red_card = red_card.max(graph.red_degree_of(new_red) + 1);
            }

            if red_neighs.cardinality() <= not_above {
                pairs.push((red_neighs.cardinality(), (u, v)));
            }
        }

        if degree_u > not_above {
            continue;
        }

        let distant_nodes = {
            two_neighbors.not();
            two_neighbors
        };

        let red_deg_of_black_neighbors = graph
            .black_neighbors_of(u)
            .iter()
            .map(|&v| graph.red_degree_of(v) + 1)
            .max()
            .unwrap_or(0);

        if red_deg_of_black_neighbors > not_above {
            continue;
        }

        for v in distant_nodes.iter().filter(|&v| v > u) {
            let degree_v = graph.degree_of(v);
            let red_degree = red_deg_of_black_neighbors.max(degree_u + degree_v);
            if degree_v > 0
                && red_degree <= not_above
                && graph
                    .black_neighbors_of(v)
                    .iter()
                    .all(|&w| graph.red_degree_of(w) < not_above)
            {
                pairs.push((red_degree, (u, v)));
            }
        }
    }
    pairs.sort();
    pairs
}

fn try_split_into_cc<G: NaiveSolvableGraph>(
    graph: &mut G,
    slack: u32,
    not_above: u32,
) -> Option<Option<(NumNodes, ContractionSequence)>> {
    let part = graph.partition_into_connected_components(true);
    if part.number_of_classes() == 1 && 10 * part.number_of_unassigned() < graph.number_of_nodes() {
        return None;
    }

    let mut max_tww = 0;
    let mut slack = slack;

    let mut contract_seq = ContractionSequence::new(graph.number_of_nodes());

    for (mut subgraph, mapper) in part.split_into_subgraphs(graph) {
        if let Some((size, sol)) = recurse(&mut subgraph, slack, not_above) {
            slack = slack.max(size);
            max_tww = max_tww.max(size);
            for &(rem, sur) in sol.merges() {
                contract_seq.merge_node_into(
                    mapper.old_id_of(rem).unwrap(),
                    mapper.old_id_of(sur).unwrap(),
                );
            }
        } else {
            return Some(None);
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
