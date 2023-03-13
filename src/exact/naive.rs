use std::ops::Neg;

use itertools::Itertools;
use log::trace;

use super::*;
use crate::prelude::*;

type SolverResultCache = crate::utils::ResultCache<digest::Output<sha2::Sha256>>;

pub fn naive_solver<G: FullfledgedGraph>(input_graph: &G) -> (NumNodes, ContractionSequence) {
    naive_solver_with_bounds(input_graph, 0, input_graph.number_of_nodes())
        .expect("Could not produce feasable solution")
}

pub fn naive_solver_two_staged<G: FullfledgedGraph>(
    input_graph: &G,
) -> (NumNodes, ContractionSequence) {
    let mut graph = input_graph.clone();

    let mut contract_seq = ContractionSequence::new(graph.number_of_nodes());
    initial_pruning(&mut graph, &mut contract_seq);

    let part = graph.partition_into_connected_components(true);
    let mut subgraphs = part.split_into_subgraphs(&graph);
    subgraphs.sort_by_key(|(g, _)| g.number_of_nodes());

    trace!("Number of CCs: {}", subgraphs.len());

    let mut slack = 0;
    for (subgraph, mapper) in subgraphs {
        let (tww, mut seq) = escalate_solvers(&subgraph, slack);
        contract_seq.append_mapped(&seq, &mapper);

        seq.add_unmerged_singletons(&subgraph).unwrap();

        slack = slack.max(tww);
    }

    contract_seq.add_unmerged_singletons(input_graph).unwrap();
    (slack, contract_seq)
}

fn escalate_solvers<G: FullfledgedGraph>(
    input_graph: &G,
    mut slack: NumNodes,
) -> (NumNodes, ContractionSequence) {
    let (heuristic_tww, heuristic_seq) = heuristic_solve(input_graph);

    if heuristic_tww <= slack {
        trace!("Accept heuristic solver");
        return (heuristic_tww, heuristic_seq);
    }

    // try to establish better lower bound using heuristic
    slack = slack.max(lower_bound_subgraph(input_graph, slack, heuristic_tww));
    if heuristic_tww <= slack {
        trace!("Accept heuristic solver after improving slack");
        return (heuristic_tww, heuristic_seq);
    }

    // invoke exact solver
    if let Some((exact_tww, exact_seq)) =
        naive_solver_with_bounds(input_graph, slack, heuristic_tww - 1)
    {
        trace!("Exact solver found better solution with exact_tww={exact_tww}, initial slack={slack} and heuristic_tww={heuristic_tww}");
        (exact_tww, exact_seq)
    } else {
        trace!("Exact solver proved heuristic was optimal");
        (heuristic_tww, heuristic_seq)
    }
}

pub fn naive_solver_with_bounds<G: FullfledgedGraph>(
    input_graph: &G,
    slack: Node,
    not_above: Node,
) -> Option<(NumNodes, ContractionSequence)> {
    let mut graph = input_graph.clone();

    let mut cache = SolverResultCache::default();

    try_split_into_cc(&mut cache, &mut graph, slack, not_above)
        .unwrap_or_else(|| recurse(&mut cache, &mut graph, slack, not_above))
}

fn recurse<G: FullfledgedGraph>(
    cache: &mut SolverResultCache,
    graph: &mut G,
    slack: Node,
    not_above: Node,
) -> Option<(NumNodes, ContractionSequence)> {
    trace!(
        "Recurse n={:>5} m={:>5} slack={slack:>5} not_above={not_above:>5}",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    );

    if graph.number_of_nodes() == 0 || graph.number_of_edges() == 0 {
        return Some((0, ContractionSequence::new(graph.number_of_nodes())));
    }

    if graph.degrees().any(|d| d == 0) {
        let (mut graph, mapper) = graph.remove_disconnected_verts();
        let (tww, mut seq) = recurse(cache, &mut graph, slack, not_above)?;
        seq.apply_mapper(&mapper);
        return Some((tww, seq));
    }

    let all_edges =
        (graph.number_of_nodes() as NumEdges) * (graph.number_of_nodes() as NumEdges - 1) / 2;
    let num_red_edges = graph.red_degrees().map(|d| d as NumEdges).sum::<NumEdges>() / 2;
    let num_black_edges = graph.number_of_edges() - num_red_edges;

    if graph.number_of_nodes() > 8 && all_edges - num_black_edges < num_black_edges {
        let mut complement = graph.trigraph_complement();
        assert!(complement.number_of_edges() < graph.number_of_edges());

        return recurse(cache, &mut complement, slack, not_above).map(|(tww, mut seq)| {
            seq.add_unmerged_singletons(&graph.trigraph_complement())
                .unwrap();
            (tww, seq)
        });
    }

    let hash = graph.binary_digest_sha256();

    if let Some(solution) = cache.get(&hash, slack, not_above) {
        return solution.cloned();
    }

    let result = recurse_impl(cache, graph, slack, not_above);

    cache.add_to_cache(hash, result.clone(), slack, not_above);

    result
}

type BestSolution = (Option<(Node, Node)>, NumNodes, ContractionSequence);

fn recurse_impl<G: FullfledgedGraph>(
    cache: &mut SolverResultCache,
    graph: &mut G,
    mut slack: Node,
    not_above: Node,
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

    default_pruning(graph, slack, &mut contract_seq);

    if graph.number_of_edges() == 0 {
        trace!("Left with empty kernel: {:?}", contract_seq.merges());
        return Some((slack, contract_seq));
    }

    let mut best_solution: Option<BestSolution> = None;
    // red bridges
    {
        let red_bridges: Vec<_> = graph
            .compute_colored_bridges()
            .into_iter()
            .filter(|&ColoredEdge(u, v, c)| {
                c.is_red() && graph.degree_of(u) > 1 && graph.degree_of(v) > 1
            })
            .collect();

        for ColoredEdge(u, v, _) in red_bridges {
            graph.remove_edge(u, v);
            let part = graph.partition_into_connected_components(true);
            graph.add_edge(u, v, EdgeColor::Red);

            if part.number_of_classes() != 2 {
                println!("Num Classes: {}", part.number_of_classes());
                continue;
            }

            if part.number_in_class(0).min(part.number_in_class(1)) * 3
                < part.number_in_class(0) + part.number_in_class(1)
            {
                continue;
            }

            for (easy_node, hard_node, easy_class) in [
                (u, v, part.class_of_node(u).unwrap()),
                (v, u, part.class_of_node(v).unwrap()),
            ] {
                let ub = best_solution
                    .as_ref()
                    .map_or(not_above, |(_, tww, _)| *tww - 1);

                let mut easy_nodes = BitSet::new_all_unset_but(
                    graph.number_of_nodes(),
                    part.members_of_class(easy_class),
                );

                let mut easy_graph: G = graph.sub_graph(&easy_nodes);
                let easy_slack = slack
                    .saturating_sub(1)
                    .max(easy_graph.red_degrees().max().unwrap());

                let (easy_tww, mut easy_sol) =
                    if let Some(x) = recurse(cache, &mut easy_graph, easy_slack, ub) {
                        x
                    } else if ub == not_above {
                        return None;
                    } else {
                        continue;
                    };

                let mut hard_nodes = {
                    easy_nodes.not();
                    easy_nodes
                };

                let mut hard_graph: G = graph.sub_graph(&hard_nodes);
                let remain_in_easy = easy_sol.merges().last().unwrap().1;
                hard_graph.add_edge(remain_in_easy, hard_node, EdgeColor::Red);

                let (hard_tww, hard_sol) =
                    if let Some(x) = recurse(cache, &mut hard_graph, slack, ub) {
                        x
                    } else if ub == not_above {
                        return None;
                    } else {
                        continue;
                    };

                let total_tww = (easy_tww + 1).max(hard_tww);

                if hard_tww < easy_tww {
                    println!("Invert");

                    let easy_nodes = {
                        hard_nodes.not();
                        hard_nodes
                    };

                    let mut easy_graph: G = graph.sub_graph(&easy_nodes);
                    easy_graph.add_edge(easy_node, hard_node, EdgeColor::Red);
                    let easy_slack = slack
                        .saturating_sub(1)
                        .max(easy_graph.red_degrees().max().unwrap());

                    let (easy_tww, easy_sol) =
                        if let Some(x) = recurse(cache, &mut easy_graph, easy_tww, not_above) {
                            x
                        } else if ub == not_above {
                            return None;
                        } else {
                            continue;
                        };

                    if easy_tww > hard_tww {
                        println!("Still");
                    }
                }

                if easy_tww < hard_tww || total_tww <= slack {
                    contract_seq.append(&easy_sol);
                    contract_seq.append(&hard_sol);
                    return Some((hard_tww, contract_seq));
                } else if total_tww < ub {
                    easy_sol.append(&hard_sol);
                    best_solution = Some((None, total_tww, easy_sol));
                    slack = slack.max(hard_tww).max(easy_tww);
                }

                break;
            }
        }
    }

    assert!(graph.red_degrees().max().unwrap() <= not_above);

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

    let pairs = contraction_candidates(graph, &mut contract_seq, &mergable, not_above);

    trace!(
        " mergeable: {:>5} pairs: {:>8}",
        mergable.cardinality(),
        pairs.len()
    );

    if graph.number_of_edges() == 0 {
        // contract_candidates may prune, so we've to check again
        return Some((slack, contract_seq));
    }

    if pairs.is_empty() {
        return None;
    };

    let mut not_above = best_solution
        .as_ref()
        .map_or(not_above, |(_, tww, _)| *tww - 1);

    'outer: for &(r, (u, v)) in &pairs {
        assert!(graph.degree_of(u) > 0);
        if graph.degree_of(v) == 0 {
            continue;
        }

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

        if let Some((sol_size, seq)) = recurse(
            cache,
            &mut local_graph,
            slack.max(max_red_degree),
            not_above,
        ) {
            let sol_size = sol_size.max(max_red_degree);
            assert!(sol_size <= not_above);
            best_solution = Some((Some((u, v)), sol_size, seq));

            if sol_size <= slack {
                break 'outer;
            }

            not_above = sol_size.checked_sub(1).unwrap();
        }
    }
    let (pair, tww, seq) = best_solution?;

    if let Some((u, v)) = pair {
        contract_seq.merge_node_into(u, v);
    }
    contract_seq.append(&seq);
    Some((tww, contract_seq))
}

fn contraction_candidates<G: FullfledgedGraph>(
    graph: &mut G,
    contract_seq: &mut ContractionSequence,
    mergeable: &BitSet,
    not_above: u32,
) -> Vec<(u32, (u32, u32))> {
    let mut pairs = Vec::new();
    let mut mergeable = mergeable.clone();

    let red_degs_of_black_neighbors = graph
        .vertices()
        .map(|u| {
            if !mergeable[u] {
                return not_above + 1;
            }

            graph
                .black_neighbors_of(u)
                .iter()
                .map(|&v| graph.red_degree_of(v) + 1)
                .max()
                .unwrap_or(0)
        })
        .collect_vec();

    for u in graph.vertices_range() {
        if !mergeable.unset_bit(u) {
            continue;
        }
        let degree_u = graph.degree_of(u);
        if degree_u == 0 {
            continue;
        }

        let mut two_neighbors = graph.closed_two_neighborhood_of(u);
        two_neighbors.and(&mergeable);
        for v in two_neighbors.iter() {
            assert!(v > u);
            let mut red_neighs = graph.red_neighbors_after_merge(u, v, false);
            let mut red_card = red_neighs.cardinality();

            if red_neighs.cardinality() == 0 {
                contract_seq.merge_node_into(v, u);
                graph.merge_node_into(v, u);
                assert_eq!(graph.red_degree_of(v), 0);
                continue;
            }

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
            let mut three_neighbors = BitSet::new(graph.number_of_nodes());
            for x in two_neighbors.iter() {
                three_neighbors.set_bits(graph.neighbors_of(x).iter().copied());
            }
            three_neighbors.and_not(&two_neighbors);
            three_neighbors.and(&mergeable);
            three_neighbors
        };

        let red_deg_of_black_neighbors = graph
            .black_neighbors_of(u)
            .iter()
            .map(|&v| graph.red_degree_of(v) + 1)
            .max()
            .unwrap_or(0);

        if red_degs_of_black_neighbors[u as usize] > not_above {
            continue;
        }

        for v in distant_nodes.iter() {
            assert!(v > u);
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

fn try_split_into_cc<G: FullfledgedGraph>(
    cache: &mut SolverResultCache,
    graph: &mut G,
    slack: u32,
    not_above: u32,
) -> Option<Option<(NumNodes, ContractionSequence)>> {
    let part = graph.partition_into_connected_components(true);
    if part.number_of_classes() == 1 && part.number_of_unassigned() == 0 {
        return None;
    }

    let mut max_tww = 0;
    let mut slack = slack;

    let mut contract_seq = ContractionSequence::new(graph.number_of_nodes());

    for (mut subgraph, mapper) in part.split_into_subgraphs(graph) {
        if let Some((size, sol)) = recurse(cache, &mut subgraph, slack, not_above) {
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
    use super::*;
    use crate::{log::build_pace_logger_for_level, testing::get_test_graphs_with_tww};
    use std::{fs::File, io::BufReader};

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
            if graph.number_of_nodes() > 15 {
                continue;
            }
            println!(" Test {filename}");
            let (tww, _seq) = naive_solver(&graph);
            assert_eq!(tww, presolved_tww, "file: {filename}");
        }
    }

    #[test]
    fn small_random_two_staged() {
        //build_pace_logger_for_level(log::LevelFilter::Trace);
        for (filename, graph, presolved_tww) in
            get_test_graphs_with_tww("instances/small-random/*.gr").step_by(3)
        {
            if graph.number_of_nodes() > 15 {
                continue;
            }
            println!(" Test {filename}");
            let (tww, _seq) = naive_solver_two_staged(&graph);
            assert_eq!(tww, presolved_tww, "file: {filename}");
        }
    }
}
