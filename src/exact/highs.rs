use core::panic;
use std::time::Duration;

use highs::{HighsModelStatus, Model, RowProblem};
use itertools::Itertools;
use log::info;

use crate::{graph::*, utils::DominatingSet};
const NOT_SET: Node = Node::MAX;

pub fn highs_solver<G: Clone + AdjacencyTest + AdjacencyList + GraphEdgeEditing>(
    graph: &G,
    is_perm_covered: &BitSet,
    never_select: &BitSet,
    upper_bound_incl: Option<NumNodes>,
    timeout: Option<Duration>,
) -> super::Result<DominatingSet> {
    // TODO: RowProblems seems to get converted to a ColProblem --- so encode it directly as such
    let mut pb = RowProblem::default();

    let mut skip_constraints_of = graph.vertex_bitset_unset();
    let mut skip_terms = 0;
    for u in never_select.iter_set_bits() {
        if graph.degree_of(u) != 2 {
            continue;
        }

        if let Some((a, b)) = graph
            .neighbors_of(u)
            .filter(|&v| !never_select.get_bit(v))
            .collect_tuple()
            && graph.has_edge(a, b)
        {
            for x in [a, b] {
                if !skip_constraints_of.set_bit(x) {
                    skip_terms += graph
                        .closed_neighbors_of(x)
                        .filter(|&v| !never_select.get_bit(v))
                        .count();
                }
            }
        }
    }
    info!(
        "Skip constraints of {} nodes with {skip_terms} edges",
        skip_constraints_of.cardinality()
    );

    let vars = (0..(graph.number_of_nodes() - never_select.cardinality()))
        .map(|_| pb.add_integer_column(1.0, 0..1))
        .collect_vec();

    let mut old_to_new = vec![NOT_SET; graph.len()];
    let mut new_to_old = Vec::with_capacity(vars.len());

    for old in never_select.iter_cleared_bits() {
        old_to_new[old as usize] = new_to_old.len() as Node;
        new_to_old.push(old);
    }

    let mut num_terms = 0;
    for u in graph.vertices_range() {
        if is_perm_covered.get_bit(u) || skip_constraints_of.get_bit(u) {
            continue;
        }

        let coverable_by = graph
            .closed_neighbors_of(u)
            .filter(|&v| !never_select.get_bit(v))
            .map(|v| {
                num_terms += 1;
                (vars[old_to_new[v as usize] as usize], 1.0)
            });

        pb.add_row(1.., coverable_by);
    }
    info!(
        "Remaining constraints: {}, Remaining terms: {num_terms}, Remaining vars: {}",
        pb.num_rows(),
        pb.num_cols()
    );

    let mut model = Model::new(pb);
    model.make_quiet();
    if let Some(tme) = timeout {
        model.set_option("time_limit", tme.as_secs_f64());
    }
    model.set_option("parallel", "off");
    model.set_sense(highs::Sense::Minimise);

    let solved = model.solve();
    match solved.status() {
        HighsModelStatus::Optimal => {}
        HighsModelStatus::Infeasible => return Err(crate::exact::ExactError::Infeasible),
        HighsModelStatus::ReachedTimeLimit => return Err(crate::exact::ExactError::Timeout),
        e => panic!("Unhandled HighsStatus: {e:?}"),
    };

    assert_eq!(solved.status(), HighsModelStatus::Optimal);

    let solution = solved.get_solution();

    let mut domset = DominatingSet::new(graph.number_of_nodes());
    domset.add_nodes(
        solution
            .columns()
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v > 0.5).then_some(new_to_old[i])),
    );

    if upper_bound_incl.is_some_and(|b| b < domset.len() as NumNodes) {
        return Err(crate::exact::ExactError::Infeasible);
    }

    debug_assert!(domset.is_valid_given_previous_cover(graph, is_perm_covered));

    Ok(domset)
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    use crate::{
        exact::naive::naive_solver,
        graph::{AdjArray, GnpGenerator, GraphNodeOrder},
    };

    #[test]
    fn cross_with_naive() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x1234567);
        const NODES: NumNodes = 24;

        let mut remaining_graphs = 300;

        let mut duration_naive = 0.0;
        let mut duration_highs = 0.0;

        loop {
            let graph = AdjArray::random_gnp(&mut rng, NODES, 3. / NODES as f64);

            let mut covered = graph.vertex_bitset_unset();
            for _ in 0..remaining_graphs % 7 {
                covered.set_bit(rng.gen_range(graph.vertices_range()));
            }
            let mut redundant = graph.vertex_bitset_unset();
            for _ in 0..remaining_graphs % 5 {
                redundant.set_bit(rng.gen_range(graph.vertices_range()));
            }
            redundant -= &covered;

            {
                // reject if infeasible
                let mut tmp = DominatingSet::new(graph.number_of_nodes());
                tmp.add_nodes(redundant.iter_cleared_bits());
                if !tmp.is_valid_given_previous_cover(&graph, &covered) {
                    continue;
                }
            }

            let start_naive = Instant::now();
            let naive = naive_solver(&graph, &covered, &redundant, None, None).unwrap();

            let start_highs = Instant::now();
            let highs = highs_solver(&graph, &covered, &redundant, None, None).unwrap();

            duration_highs += start_highs.elapsed().as_secs_f64();
            duration_naive += start_highs.duration_since(start_naive).as_secs_f64();

            assert!(highs.is_valid_given_previous_cover(&graph, &covered));
            assert_eq!(naive.len(), highs.len());
            assert!(highs.iter().all(|u| !redundant.get_bit(u)));

            remaining_graphs -= 1;
            if remaining_graphs == 0 {
                break;
            }
        }

        // usually highs is much faster, so this assertion should be safe
        assert!(
            duration_highs < duration_naive,
            "Highs: {duration_highs}, Naive: {duration_naive}"
        );
    }
}
