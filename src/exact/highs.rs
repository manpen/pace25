use core::panic;
use std::fmt::Debug;
use std::time::Duration;

use highs::{HighsModelStatus, Model, RowProblem};
use itertools::Itertools;

use crate::prelude::*;
use stream_bitset::prelude::{BitmaskStreamConsumer, ToBitmaskStream};

use crate::{exact::ExactError, graph::*, utils::DominatingSet};
const NOT_SET: Node = Node::MAX;

pub fn highs_solver<G: Clone + AdjacencyTest + AdjacencyList + Debug>(
    graph: &G,
    is_perm_covered: &BitSet,
    never_select: &BitSet,
    upper_bound_incl: Option<NumNodes>,
    timeout: Option<Duration>,
) -> super::Result<DominatingSet> {
    highs_solver_with_precious(
        graph,
        &[],
        is_perm_covered,
        never_select,
        upper_bound_incl,
        timeout,
    )
}

pub fn highs_solver_with_precious<G: Clone + AdjacencyTest + AdjacencyList + Debug>(
    graph: &G,
    precious: &[Node],
    is_perm_covered: &BitSet,
    never_select: &BitSet,
    upper_bound_incl: Option<NumNodes>,
    timeout: Option<Duration>,
) -> super::Result<DominatingSet> {
    // TODO: RowProblems seems to get converted to a ColProblem --- so encode it directly as such
    let mut pb = RowProblem::default();

    let mut skip_constraints_of = graph.vertex_bitset_unset();
    let mut skip_terms = 0;

    for u in (never_select.bitmask_stream() - is_perm_covered).iter_set_bits() {
        if let Some((a, b)) = graph
            .neighbors_of(u)
            .filter(|&v| v != u && !never_select.get_bit(v))
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
    debug!(
        "Skip constraints of {} nodes with {skip_terms} edges",
        skip_constraints_of.cardinality()
    );

    let mut vars =
        Vec::with_capacity((graph.number_of_nodes() - never_select.cardinality()) as usize);

    let mut old_to_new = vec![NOT_SET; graph.len()];
    let mut new_to_old = Vec::with_capacity(vars.len());

    let precious_weight = 1.0 - 1.0 / (1 + precious.len()) as f64;

    for old in never_select.iter_cleared_bits() {
        old_to_new[old as usize] = new_to_old.len() as Node;
        new_to_old.push(old);
        vars.push(pb.add_integer_column(
            if precious.contains(&old) {
                precious_weight
            } else {
                1.0
            },
            0..=1,
        ))
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
    debug!(
        "Remaining constraints: {}, Remaining terms: {num_terms}, Remaining vars: {}",
        pb.num_rows(),
        pb.num_cols()
    );

    let mut model = Model::new(pb);
    model.make_quiet();
    if let Some(tme) = timeout {
        model.set_option("time_limit", tme.as_secs_f64());
    }

    #[cfg(not(feature = "par"))]
    {
        model.set_option("parallel", "off");
        model.set_option("threads", "1");
    }
    model.set_sense(highs::Sense::Minimise);

    let solved = model.solve();
    let mut subopt = false;
    match solved.status() {
        HighsModelStatus::Optimal => {}
        HighsModelStatus::Infeasible => return Err(ExactError::Infeasible),
        HighsModelStatus::ReachedTimeLimit => {
            subopt = true;
        }
        e => panic!("Unhandled HighsStatus: {e:?}"),
    };

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
        return Err(ExactError::Infeasible);
    }

    if subopt {
        if domset.is_valid_given_previous_cover(graph, is_perm_covered) {
            return Err(ExactError::TimeoutWithSolution(domset));
        } else {
            return Err(ExactError::Timeout);
        }
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

            let naive = naive_solver(&graph, &covered, &redundant, None, None).unwrap();

            let highs = highs_solver(&graph, &covered, &redundant, None, None).unwrap();

            assert!(highs.is_valid_given_previous_cover(&graph, &covered));
            assert_eq!(naive.len(), highs.len());
            assert!(highs.iter().all(|u| !redundant.get_bit(u)));

            remaining_graphs -= 1;
            if remaining_graphs == 0 {
                break;
            }
        }
    }
}
