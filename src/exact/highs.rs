use std::time::Duration;

use highs::{HighsModelStatus, Model, RowProblem};
use itertools::Itertools;

use crate::{graph::*, utils::DominatingSet};

pub fn highs_solver<G: Clone + AdjacencyList + GraphEdgeEditing>(
    graph: &G,
    is_perm_covered: &BitSet,
    never_select: &BitSet,
    _upper_bound_incl: Option<NumNodes>,
    timeout: Option<Duration>,
) -> DominatingSet {
    // TODO: RowProblems seems to get converted to a ColProblem --- so encode it directly as such
    let mut pb = RowProblem::default();

    let vars = graph
        .vertices_range()
        .map(|_| pb.add_integer_column(1.0, 0..1))
        .collect_vec();

    for u in graph.vertices_range() {
        if is_perm_covered.get_bit(u) {
            continue;
        }

        let coverable_by = graph
            .closed_neighbors_of(u)
            .filter(|&v| !never_select.get_bit(v))
            .map(|v| (vars[v as usize], 1.0));

        pb.add_row(1.., coverable_by);
    }

    let mut model = Model::new(pb);
    model.make_quiet();
    if let Some(tme) = timeout {
        model.set_option("time_limit", tme.as_secs_f64());
    }
    model.set_option("parallel", "off");
    model.set_option("threads", 1);
    model.set_sense(highs::Sense::Minimise);

    let solved = model.solve();
    assert_eq!(solved.status(), HighsModelStatus::Optimal);

    let solution = solved.get_solution();

    let mut domset = DominatingSet::new(graph.number_of_nodes());
    domset.add_nodes(
        solution
            .columns()
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v > 0.5).then_some(i as Node)),
    );

    debug_assert!(domset.is_valid_given_previous_cover(graph, is_perm_covered));

    domset
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
            let highs = highs_solver(&graph, &covered, &redundant, None, None);

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
