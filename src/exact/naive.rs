use itertools::Itertools;
#[allow(unused_imports)]
use log::{info, trace};

use crate::prelude::*;

pub fn naive_solver<G: Clone + AdjacencyList + GraphEdgeEditing>(
    graph: &G,
    is_perm_covered: &BitSet,
    never_select: &BitSet,
    upper_bound_incl: Option<NumNodes>,
) -> Option<DominatingSet> {
    if is_perm_covered.are_all_set() {
        return Some(DominatingSet::new(graph.number_of_nodes()));
    }

    let mut working_ds = Vec::with_capacity(graph.len());
    let mut best_ds = None;

    let mut candidates = graph
        .vertices_range()
        .filter(|&u| !never_select.get_bit(u))
        .map(|u| {
            (
                graph
                    .closed_neighbors_of(u)
                    .filter(|&v| !is_perm_covered.get_bit(v))
                    .count() as NumNodes,
                u,
            )
        })
        .collect_vec();
    candidates.sort_unstable();

    let candidates = candidates.into_iter().map(|(_, x)| x).collect_vec();

    let size = naive_solver_impl(
        graph,
        &mut best_ds,
        &mut working_ds,
        candidates.as_slice(),
        is_perm_covered,
        upper_bound_incl.unwrap_or_else(|| graph.number_of_nodes()),
    );

    if let Some(size) = size {
        assert_eq!(size, best_ds.as_ref().unwrap().len() as NumNodes);
    }

    best_ds.map(|x| {
        let mut ds = DominatingSet::new(graph.number_of_nodes());
        ds.add_nodes(x);
        ds
    })
}

fn naive_solver_impl<G: Clone + AdjacencyList + GraphEdgeEditing>(
    graph: &G,
    best_domset: &mut Option<Vec<Node>>,
    work_domset: &mut Vec<Node>,
    candidates: &[Node],
    covered: &BitSet,
    mut upper_bound_incl: NumNodes,
) -> Option<NumNodes> {
    if covered.are_all_set() {
        // solved
        assert!(work_domset.len() <= upper_bound_incl as usize);
        *best_domset = Some(work_domset.clone());
        return Some(work_domset.len() as NumNodes);
    }

    if candidates.is_empty() || upper_bound_incl <= work_domset.len() as NumNodes {
        return None;
    }

    let (candidate, candidates) = candidates.split_last().unwrap();

    let mut covered_with = covered.clone();
    covered_with.set_bits(graph.closed_neighbors_of(*candidate));

    work_domset.push(*candidate);
    let size_with = naive_solver_impl(
        graph,
        best_domset,
        work_domset,
        candidates,
        &covered_with,
        upper_bound_incl,
    );
    work_domset.pop();

    if let Some(x) = size_with.as_ref() {
        if upper_bound_incl == 1 {
            return None;
        }
        upper_bound_incl = x - 1;
    }

    let size_without = naive_solver_impl(
        graph,
        best_domset,
        work_domset,
        candidates,
        covered,
        upper_bound_incl,
    );

    if let (Some(w), Some(wo)) = (size_with, size_without) {
        assert!(wo < w);
    }

    size_without.or(size_with)
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    use crate::graph::{AdjArray, GnpGenerator, GraphNodeOrder};

    use super::naive_solver;

    #[test]
    fn id1582() {
        let graph = AdjArray::test_only_from([
            (1 - 1, 3 - 1),
            (1 - 1, 4 - 1),
            (1 - 1, 7 - 1),
            (2 - 1, 8 - 1),
            (3 - 1, 9 - 1),
            (4 - 1, 8 - 1),
            (4 - 1, 9 - 1),
            (5 - 1, 6 - 1),
        ]);

        let solution = naive_solver(
            &graph,
            &graph.vertex_bitset_unset(),
            &graph.vertex_bitset_unset(),
            None,
        )
        .unwrap();

        assert_eq!(solution.len(), 4);
    }

    #[test]
    fn randomized() {
        let mut rng = Pcg64Mcg::seed_from_u64(1234567);
        for _ in 0..100 {
            let graph = AdjArray::random_black_gnp(&mut rng, 20, 4. / 20.);

            let solution = naive_solver(
                &graph,
                &graph.vertex_bitset_unset(),
                &graph.vertex_bitset_unset(),
                None,
            )
            .unwrap(); // since we do not give an upper bound, there is a solution!

            assert!(solution.is_valid(&graph));
        }
    }
}
