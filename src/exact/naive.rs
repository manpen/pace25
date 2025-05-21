use std::time::{Duration, Instant};

use itertools::Itertools;
#[allow(unused_imports)]
use log::{info, trace};
use thiserror::Error;

use crate::prelude::*;

#[derive(Debug, PartialEq, PartialOrd, Error)]
pub enum ExactError {
    #[error("upper bound infeasible")]
    Infeasible,
    #[error("timeout")]
    Timeout,
}

pub type Result<T> = std::result::Result<T, ExactError>;

pub fn naive_solver<G: Clone + AdjacencyList + GraphEdgeEditing>(
    graph: &G,
    is_perm_covered: &BitSet,
    never_select: &BitSet,
    upper_bound_incl: Option<NumNodes>,
    timeout: Option<Duration>,
) -> Result<DominatingSet> {
    if is_perm_covered.are_all_set() {
        return Ok(DominatingSet::new(graph.number_of_nodes()));
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

    let finish_until = timeout.map(|t| Instant::now() + t);
    let size = naive_solver_impl(
        graph,
        &mut best_ds,
        &mut working_ds,
        candidates.as_slice(),
        is_perm_covered,
        upper_bound_incl.unwrap_or(candidates.len() as NumNodes),
        finish_until,
    )?;

    let best_ds = best_ds.unwrap();

    assert_eq!(size, best_ds.len() as NumNodes);

    let mut ds = DominatingSet::new(graph.number_of_nodes());
    ds.add_nodes(best_ds);
    Ok(ds)
}

fn naive_solver_impl<G: Clone + AdjacencyList + GraphEdgeEditing>(
    graph: &G,
    best_domset: &mut Option<Vec<Node>>,
    work_domset: &mut Vec<Node>,
    candidates: &[Node],
    covered: &BitSet,
    mut upper_bound_incl: NumNodes,
    finish_until: Option<Instant>,
) -> Result<NumNodes> {
    if covered.are_all_set() {
        // solved
        assert!(work_domset.len() <= upper_bound_incl as usize);
        *best_domset = Some(work_domset.clone());
        return Ok(work_domset.len() as NumNodes);
    }

    if candidates.is_empty() || upper_bound_incl <= work_domset.len() as NumNodes {
        return Err(ExactError::Infeasible);
    }

    if covered.cardinality() + 1 == graph.number_of_nodes() {
        let uncovered = covered.iter_cleared_bits().next().unwrap();

        let cand = *candidates
            .iter()
            .find(|&&c| graph.closed_neighbors_of(c).contains(&uncovered))
            .unwrap();

        work_domset.push(cand);
        assert!(work_domset.len() <= upper_bound_incl as usize);
        *best_domset = Some(work_domset.clone());
        work_domset.pop();
        return Ok(1 + work_domset.len() as NumNodes);
    }

    if work_domset.len() + 1 == upper_bound_incl as usize {
        let num_uncovered = graph.number_of_nodes() - covered.cardinality();
        let cand = candidates.iter().copied().find(|&c| {
            graph
                .closed_neighbors_of(c)
                .filter(|&v| !covered.get_bit(v))
                .count()
                == num_uncovered as usize
        });

        return if let Some(cand) = cand {
            work_domset.push(cand);
            assert!(work_domset.len() <= upper_bound_incl as usize);
            *best_domset = Some(work_domset.clone());
            work_domset.pop();
            Ok(1 + work_domset.len() as NumNodes)
        } else {
            Err(ExactError::Infeasible)
        };
    }

    if let Some(finish_until) = finish_until
        && candidates.len() > 5
    {
        let now = Instant::now();
        if now > finish_until {
            return Err(ExactError::Timeout);
        }
    }

    let (candidate, candidates) = candidates.split_last().unwrap();

    let mut covered_with = covered.clone();
    covered_with.set_bits(graph.closed_neighbors_of(*candidate));

    let size_with = if covered_with.cardinality() != covered.cardinality() {
        work_domset.push(*candidate);
        let size_with = naive_solver_impl(
            graph,
            best_domset,
            work_domset,
            candidates,
            &covered_with,
            upper_bound_incl,
            finish_until,
        );
        work_domset.pop();

        match size_with.as_ref() {
            Ok(x) => {
                if *x == 1 {
                    return Ok(1);
                }
                upper_bound_incl = x - 1;
            }
            Err(ExactError::Timeout) => return Err(ExactError::Timeout),
            _ => {}
        }

        Some(size_with)
    } else {
        None
    };

    let size_without = naive_solver_impl(
        graph,
        best_domset,
        work_domset,
        candidates,
        covered,
        upper_bound_incl,
        finish_until,
    );

    match (size_without, size_with) {
        (Ok(res), _) => Ok(res),
        (Err(e), None) => Err(e),
        (Err(_), Some(x)) => x,
    }
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
                None,
            )
            .unwrap(); // since we do not give an upper bound, there is a solution!

            assert!(solution.is_valid(&graph));
        }
    }
}
