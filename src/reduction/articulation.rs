use std::{marker::PhantomData, sync::Arc, time::Duration};

use itertools::Itertools;
#[allow(unused_imports)]
use log::{debug, info};

use crate::{
    exact::highs_advanced::{HighsCache, HighsDominatingSetSolver, SolverResult},
    graph::*,
};

use super::*;

const SOLVER_TIMEOUT: Duration = Duration::from_secs(1);
const MAX_CC_SIZE: Node = 200;

pub struct RuleArticulationPoint<G> {
    highs_cache: Option<Arc<HighsCache>>,
    _graph: PhantomData<G>,
}

impl<Graph: AdjacencyList + Clone + AdjacencyTest + 'static> RuleArticulationPoint<Graph> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            highs_cache: None,
            _graph: Default::default(),
        }
    }

    pub fn new_with_cache(cache: Arc<HighsCache>) -> Self {
        Self {
            highs_cache: Some(cache),
            _graph: Default::default(),
        }
    }
}

impl<Graph: AdjacencyList + Clone + AdjacencyTest + 'static> ReductionRule<Graph>
    for RuleArticulationPoint<Graph>
{
    const NAME: &str = "RuleArticulationPoint";

    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        if never_select.are_all_unset() {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        let mut aps = graph.compute_articulation_points();
        if aps.are_all_unset() {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        {
            let mut aps_list = aps
                .iter_set_bits()
                .map(|u| (graph.number_of_nodes() - graph.degree_of(u), u))
                .collect_vec();

            aps_list.sort_unstable();

            for (_, u) in aps_list {
                if graph.neighbors_of(u).any(|v| aps.get_bit(v)) {
                    aps.clear_bit(u);
                }
            }
        }

        let mut solver = HighsDominatingSetSolver::new(graph.number_of_nodes());
        if let Some(cache) = &self.highs_cache {
            solver.register_cache(cache.clone());
        }

        debug_assert!(domset.iter().all(|u| covered.get_bit(u)));

        let mut changed = false;
        for u in aps.iter_set_bits() {
            if never_select.get_bit(u) {
                if Self::process_small_ccs_at_red_ap(
                    graph,
                    domset,
                    covered,
                    never_select,
                    &mut solver,
                    u,
                ) {
                    changed = true;
                    continue;
                }
            } else if Self::process_small_ccs_at_ap(
                graph,
                domset,
                covered,
                never_select,
                &mut solver,
                u,
            ) {
                changed = true;
                continue;
            }
        }

        debug_assert!(domset.iter().all(|u| covered.get_bit(u)));

        (changed, None::<Box<dyn Postprocessor<Graph>>>)
    }
}

impl<Graph: AdjacencyList + Clone + AdjacencyTest + 'static> RuleArticulationPoint<Graph> {
    #[allow(unreachable_code, unused)]
    fn process_small_ccs_at_ap(
        graph: &mut Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &BitSet,
        solver: &mut HighsDominatingSetSolver,
        art_point: u32,
    ) -> bool {
        let mut restore_u_covered_to = covered.set_bit(art_point);

        let mut changed = false;
        for v in graph.neighbors_of(art_point) {
            let mut search = DFS::new(graph, v);
            search.exclude_node(art_point);

            let mut cc = search.take(1 + MAX_CC_SIZE as usize).collect_vec();
            if cc.len() > MAX_CC_SIZE as usize {
                continue;
            }
            cc.push(art_point);

            let eps = 0.3 / cc.len() as f64;
            let eps_n = eps / (1 + graph.degree_of(art_point)) as f64;

            let problem = solver.build_problem_of_subgraph(
                graph,
                covered,
                never_select,
                cc.as_slice(),
                |w| {
                    if w == art_point {
                        1.0 - eps
                    } else if graph.has_edge(w, art_point) {
                        1.0 - eps_n
                    } else {
                        1.0
                    }
                },
            );

            if let SolverResult::Optimal(solved) = problem.solve_exact(Some(SOLVER_TIMEOUT)) {
                changed |= !solved.is_empty();
                domset.add_nodes(solved.iter().cloned());
                covered.set_bits(cc.into_iter());

                if domset.is_in_domset(art_point) {
                    covered.set_bits(graph.neighbors_of(art_point));
                    restore_u_covered_to = true;
                } else {
                    restore_u_covered_to |= graph
                        .neighbors_of(art_point)
                        .any(|v: u32| domset.is_in_domset(v));
                }

                break;
            }
        }

        if !restore_u_covered_to {
            covered.clear_bit(art_point);
        }

        changed
    }

    fn process_small_ccs_at_red_ap(
        graph: &mut Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &BitSet,
        solver: &mut HighsDominatingSetSolver,
        art_point: u32,
    ) -> bool {
        // we only support uncovered redundant (those nodes are dealt with by the reducer)
        if covered.set_bit(art_point) {
            return false;
        }

        let mut restore_u_covered_to = false;
        let mut changed = false;
        for v in graph.neighbors_of(art_point) {
            let mut search = DFS::new(graph, v);
            search.exclude_node(art_point);

            let mut cc = search.take(1 + MAX_CC_SIZE as usize).collect_vec();
            if cc.len() > MAX_CC_SIZE as usize {
                continue;
            }

            cc.push(art_point);

            let eps = 0.3 / graph.degree_of(art_point) as f64;
            let problem = solver.build_problem_of_subgraph(
                graph,
                covered,
                never_select,
                cc.as_slice(),
                |w| {
                    if graph.has_edge(w, art_point) {
                        1.0 - eps
                    } else {
                        1.0
                    }
                },
            );

            if let SolverResult::Optimal(solved) = problem.solve_exact(Some(SOLVER_TIMEOUT))
                && !solved.is_empty()
            {
                debug!("Solved CC at {art_point} with nodes {cc:?}");

                changed = true;
                domset.add_nodes(solved.iter().cloned());

                for &w in &solved {
                    covered.set_bits(graph.closed_neighbors_of(w));
                }
                debug_assert!(cc.iter().all(|&w| covered.get_bit(w)));
                debug_assert!(!domset.is_in_domset(art_point));

                restore_u_covered_to |= graph
                    .neighbors_of(art_point)
                    .any(|v: u32| domset.is_in_domset(v));

                break;
            }
        }

        if !restore_u_covered_to {
            covered.clear_bit(art_point);
        }

        changed
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use rand::{Rng, SeedableRng, seq::IteratorRandom};
    use rand_pcg::Pcg64Mcg;

    use crate::{
        exact::naive::naive_solver,
        graph::{AdjArray, GnpGenerator, GraphNodeOrder},
    };

    fn generate_random_graphs(
        n: NumNodes,
    ) -> impl Iterator<Item = (AdjArray, Node, BitSet, BitSet)> {
        let mut rng = Pcg64Mcg::seed_from_u64(0x1234567);

        (0..).filter_map(move |i| {
            let mut graph = AdjArray::random_gnp(&mut rng, n, 4. / n as f64);
            let n1 = rng.gen_range(3..n / 2);

            // we define the first entry in here as the ap
            let node_in_part0 = graph
                .vertices_range()
                .choose_multiple(&mut rng, n1 as usize);

            if graph.degree_of(node_in_part0[0]) < 2 {
                return None;
            }

            let edges_to_remove = graph
                .edges(true)
                .filter(|&Edge(u, v)| {
                    u != node_in_part0[0]
                        && v != node_in_part0[1]
                        && node_in_part0.contains(&u) != node_in_part0.contains(&v)
                })
                .collect_vec();
            graph.remove_edges(edges_to_remove.into_iter());

            let aps = graph.compute_articulation_points();
            if aps.cardinality() != 1 || !aps.get_bit(node_in_part0[0]) {
                return None;
            }

            let mut covered = graph.vertex_bitset_unset();
            for _ in 0..i % 7 {
                covered.set_bit(rng.gen_range(graph.vertices_range()));
            }
            let mut redundant = graph.vertex_bitset_unset();
            for _ in 0..i % 5 {
                redundant.set_bit(rng.gen_range(graph.vertices_range()));
            }
            redundant -= &covered;

            {
                // reject if infeasible
                let mut tmp = DominatingSet::new(graph.number_of_nodes());
                tmp.add_nodes(redundant.iter_cleared_bits());
                if !tmp.is_valid_given_previous_cover(&graph, &covered) {
                    return None;
                }
            }

            Some((graph, node_in_part0[0], covered, redundant))
        })
    }

    fn apply_rule(
        graph: &mut AdjArray,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> DominatingSet {
        let mut domset = DominatingSet::new(graph.number_of_nodes());
        let mut reducer = Reducer::new();
        let mut rule_ap = RuleArticulationPoint::new();

        reducer.apply_rule(&mut rule_ap, graph, &mut domset, covered, redundant);

        let tmp = naive_solver(&*graph, &*covered, &*redundant, None, None).unwrap();
        domset.add_nodes(tmp.iter());
        domset
    }

    #[test]
    fn random_gnps_redundant_ap() {
        const NODES: NumNodes = 16;

        for (mut graph, ap, mut covered, mut redundant) in generate_random_graphs(NODES).take(1000)
        {
            covered.clear_bit(ap);
            redundant.set_bit(ap);
            redundant.clear_bits(graph.neighbors_of(ap));

            let naive = naive_solver(&graph, &covered, &redundant, None, None).unwrap();

            let after_rule = apply_rule(&mut graph, &mut covered, &mut redundant);

            assert!(after_rule.is_valid_given_previous_cover(&graph, &covered));
            assert_eq!(
                naive.len(),
                after_rule.len(),
                "naive: {:?}, after_rule: {:?}",
                naive.iter().collect_vec(),
                after_rule.iter().collect_vec()
            );
            assert!(after_rule.iter().all(|u| !redundant.get_bit(u)));
        }
    }

    #[test]
    fn random_gnps_non_redundant_ap() {
        const NODES: NumNodes = 16;

        for (i, (mut graph, ap, mut covered, mut redundant)) in
            generate_random_graphs(NODES).take(1000).enumerate()
        {
            covered.assign_bit(ap, i % 2 == 0);
            redundant.clear_bit(ap);

            let naive = naive_solver(&graph, &covered, &redundant, None, None).unwrap();

            let after_rule = apply_rule(&mut graph, &mut covered, &mut redundant);

            assert!(after_rule.is_valid_given_previous_cover(&graph, &covered));
            assert_eq!(
                naive.len(),
                after_rule.len(),
                "naive: {:?}, after_rule: {:?}",
                naive.iter().collect_vec(),
                after_rule.iter().collect_vec()
            );
            assert!(after_rule.iter().all(|u| !redundant.get_bit(u)));
        }
    }
}
