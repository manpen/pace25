use super::{
    exact::naive::naive_solver,
    graph::*,
    prelude::*,
    reduction::{Reducer, ReductionRule},
};
use itertools::Itertools as _;
use rand::Rng;

pub fn generate_random_graph_stream(
    rng: &mut impl Rng,
    n: NumNodes,
) -> impl Iterator<Item = (AdjArray, BitSet, BitSet)> {
    (0..).filter_map(move |i| {
        let graph = AdjArray::random_gnp(rng, n, 3. / n as f64);

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

        for u in graph.vertices() {
            if graph.degree_of(u) == 0 {
                covered.set_bit(u);
            }
        }

        Some((graph, covered, redundant))
    })
}

pub fn test_before_and_after_rule<C: FnMut(&AdjArray) -> R, R: ReductionRule<AdjArray>>(
    rng: &mut impl Rng,
    mut rule_cons: C,
    nodes: NumNodes,
    attempts: u32,
) {
    let mut num_applicable = 0;
    let max_attempts = attempts * 50;
    for (mut graph, mut covered, mut never_select) in
        generate_random_graph_stream(rng, nodes).take(max_attempts as usize)
    {
        let org_graph = graph.clone();
        let org_covered = covered.clone();
        let org_never_select = never_select.clone();
        let naive = naive_solver(&graph, &covered, &never_select, None, None).unwrap();

        let mut domset_after_rule = DominatingSet::new(graph.number_of_nodes());
        let mut reducer = Reducer::new();
        reducer.remove_unnecessary_edges(
            &mut graph,
            &mut domset_after_rule,
            &mut covered,
            &mut never_select,
        );
        let mut rule = rule_cons(&graph);

        let changed = reducer.apply_rule(
            &mut rule,
            &mut graph,
            &mut domset_after_rule,
            &mut covered,
            &mut never_select,
        );
        let domset_directly_after_rule = domset_after_rule.clone();
        num_applicable += changed as u32;

        let tmp = naive_solver(&graph, &covered, &never_select, None, None).unwrap();
        domset_after_rule.add_nodes(tmp.iter());

        assert!(domset_after_rule.is_valid_given_previous_cover(&graph, &covered));
        assert_eq!(
            naive.len(),
            domset_after_rule.len(),
            "naive: {:?}, naive after_rule: {:?}, directly after_rule: {:?} \n org:   {org_graph:?}, cov: {:?}, red: {:?} \n after: {graph:?}, cov: {:?}, red: {:?}",
            naive.iter().collect_vec(),
            domset_after_rule.iter().collect_vec(),
            domset_directly_after_rule.iter().collect_vec(),
            org_covered.iter_set_bits().collect_vec(),
            org_never_select.iter_set_bits().collect_vec(),
            covered.iter_set_bits().collect_vec(),
            never_select.iter_set_bits().collect_vec()
        );
        assert!(
            domset_after_rule.iter().all(|u| !never_select.get_bit(u)),
            "selected redundant node \n naive: {:?}, naive after_rule: {:?}, directly after_rule: {:?} \n org:   {org_graph:?}, cov: {:?}, red: {:?} \n after: {graph:?}, cov: {:?}, red: {:?}",
            naive.iter().collect_vec(),
            domset_after_rule.iter().collect_vec(),
            domset_directly_after_rule.iter().collect_vec(),
            org_covered.iter_set_bits().collect_vec(),
            org_never_select.iter_set_bits().collect_vec(),
            covered.iter_set_bits().collect_vec(),
            never_select.iter_set_bits().collect_vec()
        );

        if num_applicable > attempts {
            break;
        }
    }

    assert!(
        num_applicable >= attempts,
        "Abort test as rule was only applicable {num_applicable} times within {max_attempts} attempts"
    );
}
