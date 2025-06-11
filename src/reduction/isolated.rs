use crate::graph::AdjacencyList;

use super::*;

pub struct RuleIsolatedReduction;

impl<Graph: AdjacencyList + 'static> ReductionRule<Graph> for RuleIsolatedReduction {
    const NAME: &str = "RuleIsolated";

    fn apply_rule(
        &mut self,
        graph: &mut Graph,
        domset: &mut DominatingSet,
        covered: &mut BitSet,
        never_select: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>) {
        if never_select.cardinality() == 0 {
            return (false, None::<Box<dyn Postprocessor<Graph>>>);
        }

        let mut changed = false;

        for u in never_select.iter_cleared_bits() {
            if covered.get_bit(u) || domset.is_in_domset(u) {
                continue;
            }

            if graph.neighbors_of(u).all(|v| never_select.get_bit(v)) {
                domset.add_node(u);
                covered.set_bits(graph.closed_neighbors_of(u));
                changed = true;
            }
        }

        (changed, None::<Box<dyn Postprocessor<Graph>>>)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::NumNodes;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    #[test]
    fn generic_before_and_after() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x1235342);
        const NODES: NumNodes = 20;
        crate::testing::test_before_and_after_rule(
            &mut rng,
            |_| RuleIsolatedReduction,
            false,
            NODES,
            400,
        );
    }

    #[test]
    fn generic_before_and_after_exhaust() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x12353742);
        const NODES: NumNodes = 20;
        crate::testing::test_before_and_after_rule(
            &mut rng,
            |_| RuleIsolatedReduction,
            true,
            NODES,
            400,
        );
    }
}
