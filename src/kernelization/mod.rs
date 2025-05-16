pub mod long_path;
pub mod rule1;
pub mod rule_one_reduction;
pub mod subsets;
pub use long_path::LongPathReduction;
pub use rule_one_reduction::RuleOneReduction;
pub use subsets::*;

use crate::{graph::BitSet, utils::DominatingSet};

/// Trait for kernelization rules
pub trait KernelizationRule<Graph> {
    /// Applies the rule to the given graph and a DominatingSet.
    /// Returns a BitSet indicating nodes that are guaranteed to be never in a DominatingSet.
    ///
    /// Depending on the rule, this will modify the graph and/or the DominatingSet.
    fn apply_rule(graph: Graph, sol: &mut DominatingSet) -> BitSet;
}

pub trait ReductionRule<Graph>: Sized {
    /// Applies the rule to the given graph and a partial solution.
    /// The rule may modify either. The first value return is true
    /// if the rule modified the graph; the second is Some(..) if
    /// the rule needs a post processing step.
    fn apply_rule(
        graph: &mut Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
    ) -> (bool, Option<Self>);

    /// A post-processing step is typically needed, if the rule introduced
    /// gadgets, that need to be expanded once a full solution was
    /// computed by the solver.
    /// In general, post_processing needs to happen in the reverse order
    /// the rules where applied.
    fn post_process(self, _solution: &mut DominatingSet, _covered: &mut BitSet) {
        unreachable!("This is a single stage rule without post processing"); // most rules are
    }
}
