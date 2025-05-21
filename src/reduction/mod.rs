pub mod long_path;
pub use long_path::LongPathReduction;
pub mod rule_one_reduction;
pub use rule_one_reduction::RuleOneReduction;
pub mod small_exact;
pub use small_exact::RuleSmallExactReduction;
pub mod reducer;
pub use reducer::Reducer;
pub mod subset;
pub use subset::RuleSubsetReduction;

use crate::{graph::BitSet, utils::DominatingSet};

pub trait ReductionRule<Graph> {
    const NAME: &str;

    /// Applies the rule to the given graph and a partial solution.
    /// The rule may modify either. The first value return is true
    /// if the rule modified the graph; the second is Some(..) if
    /// the rule needs a post processing step.
    fn apply_rule(
        graph: &mut Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
        redundant: &mut BitSet,
    ) -> (bool, Option<Box<dyn Postprocessor<Graph>>>);
}

pub trait Postprocessor<Graph> {
    /// A post-processing step is typically needed, if the rule introduced
    /// gadgets, that need to be expanded once a full solution was
    /// computed by the solver.
    /// In general, post_processing needs to happen in the reverse order
    /// the rules where applied.
    fn post_process(
        &mut self,
        graph: &mut Graph,
        solution: &mut DominatingSet,
        covered: &mut BitSet,
    );
}
