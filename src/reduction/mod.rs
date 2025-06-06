pub mod long_path;
pub use long_path::LongPathReduction;
pub mod rule_one_reduction;
pub use rule_one_reduction::RuleOneReduction;
pub mod two_hop_rule_two_reduction;
pub use two_hop_rule_two_reduction::TwoHopRuleTwoReduction;
pub mod small_exact;
pub use small_exact::RuleSmallExactReduction;
pub mod reducer;
pub use reducer::Reducer;
pub mod subset;
pub use subset::RuleSubsetReduction;
pub mod isolated;
pub use isolated::RuleIsolatedReduction;
pub mod vertex_cover;
pub use vertex_cover::RuleVertexCover;
pub mod red_cover;
pub use red_cover::RuleRedundantCover;
pub mod articulation;
pub use articulation::RuleArticulationPoint;

use crate::{graph::BitSet, utils::DominatingSet};

pub trait ReductionRule<Graph> {
    const NAME: &str;

    /// If true, *always* try to delete unnecessary edges afterwards;
    /// signals that this edge removal is part of the rule itself
    const REMOVAL: bool = false;

    /// Applies the rule to the given graph and a partial solution.
    /// The rule may modify either. The first value return is true
    /// if the rule modified the graph; the second is Some(..) if
    /// the rule needs a post processing step.
    fn apply_rule(
        &mut self,
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
        redundant: &mut BitSet,
    );
}
