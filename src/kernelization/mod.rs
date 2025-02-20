pub mod rule1;
pub mod subsets;
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
