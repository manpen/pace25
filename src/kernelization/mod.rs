use crate::prelude::*;

mod leaves;
mod pairs;
mod twins;

macro_rules! repeat_while {
    ($body : block) => {{
        let mut global_changes = false;
        loop {
            if !{ $body } {
                break global_changes;
            }
            global_changes = true;
        }
    }};
}
use repeat_while;

pub struct Kernelization<'a, G> {
    graph: &'a mut G,
    sequence: &'a mut ContractionSequence,

    slack: NumNodes,
    protected_nodes: BitSet,
}

impl<'a, G> Kernelization<'a, G>
where
    G: FullfledgedGraph,
{
    #[allow(dead_code)]
    pub fn new(graph: &'a mut G, sequence: &'a mut ContractionSequence) -> Self {
        let protected_nodes = BitSet::new(graph.number_of_nodes());
        Self::new_with_protected(graph, sequence, 0, protected_nodes)
    }

    pub fn new_with_protected(
        graph: &'a mut G,
        sequence: &'a mut ContractionSequence,
        slack: NumNodes,
        protected_nodes: BitSet,
    ) -> Self {
        Self {
            graph,
            sequence,
            slack,
            protected_nodes,
        }
    }

    #[allow(dead_code)]
    pub fn run_first_round(&mut self) -> bool {
        repeat_while!({ self.rule_pairs() || self.rule_leaves() || self.rule_twins() })
    }

    pub fn run_recursion_defaults(&mut self) -> bool {
        repeat_while!({ self.rule_pairs() || self.rule_leaves() || self.rule_twins() })
    }

    pub fn slack(&self) -> NumNodes {
        self.slack
    }

    fn is_protected(&self, node: Node) -> bool {
        self.protected_nodes[node]
    }
}
