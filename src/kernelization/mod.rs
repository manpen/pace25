use crate::prelude::*;
use paste::paste;

mod leaves;
mod pairs;
mod tree;
mod twins;
mod two_paths;

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

#[derive(Clone, Copy, Debug)]
pub struct KernelRules {
    leaves: (bool, bool),
    pairs: (bool, bool),
    tree: (bool, bool),
    twins: (bool, bool),
    two_paths: (bool, bool),
}

impl Default for KernelRules {
    fn default() -> Self {
        Self {
            leaves: (true, true),
            pairs: (true, true),
            tree: (false, false),
            twins: (true, true),
            two_paths: (true, true),
        }
    }
}

impl KernelRules {
    pub fn minimal_set() -> Self {
        Self {
            leaves: (true, true),
            twins: (true, true),
            pairs: (false, false),
            tree: (false, false),
            two_paths: (false, false),
        }
    }
}

pub struct Kernelization<'a, G> {
    graph: &'a mut G,
    sequence: &'a mut ContractionSequence,

    slack: NumNodes,
    protected_nodes: BitSet,

    rules: KernelRules,
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
            rules: Default::default(),
        }
    }

    #[allow(dead_code)]
    pub fn run_first_round(&mut self) -> bool {
        macro_rules! invoke_rule {
            ($rule : ident) => {{
                (self.rules.$rule.0 && paste! {self.[< rule_ $rule >]()})
            }};
        }

        repeat_while!({
            invoke_rule!(pairs)
                || invoke_rule!(leaves)
                || invoke_rule!(twins)
                || invoke_rule!(tree)
                || invoke_rule!(two_paths)
        })
    }

    pub fn run_recursion_defaults(&mut self) -> bool {
        macro_rules! invoke_rule {
            ($rule : ident) => {{
                (self.rules.$rule.1 && paste! {self.[< rule_ $rule >]()})
            }};
        }

        repeat_while!({
            invoke_rule!(pairs)
                || invoke_rule!(leaves)
                || invoke_rule!(twins)
                || invoke_rule!(tree)
                || invoke_rule!(two_paths)
        })
    }

    pub fn configure_rules(&mut self, rules: KernelRules) {
        self.rules = rules;
    }

    pub fn slack(&self) -> NumNodes {
        self.slack
    }

    fn is_protected(&self, node: Node) -> bool {
        self.protected_nodes[node]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::testing::get_test_graphs_with_tww;
    use paste::paste;

    macro_rules! impl_test_rule {
        ($rule:ident) => {
            paste! {
                #[test]
                fn [< rule_$rule >]() {
                    for (filename, graph, presolved_tww) in
                        get_test_graphs_with_tww("instances/small-random/*.gr").step_by(3)
                    {
                        if graph.number_of_nodes() > 15 {
                            continue;
                        }


                        let mut kernel_rules = KernelRules::minimal_set();
                        kernel_rules.$rule = (true, true);

                        let mut bb_features = crate::exact::branch_and_bound::FeatureConfiguration::default();
                        bb_features.kernelize = true;
                        bb_features.kernel_rules = kernel_rules;

                        let mut algo = BranchAndBound::new(graph);
                        algo.configure_features(bb_features);

                        println!(" Test {filename}");
                        let (tww, _seq) = algo.solve().unwrap();
                        assert_eq!(tww, presolved_tww, "file: {filename}");
                    }
                }
            }
        };
    }

    impl_test_rule!(pairs);
    //impl_test_rule!(tree);
    impl_test_rule!(two_paths);
}
