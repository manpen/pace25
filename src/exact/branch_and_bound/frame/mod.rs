use std::mem::take;

use crate::prelude::default_pruning;

mod contractions;

use self::contractions::ContractBranch;

use super::*;

/// The frame implements the heavy lifting of the branch and bound algorithm (see also [`BranchAndBound`]
/// for further information). After construction, the computation starts with [`Frame::initialize`].
/// It may either directly return a result using [`BBResult::Result`] or branch by returning [`BBResult::Branch`].
/// In the latter case, we expect the child's computation to complete before calling the parents [`Frame::resume`]
/// method. Internally, the resumption point (and information to resume) are kept in the attribute [`Frame::resume_with`].
pub(super) struct Frame<G> {
    /// The graph instance to be solved. It may get updated (e.g. by kernelization) or even taken (e.g. to pass to a
    /// child without copying if we know that the graph is not required by the parent anymore).
    pub(super) graph: G,

    /// upper bound on the solution size for `graph`.
    /// The upper bound is INCLUSIVE, i.e. a solution must always have `tww <= not_above`.
    pub(super) not_above: NumNodes,

    /// Lower bound on the solution size for `graph`.
    /// The lower bound is inclusive, i.e. a solution has always size `solution.len() >= lower_bound as usize`.
    pub(super) slack: NumNodes,

    pub(super) resume_with: Option<Box<dyn BranchMode<G>>>,

    pub(super) contract_seq: ContractionSequence,

    pub(super) initial_slack: NumNodes,
    pub(super) initial_not_above: NumNodes,
}

pub(super) trait BranchMode<G> {
    fn branch(&mut self, frame: &mut Frame<G>) -> BBResult<G> {
        self.resume(frame, None)
    }
    fn resume(&mut self, frame: &mut Frame<G>, from_child: OptionalTwwSolution) -> BBResult<G>;
    fn describe(&self) -> String;
}

use log::trace;

#[allow(dead_code)]
impl<G: FullfledgedGraph> Frame<G> {
    pub(super) fn new(graph: G, slack: Node, not_above: Node) -> Self {
        let n = graph.number_of_nodes();
        Self {
            graph,
            resume_with: None,
            slack,
            not_above,
            contract_seq: ContractionSequence::new(n),
            initial_slack: slack,
            initial_not_above: not_above,
        }
    }

    /// This function carries out kernelization and decides upon a branching strategy.
    pub(super) fn initialize(&mut self) -> BBResult<G> {
        default_pruning(&mut self.graph, self.slack, &mut self.contract_seq);

        if self.graph.number_of_edges() == 0 {
            trace!("Left with empty kernel");
            return BBResult::Result(Some(TwwSolution {
                tww: self.slack,
                sequence: take(&mut self.contract_seq),
            }));
        }

        self.update_slack(self.graph.max_red_degree());

        self.resume_with = Some(Box::new(ContractBranch::new(self)));
        self.resume(None)
    }

    /// This function is called if this previously branched using `BBResult::Branch` and the computation
    /// at the child (and possibly its children) completed. We then receive the result computed by
    /// the child.
    pub(super) fn resume(&mut self, from_child: OptionalTwwSolution) -> BBResult<G> {
        // we have to temporarily take out `self.resume_with` to work around the borrow checker,
        // since `branching.[..].resume` needs a `&mut self` as well
        let mut branching = take(&mut self.resume_with);
        let result = branching.as_mut().unwrap().resume(self, from_child);
        self.resume_with = branching;
        result
    }

    fn update_slack(&mut self, slack: NumNodes) {
        self.slack = self.slack.max(slack);
    }

    fn update_not_above(&mut self, not_above: NumNodes) {
        self.not_above = self.not_above.min(not_above);
    }

    fn complete_solution(&mut self, partial: TwwSolution) -> BBResult<G> {
        self.contract_seq.append(&partial.sequence);
        BBResult::Result(Some(TwwSolution {
            tww: partial.tww.max(self.slack),
            sequence: take(&mut self.contract_seq),
        }))
    }

    fn complete_optional_solution(&mut self, partial: OptionalTwwSolution) -> BBResult<G> {
        if let Some(partial) = partial {
            self.complete_solution(partial)
        } else {
            self.fail()
        }
    }

    fn fail(&self) -> BBResult<G> {
        BBResult::Result(None)
    }
}
