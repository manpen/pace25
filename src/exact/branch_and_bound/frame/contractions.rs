use std::cmp::Reverse;

use super::*;

pub(super) struct ContractBranch {
    best_known: Option<((Node, Node), TwwSolution)>,
    last_contraction: Option<(Node, Node)>,
    branches: Vec<ContractBranchDescriptor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct ContractBranchDescriptor {
    score: NumNodes,
    nodes: (Node, Node),
}

impl<G: FullfledgedGraph> BranchMode<G> for ContractBranch {
    fn describe(&self) -> String {
        format!("ContractBranch[remaining={}]", self.branches.len())
    }

    fn resume(&mut self, frame: &mut Frame<G>, from_child: OptionalTwwSolution) -> BBResult<G> {
        // "new is always better" [b. stinson]
        // it's true here, since we contrained the child call to yield a strictly better solution (or none)
        if let Some(new_solution) = from_child {
            frame.update_not_above(new_solution.tww);

            if new_solution.tww <= frame.slack {
                self.branches.clear(); // prune all further branches
            }

            self.best_known = Some((self.last_contraction.unwrap(), new_solution));
        }

        while let Some(next_branch) = self.branches.pop() {
            if next_branch.score > frame.not_above {
                break; // scores are sorted desc; so if the last is too large, all before will be
            }

            let mut local_graph = frame.graph.clone();
            local_graph.merge_node_into(next_branch.nodes.0, next_branch.nodes.1);

            let red_degree = local_graph.max_red_degree();
            if red_degree > frame.not_above {
                continue;
            }

            self.last_contraction = Some(next_branch.nodes);
            return BBResult::Branch(Frame::new(
                local_graph,
                frame.slack.max(red_degree),
                frame.not_above,
            ));
        }

        // exhausted all branches; return best known solution (or fail otherwise)
        if let Some(((u, v), sol)) = take(&mut self.best_known) {
            frame.contract_seq.merge_node_into(u, v);
            frame.complete_solution(sol)
        } else {
            frame.fail()
        }
    }
}

impl ContractBranch {
    pub(super) fn new<G: FullfledgedGraph>(frame: &Frame<G>) -> Self {
        let mergeable = Self::identify_mergables(frame);
        let branches = Self::contraction_candidates(frame, mergeable);
        Self {
            best_known: None,
            last_contraction: None,
            branches,
        }
    }

    fn identify_mergables<G: FullfledgedGraph>(frame: &Frame<G>) -> BitSet {
        let mut mergable = BitSet::new(frame.graph.number_of_nodes());

        if frame.graph.degrees().all(|d| d == 0 || d == 2) {
            mergable.set_all();
        } else {
            for u in frame.graph.vertices() {
                if frame.graph.degree_of(u) == 2 && frame.graph.red_degree_of(u) == 0 {
                    continue;
                }
                mergable.set_bit(u);
                mergable.set_bits(frame.graph.neighbors_of(u).iter().copied());
            }
        }

        mergable
    }

    fn contraction_candidates<G: FullfledgedGraph>(
        frame: &Frame<G>,
        mut mergeable: BitSet,
    ) -> Vec<ContractBranchDescriptor> {
        let mut pairs = Vec::new();

        let red_degs_of_black_neighbors = frame
            .graph
            .vertices()
            .map(|u| {
                if !mergeable[u] {
                    return frame.not_above + 1;
                }

                frame
                    .graph
                    .black_neighbors_of(u)
                    .iter()
                    .map(|&v| frame.graph.red_degree_of(v) + 1)
                    .max()
                    .unwrap_or(0)
            })
            .collect_vec();

        for u in frame.graph.vertices_range() {
            if !mergeable.unset_bit(u) {
                continue;
            }
            let degree_u = frame.graph.degree_of(u);
            if degree_u == 0 {
                continue;
            }

            let mut two_neighbors = frame.graph.closed_two_neighborhood_of(u);
            two_neighbors.and(&mergeable);
            for v in two_neighbors.iter() {
                debug_assert!(v > u);
                let mut red_neighs = frame.graph.red_neighbors_after_merge(u, v, false);
                let mut red_card = red_neighs.cardinality();

                assert!(
                    red_neighs.cardinality() > 0,
                    "u: {u}, v: {v} graph: {:?}",
                    frame.graph
                );

                if red_card > frame.not_above {
                    continue;
                }

                for &x in frame.graph.red_neighbors_of(u) {
                    red_neighs.unset_bit(x);
                }

                for &x in frame.graph.red_neighbors_of(v) {
                    red_neighs.unset_bit(x);
                }

                for new_red in red_neighs.iter() {
                    red_card = red_card.max(frame.graph.red_degree_of(new_red) + 1);
                }

                if red_neighs.cardinality() <= frame.not_above {
                    pairs.push(ContractBranchDescriptor {
                        score: red_neighs.cardinality(),
                        nodes: (u, v),
                    });
                }
            }

            if degree_u > frame.not_above {
                continue;
            }

            let distant_nodes = {
                let mut three_neighbors = BitSet::new(frame.graph.number_of_nodes());
                for x in two_neighbors.iter() {
                    three_neighbors.set_bits(frame.graph.neighbors_of(x).iter().copied());
                }
                three_neighbors.and_not(&two_neighbors);
                three_neighbors.and(&mergeable);
                three_neighbors
            };

            let red_deg_of_black_neighbors = frame
                .graph
                .black_neighbors_of(u)
                .iter()
                .map(|&v| frame.graph.red_degree_of(v) + 1)
                .max()
                .unwrap_or(0);

            if red_degs_of_black_neighbors[u as usize] > frame.not_above {
                continue;
            }

            for v in distant_nodes.iter() {
                assert!(v > u);
                let degree_v = frame.graph.degree_of(v);
                let red_degree = red_deg_of_black_neighbors.max(degree_u + degree_v);
                if degree_v > 0
                    && red_degree <= frame.not_above
                    && frame
                        .graph
                        .black_neighbors_of(v)
                        .iter()
                        .all(|&w| frame.graph.red_degree_of(w) < frame.not_above)
                {
                    pairs.push(ContractBranchDescriptor {
                        score: red_degree,
                        nodes: (u, v),
                    });
                }
            }
        }
        pairs.sort_by_key(|x| Reverse(*x));
        pairs
    }
}
