use super::*;
use std::{cmp, collections::HashSet};

pub trait AdjacencyList {
    fn num_nodes(&self) -> usize;
    fn neighbors(&self, node: Node) -> Vec<Node>;
    // Add other required graph operations...
}

pub struct GraphCutBuilder<'a, G: AdjacencyList + ArticluationPoint + Traversal> {
    graph: &'a G,
    enabled_cuts: Vec<CutType>,
    min_cc_size: Option<NumNodes>, // Minimum connected component size to keep
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CutType {
    OneCut,
    TwoCut,
    ThreeCut,
}

impl<'a, G: AdjacencyList + ArticluationPoint + Traversal> GraphCutBuilder<'a, G> {
    pub fn new(graph: &'a G) -> Self {
        Self {
            graph,
            enabled_cuts: vec![CutType::OneCut, CutType::TwoCut, CutType::ThreeCut],
            min_cc_size: None,
        }
    }

    pub fn enable_cut(mut self, cut_type: CutType) -> Self {
        if self.enabled_cuts.contains(&cut_type) {
            self
        } else {
            self.enabled_cuts.push(cut_type);
            self
        }
    }

    pub fn disable_cut(mut self, cut_type: CutType) -> Self {
        if let Some(index) = self
            .enabled_cuts
            .iter()
            .position(|value| *value == cut_type)
        {
            self.enabled_cuts.swap_remove(index);
        }
        self
    }

    pub fn min_cc_size(mut self, size: Option<NumNodes>) -> Self {
        self.min_cc_size = size;
        self
    }

    pub fn compute(self) -> Vec<Vec<Node>> {
        let mut all_cuts: Vec<Vec<Node>> = Vec::new();

        for &cut_type in &self.enabled_cuts {
            let cut = match cut_type {
                CutType::OneCut => self.compute_one_cut(),
                CutType::TwoCut => self.compute_two_cut(),
                CutType::ThreeCut => self.compute_three_cut(),
            };
            all_cuts.extend(cut.into_iter());
        }
        all_cuts
    }

    fn filter_cuts(&self, candidates: Vec<Vec<Node>>) -> Vec<Vec<Node>> {
        let mut good_cuts: Vec<Vec<Node>> = vec![];
        if let Some(min_cc_size) = self.min_cc_size {
            for candidate in candidates {
                let partition = self
                    .graph
                    .partition_into_connected_components_exclude_nodes(true, candidate.clone());
                let mut smallest = 0;
                let mut second_smallest = 0;
                if partition.number_of_classes() > 1 {
                    for i in 0..partition.number_of_classes() {
                        let a = partition.number_in_class(i);
                        if a < smallest {
                            second_smallest = smallest;
                            smallest = a;
                        } else if a < second_smallest {
                            second_smallest = a;
                        }
                    }
                    if second_smallest >= min_cc_size {
                        good_cuts.push(candidate);
                    }
                }
            }
        }
        good_cuts
    }

    fn compute_one_cut(&self) -> Vec<Vec<Node>> {
        let candidates: Vec<Vec<Node>> = self
            .graph
            .compute_articulation_points()
            .iter_set_bits()
            .map(|u| vec![u])
            .collect();
        self.filter_cuts(candidates)
    }

    fn compute_two_cut(&self) -> Vec<Vec<Node>> {
        let mut cuts: HashSet<(Node, Node)> = Default::default();
        let n = self.graph.number_of_nodes();
        for u in 0..n {
            let mut visited = BitSet::new(n);
            visited.set_bit(u);
            cuts.extend(
                self.graph
                    .compute_articulation_points_with_visited(visited)
                    .iter_set_bits()
                    .map(|v| (cmp::min(u, v), cmp::max(u, v))),
            );
        }
        let vec_cuts: Vec<Vec<Node>> = cuts.iter().map(|(u, v)| vec![*u, *v]).collect();
        self.filter_cuts(vec_cuts)
    }

    fn compute_three_cut(&self) -> Vec<Vec<Node>> {
        let mut cuts: HashSet<(Node, Node, Node)> = Default::default();
        let n = self.graph.number_of_nodes();
        for u in 0..n {
            for v in (u + 1)..n {
                let mut visited = BitSet::new(n);
                visited.set_bit(u);
                visited.set_bit(v);
                cuts.extend(
                    self.graph
                        .compute_articulation_points_with_visited(visited)
                        .iter_set_bits()
                        .map(|w| {
                            let mut tmp = [u, v, w];
                            tmp.sort();
                            (tmp[0], tmp[1], tmp[2])
                        }),
                );
            }
        }
        let vec_cuts: Vec<Vec<Node>> = cuts.iter().map(|(u, v, w)| vec![*u, *v, *w]).collect();
        self.filter_cuts(vec_cuts)
    }
}
