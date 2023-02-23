use std::io::Write;

use crate::graph::*;

#[derive(Clone)]
pub struct ContractionSequence {
    num_nodes: NumNodes,
    seq: Vec<(Node, Node)>,
}

#[derive(Copy, Clone)]
pub struct CSCheckPoint(usize);

impl ContractionSequence {
    pub fn new(num_nodes: NumNodes) -> Self {
        Self {
            seq: Vec::new(),
            num_nodes,
        }
    }

    pub fn with_capacity(num_nodes: NumNodes) -> Self {
        Self {
            seq: Vec::with_capacity(num_nodes as usize - 1),
            num_nodes,
        }
    }

    pub fn merge_node_into(&mut self, removed: Node, survivor: Node) {
        debug_assert!(self
            .seq
            .iter()
            .all(|(rem, _)| *rem != removed && *rem != survivor));
        self.seq.push((removed, survivor))
    }

    pub fn compute_twin_width<G>(&self, mut graph: G) -> Option<NumNodes>
    where
        G: GraphEdgeEditing + ColoredAdjacencyList,
    {
        // this is a checker; let's make it plain stupid to avoid bugs
        let mut twin_width = 0;

        for &(removed, survivor) in &self.seq {
            graph.merge_node_into(removed, survivor);
            let max_red_deg = graph.red_degrees().max().unwrap();
            twin_width = twin_width.max(max_red_deg);
        }

        graph.degrees().all(|d| d == 0).then_some(twin_width)
    }

    /// If the sequence is legal and produces a graph without edges, this function
    /// adds merges of degree zero singletons until only one isolated node remains.
    pub fn add_unmerged_singletons<G>(&mut self, graph: &G) -> Option<NumNodes>
    where
        G: Clone + GraphEdgeEditing + ColoredAdjacencyList,
    {
        // let's first check whether the second is legal and there's work left to do
        // (this call is much faster than the actual contraction of the graph)
        let remaining = self.remaining_nodes()?;

        if remaining.cardinality() > 1 {
            let mut graph = graph.clone();
            for &(removed, survivor) in &self.seq {
                graph.merge_node_into(removed, survivor);
            }

            if graph.degrees().any(|d| d > 0) {
                return None;
            }

            let survivor = remaining.get_first_set().unwrap();
            self.seq.extend(
                remaining
                    .iter()
                    .skip(1)
                    .map(|remove| (remove as Node, survivor as Node)),
            );
        }

        Some(remaining.cardinality().saturating_sub(1) as NumNodes)
    }

    /// If the sequence is valid (but possibly incomplete), this method returns
    /// `Some(nodes)` where `nodes` is the set of still unmerged nodes. If the sequence
    /// is infeasible (e.g. because a node is removed twice), returns `None`.
    pub fn remaining_nodes(&self) -> Option<BitSet> {
        let mut node_exists = BitSet::new_all_set(self.num_nodes);

        for &(removed, survivor) in &self.seq {
            let exists_before_removal = node_exists.unset_bit(removed);
            let still_exists = node_exists[survivor];

            assert!(still_exists, "{survivor} {:?}", &self.seq);
            assert!(exists_before_removal, "{:?}", &self.seq);

            if !still_exists || !exists_before_removal {
                return None;
            }
        }

        Some(node_exists)
    }

    pub fn number_of_nodes(&self) -> u32 {
        self.num_nodes
    }

    /// Appends another contraction sequence at the end of this one
    pub fn append(&mut self, other: &ContractionSequence) {
        self.seq.extend(&other.seq);
    }

    pub fn checkpoint(&self) -> CSCheckPoint {
        CSCheckPoint(self.seq.len())
    }

    pub fn restore(&mut self, checkpoint: CSCheckPoint) {
        self.seq.truncate(checkpoint.0)
    }

    pub fn merges(&self) -> &[(Node, Node)] {
        &self.seq
    }

    pub fn pace_writer<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        for &(rem, sur) in &self.seq {
            writeln!(writer, "{} {}", sur + 1, rem + 1)?;
        }
        Ok(())
    }
}
