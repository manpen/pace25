use std::{
    io::{BufRead, Write},
    slice::from_raw_parts,
};

use ::digest::Digest;
use itertools::Itertools;
use log::debug;
use serde::Serialize;

use crate::{graph::*, prelude::PaceReader};

#[derive(Clone, Debug, Default, Eq, PartialEq, Hash, Serialize)]
pub struct ContractionSequence {
    #[serde(skip_serializing)]
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

    pub fn compute_twin_width<G>(&self, graph: G) -> Option<NumNodes>
    where
        G: GraphEdgeEditing + ColoredAdjacencyList + std::fmt::Debug,
    {
        let n = graph.number_of_nodes();
        self.compute_twin_width_upto(graph, n)
    }

    pub fn compute_twin_width_upto<G>(&self, mut graph: G, upto: NumNodes) -> Option<NumNodes>
    where
        G: GraphEdgeEditing + ColoredAdjacencyList + std::fmt::Debug,
    {
        // this is a checker; let's make it plain stupid to avoid bugs
        let mut twin_width = 0;

        for &(removed, survivor) in &self.seq {
            graph.merge_node_into(removed, survivor);
            let max_red_deg = graph.red_degrees().max().unwrap();
            twin_width = twin_width.max(max_red_deg);

            if twin_width >= upto {
                return Some(twin_width);
            }
        }

        assert!(
            // TODO: Removes this assertion and return None instead
            graph.degrees().all(|d| d == 0),
            "merges: {:?} graph: {:?}",
            self.seq,
            &graph
        ); //  .then_some(twin_width)

        Some(twin_width)
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
                debug!(
                    "Contraction Sequence does not completely merge graph: {:?}",
                    graph.vertices_with_neighbors().collect_vec()
                );
                return None;
            }

            let survivor = remaining.iter_set_bits().next().unwrap();
            self.seq.extend(
                remaining
                    .iter_set_bits()
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
            assert!(removed < self.num_nodes, "{removed} {}", self.num_nodes);
            let exists_before_removal = node_exists.clear_bit(removed);
            let still_exists = node_exists.get_bit(survivor);

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
        self.seq.reserve(other.seq.len());
        for &(u, v) in &other.seq {
            self.merge_node_into(u, v);
        }
        //self.seq.extend(&other.seq);
    }

    pub fn append_mapped<M: Getter>(&mut self, other: &ContractionSequence, mapper: &M) {
        self.seq.reserve(other.seq.len());
        for &(u, v) in &other.seq {
            self.merge_node_into(mapper.old_id_of(u).unwrap(), mapper.old_id_of(v).unwrap());
        }
        //self.seq.extend(&other.seq);
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

    pub fn last_survivor(&self) -> Option<Node> {
        self.seq.last().map(|(_, v)| *v)
    }

    pub fn apply_mapper<M: Getter>(&mut self, mapper: &M) {
        for (u, v) in &mut self.seq {
            *u = mapper.old_id_of(*u).unwrap();
            *v = mapper.old_id_of(*v).unwrap();
        }

        if let Some(max) = self.seq.iter().map(|&(u, v)| u.max(v)).max() {
            self.num_nodes = self.num_nodes.max(max + 1);
        }
    }

    pub fn pace_writer<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        for &(rem, sur) in &self.seq {
            writeln!(writer, "{} {}", sur + 1, rem + 1)?;
        }
        Ok(())
    }

    pub fn pace_reader<R: BufRead>(reader: R, number_of_nodes: Node) -> std::io::Result<Self> {
        let reader = PaceReader::try_new_contraction_sequence(reader, number_of_nodes)?;
        let mut cs = ContractionSequence::with_capacity(number_of_nodes);
        for Edge(sur, rem) in reader {
            cs.merge_node_into(rem, sur);
        }
        Ok(cs)
    }

    pub fn len(&self) -> NumNodes {
        self.merges().len() as NumNodes
    }

    pub fn is_empty(&self) -> bool {
        self.merges().is_empty()
    }

    pub fn swap_merges(&mut self, fst: usize, snd: usize) {
        if fst == snd {
            return;
        }
        if fst > snd {
            return self.swap_merges(snd, fst);
        }

        self.seq.swap(fst, snd);

        let (fst_rem, fst_sur) = self.seq[fst];
        for (rem, sur) in &mut self.seq[fst + 1..=snd] {
            if *rem == fst_rem {
                *rem = fst_sur;
            } else if *sur == fst_rem {
                *sur = fst_sur;
            }
        }
    }

    pub fn normalize(&mut self) {
        let mut mapper: Vec<_> = (0..self.num_nodes).collect();

        let search = |mapper: &mut Vec<Node>, mut node: usize| loop {
            let target = mapper[node] as usize;
            if target == node {
                return target as Node;
            }
            mapper[node] = mapper[target];
            node = target;
        };

        for (rem, sur) in &mut self.seq {
            *rem = search(&mut mapper, *rem as usize);
            *sur = search(&mut mapper, *sur as usize);

            if *rem < *sur {
                std::mem::swap(rem, sur);
            }

            mapper[*rem as usize] = *sur;
        }
    }

    pub fn binary_digest(&self) -> digest::Output<sha2::Sha256> {
        let mut hasher = sha2::Sha256::new();

        let base = self.seq.as_ptr() as *const u8;

        hasher.update(unsafe {
            from_raw_parts(base, self.seq.len() * std::mem::size_of::<(Node, Node)>())
        });
        hasher.finalize()
    }
}
