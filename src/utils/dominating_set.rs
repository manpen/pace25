use crate::prelude::*;
use std::io::Write;

#[derive(Clone, Debug, Default)]
pub struct DominatingSet {
    set: Vec<Node>,
}

impl DominatingSet {
    pub fn add_node(&mut self, node: Node) {
        self.set.push(node);
    }

    pub fn write<W: Write>(&self, mut writer: W) -> anyhow::Result<()> {
        writeln!(&mut writer, "{}", self.set.len())?;
        for u in &self.set {
            writeln!(&mut writer, "{}", u)?;
        }
        Ok(())
    }

    pub fn is_valid(&self, graph: &impl FullfledgedGraph) -> bool {
        let mut covered = BitSet::new(graph.number_of_nodes());

        for &u in &self.set {
            covered.set_bits(graph.neighbors_of(u));
            covered.set_bit(u);
        }

        covered.cardinality() == covered.number_of_bits()
    }
}
