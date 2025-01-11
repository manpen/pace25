use crate::prelude::*;
use std::io::Write;

#[derive(Clone, Debug)]
pub struct DominatingSet {
    number_of_nodes: Node,
    set: Vec<Node>,
}

impl DominatingSet {
    pub fn new(number_of_nodes: Node) -> Self {
        Self {
            number_of_nodes,
            set: Vec::new(),
        }
    }

    /// Adds a node to the dominating set.
    ///
    /// # Example
    /// ```
    /// use dss::utils::DominatingSet;
    /// let mut domset = DominatingSet::new(5);
    /// domset.add_node(0);
    /// ```
    pub fn add_node(&mut self, node: Node) {
        assert!(node < self.number_of_nodes);
        self.set.push(node);
    }

    /// Adds multiple nodes to the dominating set.
    ///
    /// # Example
    /// ```
    /// use dss::utils::DominatingSet;
    /// let mut domset = DominatingSet::new(5);
    /// domset.add_nodes([0, 1, 2].into_iter());
    /// ```
    pub fn add_nodes(&mut self, nodes: impl IntoIterator<Item = Node>) {
        for u in nodes {
            self.add_node(u);
        }
    }

    /// Returns true if the dominating set is empty.
    ///
    /// # Example
    /// ```
    /// use dss::utils::DominatingSet;
    /// let domset = DominatingSet::new(5);
    /// assert!(domset.is_empty());
    /// domset.add_node(0);
    /// assert!(!domset.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// Returns the number of nodes in the dominating set.
    ///
    /// # Example
    /// ```
    /// use dss::utils::DominatingSet;
    /// let mut domset = DominatingSet::new(5);
    /// assert_eq!(domset.len(), 0);
    /// domset.add_node(0);
    /// assert_eq!(domset.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.set.len()
    }

    /// Returns an iterator over the nodes in the dominating set.
    ///
    /// # Example
    /// ```
    /// use dss::utils::DominatingSet;
    /// let mut domset = DominatingSet::new(5);
    /// domset.add_nodes([0, 1, 2].into_iter());
    /// let mut iter = domset.iter();
    /// assert_eq!(iter.next(), Some(0));
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = Node> + '_ {
        self.set.iter().copied()
    }

    /// Writes the dominating set to a writer using 1-based indexing,
    /// as required by the PACE competition.
    ///
    /// ```
    /// let mut domset = DominatingSet::new(5);
    /// domset.add_node(2);
    /// domset.add_node(4);
    ///
    /// let mut buffer: Vec<u8> = Vec::new(); // implements Write
    /// domset.write(&mut buffer).unwrap();
    /// let expected = b"2\n3\n5\n";
    /// assert_eq!(buffer, expected);
    /// ```
    pub fn write<W: Write>(&self, mut writer: W) -> anyhow::Result<()> {
        writeln!(&mut writer, "{}", self.set.len())?;
        for u in &self.set {
            writeln!(&mut writer, "{}", u + 1)?;
        }
        Ok(())
    }

    /// Computes the set of nodes covered by the dominating set.
    pub fn compute_covered(&self, graph: &impl FullfledgedGraph) -> BitSet {
        let mut covered = BitSet::new(graph.number_of_nodes());

        for &u in &self.set {
            covered.set_bits(graph.neighbors_of(u));
            covered.set_bit(u);
        }

        covered
    }

    /// Returns true if the dominating set is valid, i.e. it covers all nodes.
    pub fn is_valid(&self, graph: &impl FullfledgedGraph) -> bool {
        let covered = self.compute_covered(graph);
        covered.are_all_set()
    }
}
