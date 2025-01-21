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
    /// let mut domset = DominatingSet::new(5);
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
    /// use dss::utils::DominatingSet;
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
    pub fn compute_covered(&self, graph: &impl AdjacencyList) -> BitSet {
        let mut covered = BitSet::new(graph.number_of_nodes());

        for &u in &self.set {
            covered.set_bits(graph.neighbors_of(u));
            covered.set_bit(u);
        }

        covered
    }

    /// Returns true if the dominating set is valid, i.e. it covers all nodes.
    pub fn is_valid(&self, graph: &impl AdjacencyList) -> bool {
        let covered = self.compute_covered(graph);
        covered.are_all_set()
    }
}

/// An extended DominatingSet that allows differentiation between fixed and non-fixed nodes in the
/// set. Supports constant time queries of appearence in set.
#[derive(Debug, Clone)]
pub struct ExtDominatingSet {
    /// List of all nodes in the set, partitioned by fixed/non-fixed
    solution: Vec<Node>,
    /// Position for each possible node in the set (= NumNodes::MAX if not)
    positions: Vec<NumNodes>,
    /// Numver of fixed nodes, ie solution[..num_fixed] are fixed nodes, rest not
    num_fixed: NumNodes,
}

impl ExtDominatingSet {
    /// Creates a new dominating set
    pub fn new(n: usize) -> Self {
        Self {
            solution: Vec::new(),
            positions: vec![NumNodes::MAX; n],
            num_fixed: 0,
        }
    }

    /// Size of the set
    pub fn len(&self) -> usize {
        self.solution.len()
    }

    /// Is the set empty (thanks clippy)
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add a (non-fixed) node to the set
    pub fn push(&mut self, u: Node) {
        debug_assert!(self.positions[u as usize] == NumNodes::MAX);
        self.solution.push(u);
        self.positions[u as usize] = (self.len() - 1) as NumNodes;
    }

    /// Remove a node from the set
    pub fn remove(&mut self, u: Node) {
        let pos = self.positions[u as usize] as usize;
        debug_assert!(pos < self.len());
        if pos < self.num_fixed as usize {
            self.unfix_node(u);
            return;
        }

        self.solution.swap_remove(pos);
        if pos < self.len() {
            self.positions[self.solution[pos] as usize] = pos as NumNodes;
        }

        self.positions[u as usize] = NumNodes::MAX;
    }

    /// Add a fixed node to the set
    pub fn fix_node(&mut self, u: Node) {
        debug_assert!(!self.is_in_domset(u));
        if self.solution.len() > self.num_fixed as usize {
            let current_head = self.solution[self.num_fixed as usize];

            self.solution.push(current_head);
            self.positions[current_head as usize] = (self.len() - 1) as NumNodes;

            self.solution[self.num_fixed as usize] = u;
            self.positions[u as usize] = self.num_fixed;
        } else {
            self.solution.push(u);
            self.positions[u as usize] = 0;
        }
        self.num_fixed += 1;
    }

    /// Remove a fixed node from the set
    pub fn unfix_node(&mut self, u: Node) {
        assert!(self.is_fixed_node(u));
        self.num_fixed -= 1;

        let pos = self.positions[u as usize];
        if pos != self.num_fixed {
            let last_fixed = self.solution[self.num_fixed as usize];

            self.solution.swap(pos as usize, self.num_fixed as usize);
            self.positions[last_fixed as usize] = pos;
        }

        let pos = self.num_fixed as usize;
        self.solution.swap_remove(pos);
        if pos < self.len() {
            self.positions[self.solution[pos] as usize] = pos as NumNodes;
        }

        self.positions[u as usize] = NumNodes::MAX;
    }

    /// Replace a node by another
    pub fn replace(&mut self, u: Node, v: Node) {
        assert!(self.is_in_domset(u) && !self.is_in_domset(v));

        self.solution[self.positions[u as usize] as usize] = v;
        self.positions[v as usize] = self.positions[u as usize];
        self.positions[u as usize] = NumNodes::MAX;
    }

    /// Is u part of the set?
    pub fn is_in_domset(&self, u: Node) -> bool {
        self.positions[u as usize] != NumNodes::MAX
    }

    /// Is u a fixed node of the set?
    pub fn is_fixed_node(&self, u: Node) -> bool {
        self.positions[u as usize] < self.num_fixed
    }

    /// How many fixed nodes are there?
    pub fn num_of_fixed_nodes(&self) -> usize {
        self.num_fixed as usize
    }

    /// Are all nodes fixed?
    pub fn all_fixed(&self) -> bool {
        self.num_fixed as usize == self.len()
    }

    /// Iterator over all nodes in the set
    pub fn iter(&self) -> impl Iterator<Item = Node> + '_ {
        self.solution.iter().copied()
    }

    /// Iterator over all fixed nodes in the set
    pub fn iter_fixed(&self) -> impl Iterator<Item = Node> + '_ {
        self.solution[..self.num_fixed as usize].iter().copied()
    }

    /// Iterator over all non-fixed nodes in the set
    pub fn iter_non_fixed(&self) -> impl Iterator<Item = Node> + '_ {
        self.solution[(self.num_fixed as usize)..].iter().copied()
    }

    /// Returns the ith node in the set
    pub fn ith_node(&self, i: usize) -> Node {
        self.solution[i]
    }

    /// Computes the set of nodes covered by the dominating set.
    pub fn compute_covered(&self, graph: &impl AdjacencyList) -> BitSet {
        let mut covered = BitSet::new(graph.number_of_nodes());

        for &u in &self.solution {
            covered.set_bits(graph.neighbors_of(u));
            covered.set_bit(u);
        }

        covered
    }

    /// Returns true if the dominating set is valid, i.e. it covers all nodes.
    pub fn is_valid(&self, graph: &impl AdjacencyList) -> bool {
        let covered = self.compute_covered(graph);
        covered.are_all_set()
    }

    /// Writes the dominating set to a given output
    pub fn write<W: Write>(&self, mut writer: W) -> anyhow::Result<()> {
        writeln!(&mut writer, "{}", self.solution.len())?;
        for u in &self.solution {
            writeln!(&mut writer, "{}", u + 1)?;
        }
        Ok(())
    }
}
