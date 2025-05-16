use itertools::Itertools;

use crate::prelude::*;
use std::io::Write;

/// A DominatingSet that allows differentiation between fixed and non-fixed nodes in the
/// set. Supports constant time queries of membership in set.
#[derive(Debug, Clone)]
pub struct DominatingSet {
    /// List of all nodes in the set, partitioned by fixed/non-fixed
    solution: Vec<Node>,
    /// Position for each possible node in the set (= NumNodes::MAX if not)
    positions: Vec<NumNodes>,
    /// Numver of fixed nodes, ie solution[..num_fixed] are fixed nodes, rest not
    num_fixed: NumNodes,
}

impl DominatingSet {
    /// Creates a new dominating set
    pub fn new(n: NumNodes) -> Self {
        Self {
            solution: Vec::new(),
            positions: vec![NumNodes::MAX; n as usize],
            num_fixed: 0,
        }
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
        self.solution.len()
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
        self.len() == 0
    }

    /// Adds a node to the dominating set.
    ///
    /// # Example
    /// ```
    /// use dss::utils::DominatingSet;
    /// let mut domset = DominatingSet::new(5);
    /// domset.add_node(0);
    /// ```
    pub fn add_node(&mut self, u: Node) {
        debug_assert!(self.positions[u as usize] == NumNodes::MAX);
        self.positions[u as usize] = self.len() as NumNodes;
        self.solution.push(u);
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

    /// Removes a node from the dominating set.
    pub fn remove_node(&mut self, u: Node) {
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

    /// Removes multiple nodes from the dominating set.
    pub fn remove_nodes(&mut self, nodes: impl IntoIterator<Item = Node>) {
        for u in nodes {
            self.remove_node(u);
        }
    }

    /// Add a fixed node to the set
    pub fn fix_node(&mut self, u: Node) {
        debug_assert!(!self.is_in_domset(u));
        if self.solution.len() > self.num_fixed as usize {
            let current_head = self.solution[self.num_fixed as usize];

            self.positions[current_head as usize] = self.len() as NumNodes;
            self.solution.push(current_head);

            self.solution[self.num_fixed as usize] = u;
            self.positions[u as usize] = self.num_fixed;
        } else {
            self.positions[u as usize] = self.len() as NumNodes;
            self.solution.push(u);
        }
        self.num_fixed += 1;
    }

    /// Fixes multiple nodes in the dominating set.
    pub fn fix_nodes(&mut self, nodes: impl IntoIterator<Item = Node>) {
        for u in nodes {
            self.fix_node(u);
        }
    }

    /// Remove a fixed node from the set
    pub fn unfix_node(&mut self, u: Node) {
        debug_assert!(self.is_fixed_node(u));
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

    /// Removes multiple fixed nodes in the dominating set.
    pub fn unfix_nodes(&mut self, nodes: impl IntoIterator<Item = Node>) {
        for u in nodes {
            self.unfix_node(u);
        }
    }

    /// Replace a node in the dominating set by another.
    pub fn replace(&mut self, u: Node, v: Node) {
        debug_assert!(self.is_in_domset(u) && !self.is_in_domset(v));

        self.solution[self.positions[u as usize] as usize] = v;
        self.positions[v as usize] = self.positions[u as usize];
        self.positions[u as usize] = NumNodes::MAX;
    }

    /// Returns *true* if u is in the dominating set.
    pub fn is_in_domset(&self, u: Node) -> bool {
        self.positions[u as usize] != NumNodes::MAX
    }

    /// Returns *true* if u is a fixed node of the dominating set.
    pub fn is_fixed_node(&self, u: Node) -> bool {
        self.positions[u as usize] < self.num_fixed
    }

    /// Returns *true* if u is a fixed node of the dominating set.
    #[inline(always)]
    pub fn is_non_fixed_node(&self, u: Node) -> bool {
        self.is_in_domset(u) && !self.is_fixed_node(u)
    }

    /// Returns the number of fixed nodes in the dominating set.
    pub fn num_of_fixed_nodes(&self) -> usize {
        self.num_fixed as usize
    }

    /// Returns the number of non-fixed nodes in the dominating set.
    pub fn num_of_non_fixed_nodes(&self) -> usize {
        self.len() - self.num_of_fixed_nodes()
    }

    /// Returns *true* if all nodes in the dominating set are fixed.
    pub fn all_fixed(&self) -> bool {
        self.num_fixed as usize == self.len()
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
        self.solution.iter().copied()
    }

    /// Returns an iterator over all the fixed nodes in the dominating set.
    pub fn iter_fixed(&self) -> impl Iterator<Item = Node> + '_ {
        self.solution[..self.num_fixed as usize].iter().copied()
    }

    /// Returns an iterator over all the non-fixed nodes in the dominating set.
    pub fn iter_non_fixed(&self) -> impl Iterator<Item = Node> + '_ {
        self.solution[(self.num_fixed as usize)..].iter().copied()
    }

    /// Returns the ith node in the dominating set.
    pub fn ith_node(&self, i: usize) -> Node {
        self.solution[i]
    }

    /// Computes the set of nodes covered by the dominating set.
    pub fn compute_covered(&self, graph: &impl AdjacencyList) -> BitSet {
        let mut covered = graph.vertex_bitset_unset();

        for &u in &self.solution {
            covered.set_bits(graph.neighbors_of(u));
            covered.set_bit(u);
        }

        covered
    }

    /// Returns true if the dominating set is valid, ie. it covers all nodes.
    pub fn is_valid(&self, graph: &impl AdjacencyList) -> bool {
        let covered = self.compute_covered(graph);
        covered.are_all_set()
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
        writeln!(&mut writer, "{}", self.solution.len())?;
        for u in &self.solution {
            writeln!(&mut writer, "{}", u + 1)?;
        }
        Ok(())
    }

    /// Returns *true* if Self and another DomSet are identical to each other.
    pub fn equals(&self, other: &Self) -> bool {
        if self.len() != other.len() || self.num_of_fixed_nodes() != other.num_of_fixed_nodes() {
            return false;
        }

        let mut sol1 = self.solution.clone();
        let mut sol2 = other.solution.clone();

        let num_fixed = self.num_fixed as usize;

        sol1[..num_fixed].sort_unstable();
        sol2[..num_fixed].sort_unstable();

        if sol1[..num_fixed] != sol2[..num_fixed] {
            return false;
        }

        sol1[num_fixed..].sort_unstable();
        sol2[num_fixed..].sort_unstable();

        sol1[num_fixed..] == sol2[num_fixed..]
    }

    pub fn complete_set(n: NumNodes) -> Self {
        Self {
            solution: (0..n).collect_vec(),
            positions: (0..n).collect_vec(),
            num_fixed: 0,
        }
    }
}
