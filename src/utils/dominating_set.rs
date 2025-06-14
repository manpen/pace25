use itertools::Itertools;
use rand::Rng;
use rand_distr::{Distribution, Uniform};

use crate::prelude::*;
use std::{fmt::Debug, io::Write};

/// A DominatingSet that allows differentiation between fixed and non-fixed nodes in the
/// set. Supports constant time queries of membership in set.
#[derive(Clone)]
pub struct DominatingSet {
    /// List of all nodes in the set, partitioned by fixed/non-fixed
    solution: Vec<Node>,
    /// Position for each possible node in the set (= NumNodes::MAX if not)
    positions: Vec<NumNodes>,
}

impl Debug for DominatingSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("DominatingSet")
            .field(&self.solution)
            .finish()
    }
}

impl DominatingSet {
    /// Creates a new dominating set
    pub fn new(n: NumNodes) -> Self {
        Self {
            solution: Vec::new(),
            positions: vec![NumNodes::MAX; n as usize],
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

    pub fn clear(&mut self) {
        for x in self.solution.drain(..) {
            self.positions[x as usize] = NumNodes::MAX;
        }
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

    /// Returns the ith node in the dominating set.
    pub fn ith_node(&self, i: usize) -> Node {
        self.solution[i]
    }

    /// Computes the set of nodes covered by the dominating set.
    pub fn compute_covered(&self, graph: &impl AdjacencyList) -> BitSet {
        let mut covered = graph.vertex_bitset_unset();

        for &u in &self.solution {
            covered.set_bits(graph.closed_neighbors_of(u));
            covered.set_bit(u);
        }

        covered
    }

    /// Returns true if the dominating set is valid, ie. it covers all nodes.
    pub fn is_valid(&self, graph: &impl AdjacencyList) -> bool {
        let covered = self.compute_covered(graph);
        if covered.are_all_set() {
            return true;
        }

        false
    }

    /// Returns true if the dominating set is valid, ie. it covers all nodes.
    pub fn is_valid_given_previous_cover(
        &self,
        graph: &impl AdjacencyList,
        previous_cover: &BitSet,
    ) -> bool {
        let mut covered = self.compute_covered(graph);
        covered |= previous_cover;
        if covered.are_all_set() {
            return true;
        }

        false
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
        if self.len() != other.len() {
            return false;
        }

        let mut sol1 = self.solution.clone();
        let mut sol2 = other.solution.clone();

        sol1.sort_unstable();
        sol2.sort_unstable();

        sol1 == sol2
    }

    /// Samples a uniform node from all non-fixed nodes in the DominatingSet.
    /// Panics if there are no there are no non-fixed nodes in the DominatingSet.
    pub fn sample_non_fixed<R: Rng>(&self, rng: &mut R) -> Node {
        self.solution[rng.gen_range(0..self.len())]
    }

    /// Samples multiple non-fixed nodes in the DominatingSet as an iterator.
    /// Panics if there are no non-fixed nodes in the DominatingSet.
    pub fn sample_many_non_fixed<'a, R: Rng, const NUM_SAMPLES: usize>(
        &'a self,
        rng: &'a mut R,
    ) -> impl Iterator<Item = Node> + 'a {
        Uniform::new(0, self.len())
            .sample_iter(rng)
            .take(NUM_SAMPLES)
            .map(move |idx| self.solution[idx])
    }

    pub fn complete_set(n: NumNodes) -> Self {
        Self {
            solution: (0..n).collect_vec(),
            positions: (0..n).collect_vec(),
        }
    }
}
