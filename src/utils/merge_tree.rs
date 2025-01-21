use crate::graph::{Node, NumEdges, NumNodes};

/// A MergeEntry assigns a list of nodes (neighbors) to a specific node.
/// See MergeTree for more explanation
#[derive(Debug, Clone)]
struct MergeEntry {
    /// Entries
    data: Vec<Node>,
    /// Node-Representative
    value: Node,
}

/// MergeTree - Constant time queries for common neighbors
///
/// A MergeTree is a binary tree used to store common neighbors of all nodes inserted into the
/// tree. A parent node holds the common entries of its two child node entries as well as its own
/// original entries.
///
/// A leaf is either empty as no node was yet assigned (=> the parent node only considers entries
/// of its sibling and its own), or holds a node and all its neighbors as entries.
///
/// The root of the tree thus holds all nodes that are incident to all nodes inserted into the
/// tree.
#[derive(Debug, Clone)]
pub struct MergeTree {
    /// Binary tree compacted into a list
    tree_nodes: Vec<Option<MergeEntry>>,
    /// Indexes in `tree_nodes` that can be (re-)assigned
    free_nodes: Vec<Node>,
    /// Owner of the tree (see MergeTrees for explanation)
    owner: Node,
}

impl MergeTree {
    /// Creates an empty tree
    pub fn new(owner: Node) -> Self {
        Self {
            tree_nodes: Vec::new(),
            free_nodes: Vec::new(),
            owner,
        }
    }

    /// Resets the tree
    pub fn clear(&mut self) {
        self.free_nodes.clear();
        self.tree_nodes.clear();
    }

    /// Helper function to merge two lists of varying length where one list is much shorter than
    /// the other.
    ///
    /// Overwrites the first list.
    ///
    /// IMPORTANT: We assume both lists to be sorted.
    fn merge_unbalanced(dest: &mut Vec<Node>, other: &[Node]) {
        let mut ptrw = 0usize;
        let mut ptr1 = 0usize;
        let mut ptr2 = 0usize;

        if dest.len() < other.len() {
            while ptr1 < dest.len() {
                let item = dest[ptr1];
                let idx = other[ptr2..].partition_point(|x| *x < item) + ptr2;
                if idx == other.len() {
                    break;
                }

                if other[idx] == item {
                    dest[ptrw] = item;
                    ptrw += 1;
                }
                ptr2 = idx;
                ptr1 += 1;
            }
        } else {
            while ptr2 < other.len() {
                let item = other[ptr2];
                let idx = dest[ptr1..].partition_point(|x| *x < item) + ptr1;
                if idx == dest.len() {
                    break;
                }

                if dest[idx] == item {
                    dest[ptrw] = item;
                    ptrw += 1;
                }
                ptr1 = idx;
                ptr2 += 1;
            }
        }

        dest.truncate(ptrw);
    }

    /// Merge two lists into one only containing common elements.
    ///
    /// Overwrites the first list.
    ///
    /// IMPORTANT: We assume both lists to be sorted.
    fn merge(dest: &mut Vec<Node>, other: &[Node]) {
        let min_len = dest.len().min(other.len());

        if dest.len() == 1 {
            return;
        }
        if other.len() == 1 {
            dest.clear();
            dest.push(other[0]);

            return;
        }

        if min_len << 4 <= dest.len().max(other.len()) {
            return Self::merge_unbalanced(dest, other);
        }

        let mut ptrw = 0usize;
        let mut ptr1 = 0usize;
        let mut ptr2 = 0usize;

        while ptr1 < dest.len() && ptr2 < other.len() {
            let item1 = dest[ptr1];

            match other[ptr2..].binary_search(&item1) {
                Ok(pos) => {
                    dest[ptrw] = item1;
                    ptrw += 1;

                    ptr1 += 1;
                    ptr2 += pos + 1;
                }
                Err(pos) => {
                    ptr1 += 1;
                    ptr2 += pos;
                }
            }

            if ptr2 >= other.len() {
                break;
            }

            let item2 = other[ptr2];
            match dest[ptr1..].binary_search(&item2) {
                Ok(pos) => {
                    dest[ptrw] = item2;
                    ptrw += 1;

                    ptr1 += pos + 1;
                    ptr2 += 1;
                }
                Err(pos) => {
                    ptr1 += pos;
                    ptr2 += 1;
                }
            }
        }

        dest.truncate(ptrw);
    }

    /// Get all entries of the root node = all common neighbors
    pub fn get_root_nodes(&self) -> &[Node] {
        if self.tree_nodes.is_empty() {
            return &[];
        }

        match &self.tree_nodes[0] {
            Some(entry) => &entry.data,
            None => &[],
        }
    }

    /// Add a node and its neighbors to the tree
    ///
    /// IMPORTANT: neighbors must be sorted
    pub fn add_entry(&mut self, node: Node, neighbors: &[Node]) -> usize {
        let orig_pos = if let Some(free_node) = self.free_nodes.pop() {
            if let Some(entry) = &mut self.tree_nodes[free_node as usize] {
                entry.value = node;
                Self::merge(&mut entry.data, neighbors);
            } else {
                let entry = MergeEntry {
                    data: neighbors.to_vec(),
                    value: node,
                };
                self.tree_nodes[free_node as usize] = Some(entry);
            }
            free_node as usize
        } else {
            let entry = MergeEntry {
                data: neighbors.to_vec(),
                value: node,
            };
            self.tree_nodes.push(Some(entry));
            if self.tree_nodes.len() & 1 == 0 {
                self.tree_nodes.push(None);
                self.free_nodes.push(self.tree_nodes.len() as Node - 1);
                self.tree_nodes.len() - 2
            } else {
                self.tree_nodes.len() - 1
            }
        };

        let mut pos = orig_pos;

        while pos > 0 {
            let (beg, end) = self.tree_nodes.split_at_mut(pos);
            pos = (pos - 1) >> 1;

            let child = end[0].as_ref().unwrap();

            let parent = &mut beg[pos];
            if let Some(parent_entry) = parent {
                Self::merge(&mut parent_entry.data, &child.data);
            } else {
                let entry = MergeEntry {
                    data: child.data.to_vec(),
                    value: Node::MAX,
                };
                self.tree_nodes[pos] = Some(entry);
            }
        }

        orig_pos
    }

    /// Removes a node at a specific position (see MergeTrees).
    /// Requires access to all possible neighbor queries (here as a CSR implementation).
    ///
    /// IMPORTANT: individual neighbor-lists must be sorted in edges
    pub fn remove_entry(&mut self, mut position: usize, edges: &[Node], offsets: &[NumEdges]) {
        let mut allow_copy = true;

        if position + 1 < self.tree_nodes.len() {
            let (beg, end) = self.tree_nodes.split_at_mut(position + 1);

            let entry = beg[position].as_mut().unwrap();
            entry.data.clear();
            entry.value = Node::MAX;

            // Due to splitting at `position + 1`, the first `position + 1` elements are removed
            // from end and the indexing shifts
            let child1 = position;
            let child2 = position + 1;
            if child2 < end.len() {
                if let Some(child_entry) = &end[child2] {
                    entry.data = child_entry.data.to_vec();
                    allow_copy = false;
                }

                if let Some(child_entry) = &end[child1] {
                    if allow_copy {
                        entry.data = child_entry.data.to_vec();
                        allow_copy = false;
                    } else {
                        Self::merge(&mut entry.data, &child_entry.data);
                    }
                }
            }
        } else if position < self.tree_nodes.len() {
            let entry = self.tree_nodes[position].as_mut().unwrap();
            entry.data.clear();
            entry.value = Node::MAX;
        } else {
            return;
        }

        if allow_copy {
            self.tree_nodes[position] = None;
        }

        while position > 0 {
            let is_right_child = ((position & 1) == 0) as usize;

            position -= is_right_child;
            let (beg, end) = self.tree_nodes.split_at_mut(position);
            position >>= 1;

            let parent = beg[position].as_mut().unwrap();
            parent.data.clear();

            allow_copy = true;

            if let Some(child_entry) = &end[is_right_child] {
                parent.data = child_entry.data.to_vec();
                allow_copy = false;
            }

            if let Some(child_entry) = &end[1 - is_right_child] {
                if allow_copy {
                    parent.data = child_entry.data.to_vec();
                    allow_copy = false;
                } else {
                    Self::merge(&mut parent.data, &child_entry.data);
                }
            }

            if parent.value != Node::MAX {
                let neighbors = &edges[offsets[parent.value as usize] as usize
                    ..offsets[parent.value as usize + 1] as usize];
                if allow_copy {
                    parent.data = neighbors.to_vec();
                } else {
                    Self::merge(&mut parent.data, neighbors);
                }
            } else if allow_copy {
                self.tree_nodes[position] = None;
            }
        }
    }
}

/// A collection of MergeTrees
///
/// Keeps track of all current trees for a given graph.
#[derive(Debug, Clone)]
pub struct MergeTrees {
    /// A list of all current trees
    trees: Vec<MergeTree>,
    /// Stores at which position the MergeTree of node u is in trees
    index: Vec<NumNodes>,
    /// Stores at which position lies inside a MergeTree
    /// Assumes that every node can be in at most one MergeTree at a time (see GreedyReverseSearch)
    positions: Vec<NumNodes>,
    /// A compacted CSR edge list
    edges: Vec<Node>,
    /// CSR-Offsets into edges
    offsets: Vec<NumEdges>,
}

impl MergeTrees {
    /// Create a new instance from a CSR-edge list
    pub fn new(edges: Vec<Node>, offsets: Vec<NumEdges>) -> Self {
        assert!(!offsets.is_empty());
        let n = offsets.len() - 1;
        Self {
            trees: Vec::new(),
            index: vec![NumNodes::MAX; n],
            edges,
            offsets,
            positions: vec![NumNodes::MAX; n],
        }
    }

    /// Get the root entries of a specific tree
    pub fn get_root_nodes(&self, u: Node) -> &[Node] {
        if self.index[u as usize] == NumNodes::MAX {
            return &[];
        }
        self.trees[self.index[u as usize] as usize].get_root_nodes()
    }

    /// Inserts v into the MergeTree of u
    pub fn add_entry(&mut self, u: Node, v: Node) {
        if self.index[u as usize] == NumNodes::MAX {
            self.index[u as usize] = self.trees.len() as NumNodes;
            self.trees.push(MergeTree::new(u));
        }

        let (beg, end) = (
            self.offsets[v as usize] as usize,
            self.offsets[v as usize + 1] as usize,
        );
        self.positions[v as usize] = self.trees[self.index[u as usize] as usize]
            .add_entry(v, &self.edges[beg..end]) as NumNodes;
    }

    /// Removes v from the MergeTree of u
    pub fn remove_entry(&mut self, u: Node, v: Node) {
        if self.index[u as usize] == NumNodes::MAX {
            return;
        }
        self.trees[self.index[u as usize] as usize].remove_entry(
            self.positions[v as usize] as usize,
            &self.edges,
            &self.offsets,
        );
        self.positions[v as usize] = NumNodes::MAX;
    }

    /// Clears (and deletes) the MergeTree of u
    pub fn clear(&mut self, u: Node) {
        if self.index[u as usize] == NumNodes::MAX {
            return;
        }

        let pos = self.index[u as usize] as usize;
        self.trees.swap_remove(pos);
        self.index[u as usize] = NumNodes::MAX;

        if self.trees.len() > pos {
            self.index[self.trees[pos].owner as usize] = pos as NumNodes;
        }
    }

    /// Transfers ownership of the MergeTree of u to v
    pub fn transfer(&mut self, u: Node, v: Node) {
        self.index[v as usize] = self.index[u as usize];
        self.index[u as usize] = NumNodes::MAX;
        self.trees[self.index[v as usize] as usize].owner = v;
    }
}
