use crate::{
    graph::{Node, NumNodes},
    prelude::{NeighborsSlice, SortedNeighborhoods},
};

/// An InsertionEntry assigns a list of nodes (neighbors) to a specific node.
/// See InsertionTree for more explanation
#[derive(Debug, Clone)]
struct InsertionEntry {
    /// Entries
    data: Vec<Node>,
    /// Node-Representative
    value: Node,
}

const NOT_SET: Node = Node::MAX;

/// InsertionTree - Constant time queries for common neighbors
///
/// An InsertionTree is a binary tree used to store common neighbors of all nodes inserted into the
/// tree. A parent node stores the common entries of its two child node entries as well as its own
/// original entries, ie. the intersection of all three InsertionEntries.
///
/// A leaf is either empty as no node was yet assigned or stores a node and all its neighbors as entries.
///
/// The root of the tree thus stores all nodes that are incident to all nodes inserted into the
/// tree as entries.
///
/// The tree is constructed in a way that
/// (I4) is either a leaf or has two children.
///
/// See InsertionForest for more important invariants (I1 - I3).
#[derive(Debug, Clone)]
struct InsertionTree {
    /// Binary tree compacted into a list
    tree_nodes: Vec<Option<InsertionEntry>>,
    /// Indexes in `tree_nodes` that can be (re-)assigned
    free_nodes: Vec<Node>,
    /// Owner of the tree (see InsertionTrees for explanation)
    owner: Node,
}

impl InsertionTree {
    /// Creates an empty tree for a new owner
    pub fn new(owner: Node) -> Self {
        Self {
            tree_nodes: Vec::new(),
            free_nodes: Vec::new(),
            owner,
        }
    }

    /// Overwrites `dest` with all elements that are both in `dest` and `other`.
    /// Faster than `intersect_balanced` if one list is significantly longer than the other.
    ///
    /// (I3) The lists are sorted.
    fn intersect_unbalanced(dest: &mut Vec<Node>, other: &[Node]) {
        debug_assert!(dest.is_sorted() && other.is_sorted());

        let mut ptrw = 0usize;
        let mut ptrr = 0usize;

        if dest.len() < other.len() {
            for ptr1 in 0..dest.len() {
                let item = dest[ptr1];
                match other[ptrr..].binary_search_by(|x| x.cmp(&item)) {
                    Ok(idx) => {
                        dest[ptrw] = item;
                        ptrw += 1;
                        ptrr += idx;
                    }
                    Err(idx) => {
                        ptrr += idx;
                        if idx == other.len() {
                            break;
                        }
                    }
                };
            }
        } else {
            for item in other {
                match dest[ptrr..].binary_search_by(|x| x.cmp(item)) {
                    Ok(idx) => {
                        dest[ptrw] = *item;
                        ptrw += 1;
                        ptrr += idx;
                    }
                    Err(idx) => {
                        ptrr += idx;
                        if idx == dest.len() {
                            break;
                        }
                    }
                };
            }
        }

        dest.truncate(ptrw);
    }

    /// Overwrites `dest` with all elements that are both in `dest` and `other`.
    /// Faster than `intersect_unbalanced` if the lists are roughly the same in length.
    ///
    /// (I3) The lists are sorted.
    fn intersect_balanced(dest: &mut Vec<Node>, other: &[Node]) {
        debug_assert!(dest.is_sorted() && other.is_sorted());

        let mut ptrw = 0usize;
        let mut ptr1 = 0usize;
        let mut ptr2 = 0usize;

        while ptr1 < dest.len() && ptr2 < other.len() {
            let item1 = dest[ptr1];
            let item2 = other[ptr2];

            // Avoid branching
            ptr1 += (item1 <= item2) as usize;
            ptr2 += (item1 >= item2) as usize;
            if item1 == item2 {
                dest[ptrw] = item1;
                ptrw += 1;
            }
        }

        dest.truncate(ptrw);
    }

    /// Overwrites `dest` with all elements that are both in `dest` and `other`.
    /// Depending on the list lengths, the intersection is computed with either `intersect_balanced` or `intersect_unbalanced`.
    ///
    /// (I3) The lists are sorted.
    /// (I2) Both `dest` and `other` must at least contain `self.owner`.
    fn intersect(dest: &mut Vec<Node>, other: &[Node]) {
        debug_assert!(dest.is_sorted() && other.is_sorted());

        let min_len = dest.len().min(other.len());

        // Application of (I2)
        if dest.len() == 1 {
            return;
        }
        if other.len() == 1 {
            dest.clear();
            dest.push(other[0]);

            return;
        }

        if min_len << 4 <= dest.len().max(other.len()) {
            Self::intersect_unbalanced(dest, other);
        } else {
            Self::intersect_balanced(dest, other);
        }
    }

    /// Return all entries in the root of the tree,
    /// ie. all nodes that are incident to *every* node inserted into the tree.
    pub fn get_root_nodes(&self) -> &[Node] {
        self.tree_nodes.first().map_or_else(
            || -> &[Node] { &[] },
            |node| {
                node.as_ref()
                    .map_or_else(|| -> &[Node] { &[] }, |entry| &entry.data)
            },
        )
    }

    /// Inserts a node into the tree and propagates potential changes in the intersections.
    /// Returns the position of `node` in `self.tree_nodes`.
    ///
    /// (I3) `neighbors` is sorted.
    /// (I4) Length of the tree must be odd such that every node has a sibling (except the root).
    pub fn add_entry(&mut self, node: Node, neighbors: &[Node]) -> usize {
        debug_assert!(neighbors.is_sorted());
        debug_assert!(neighbors.contains(&self.owner));

        // Insert node into free node or create new one -> return position of node in list
        let orig_pos = if let Some(free_node) = self.free_nodes.pop() {
            if let Some(entry) = &mut self.tree_nodes[free_node as usize] {
                entry.value = node;
                Self::intersect(&mut entry.data, neighbors);
            } else {
                let entry = InsertionEntry {
                    data: neighbors.to_vec(),
                    value: node,
                };
                self.tree_nodes[free_node as usize] = Some(entry);
            }
            free_node as usize
        } else {
            let entry = InsertionEntry {
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

        // Propagate changes in intersections bottom-up
        let mut pos = orig_pos;
        while pos > 0 {
            let (beg, end) = self.tree_nodes.split_at_mut(pos);
            pos = (pos - 1) >> 1;

            let child = end[0].as_ref().unwrap();

            let parent = &mut beg[pos];
            if let Some(parent_entry) = parent {
                Self::intersect(&mut parent_entry.data, &child.data);
            } else {
                let entry = InsertionEntry {
                    data: child.data.to_vec(),
                    value: NOT_SET,
                };
                self.tree_nodes[pos] = Some(entry);
            }
        }

        orig_pos
    }

    /// Removes `position`th node from the tree and re-computes the intersections bottom-up.
    ///
    /// (I3) All neighborhoods are sorted.
    /// (I4) Every node has a sibling (except the root).
    pub fn remove_entry<Neighbors: NeighborsSlice>(
        &mut self,
        mut position: usize,
        neighborhoods: &Neighbors,
    ) {
        debug_assert!(neighborhoods.are_all_sorted());
        debug_assert!(position < self.tree_nodes.len());

        let mut allow_copy = true;

        // Clear node entries and fill up with entries from children if possible
        if position + 1 < self.tree_nodes.len() {
            let (beg, end) = self.tree_nodes.split_at_mut(position + 1);

            let entry = beg[position].as_mut().unwrap();
            entry.data.clear();
            entry.value = NOT_SET;

            // Due to splitting at `position + 1`, the first `position + 1` elements are removed
            // from end and the indexing shifts
            let child1 = position;
            let child2 = position + 1;

            // (I4) Every child has a sibling => both childs exist if and only if `child2` exists
            if child2 < end.len() {
                if let Some(child_entry) = &end[child1] {
                    entry.data = child_entry.data.to_vec();
                    allow_copy = false;
                }

                if let Some(child_entry) = &end[child2] {
                    if allow_copy {
                        entry.data = child_entry.data.to_vec();
                        allow_copy = false;
                    } else {
                        Self::intersect(&mut entry.data, &child_entry.data);
                    }
                }
            }
        } else if position < self.tree_nodes.len() {
            let entry = self.tree_nodes[position].as_mut().unwrap();
            entry.data.clear();
            entry.value = NOT_SET;
        }

        // Node is a leaf
        if allow_copy {
            self.tree_nodes[position] = None;
        }

        // Propagate changes in intersections bottom-up
        while position > 0 {
            let is_right_child = ((position & 1) == 0) as usize;

            position -= is_right_child;
            let (beg, end) = self.tree_nodes.split_at_mut(position);
            position >>= 1;

            let parent = beg[position].as_mut().unwrap();
            parent.data.clear();

            allow_copy = true;

            if let Some(child_entry) = &end[0] {
                parent.data = child_entry.data.to_vec();
                allow_copy = false;
            }

            if let Some(child_entry) = &end[1] {
                if allow_copy {
                    parent.data = child_entry.data.to_vec();
                    allow_copy = false;
                } else {
                    Self::intersect(&mut parent.data, &child_entry.data);
                }
            }

            if parent.value != NOT_SET {
                if allow_copy {
                    parent.data = neighborhoods.neighbors_slice(parent.value).to_vec();
                } else {
                    Self::intersect(
                        &mut parent.data,
                        neighborhoods.neighbors_slice(parent.value),
                    );
                }
            } else if allow_copy {
                self.tree_nodes[position] = None;
            }
        }
    }
}

/// InsertionForest - Efficient queries into common neighborhoods
///
/// This data structure is responsible for storing and querying common neighbors of a select subset of nodes
/// in multiple instances. That is, every node u can get an InsertionTree assigned to itself in which neighbors
/// of u can be inserted. Then, one can query the list of all nodes that are incident to all nodes
/// inserted into the InsertionTree of u.
///
/// It works using the InsertionTree datastructure: a binary tree that computes intersections of
/// neighborhoods bottom-up to yield an intersection of all neighborhoods in its root.
///
/// See the GreedyReverseSearch heuristic for a more in-depth example.
///
/// The datastructure relies on a series of invariants:
/// (I1) Every node can be inserted in at most one InsertionTree at a time
/// (I2) The owner of an InsertionTree must be a neighbor to all nodes inserted into it
/// (I3) Neighborhoods are sorted to allow for faster intersection-computation.
///
#[derive(Debug, Clone)]
pub struct InsertionForest<Neighbors: NeighborsSlice> {
    /// A list of all current InsertionTrees
    trees: Vec<InsertionTree>,
    /// Stores at which position the InsertionTreeTree of node u is in `self.trees`
    index: Vec<NumNodes>,
    /// Stores at which position lies inside a InsertionTreeTree
    /// (I1) Every node can be in at least one tree
    positions: Vec<NumNodes>,
    /// Access to neighborhoods as slices
    /// (I3) All slices are sorted
    neighborhoods: Neighbors,
}

const NOT_EXISTING: NumNodes = NumNodes::MAX;

impl<Neighbors: NeighborsSlice> InsertionForest<Neighbors> {
    /// Create a new instance from neighborhoods
    /// (I3) Neighborhoods will be sorted
    pub fn new(mut neighborhoods: Neighbors) -> Self {
        neighborhoods.sort_all_neighbors_unstable();
        Self::new_sorted(neighborhoods)
    }

    /// Create a new instance from neighborhoods that are already sorted.
    pub fn new_sorted(neighborhoods: Neighbors) -> Self {
        debug_assert!(neighborhoods.are_all_sorted());

        let n = neighborhoods.number_of_nodes() as usize;
        Self {
            trees: Vec::new(),
            index: vec![NOT_EXISTING; n],
            positions: vec![NOT_EXISTING; n],
            neighborhoods,
        }
    }

    /// Get the root entries of a specific tree
    /// Panics if u has no InsertionTree assigned to it
    pub fn get_root_nodes(&self, u: Node) -> &[Node] {
        debug_assert!(self.index[u as usize] < self.trees.len() as NumNodes);
        self.trees[self.index[u as usize] as usize].get_root_nodes()
    }

    /// Inserts v into the InsertionTree of u
    /// (I1) v is in no other InsertionTree at the time
    pub fn add_entry(&mut self, u: Node, v: Node) {
        debug_assert!(self.positions[v as usize] == NOT_EXISTING);

        if self.index[u as usize] == NOT_EXISTING {
            self.index[u as usize] = self.trees.len() as NumNodes;
            self.trees.push(InsertionTree::new(u));
        }

        self.positions[v as usize] = self.trees[self.index[u as usize] as usize]
            .add_entry(v, self.neighborhoods.neighbors_slice(v))
            as NumNodes;
    }

    /// Removes v from the InsertionTree of u.
    /// (I1) The caller is responsible or ensuring that u and v are correct.
    ///
    /// Panics if u has no InsertionTree  
    pub fn remove_entry(&mut self, u: Node, v: Node) {
        debug_assert!(self.index[u as usize] != NOT_EXISTING);
        self.trees[self.index[u as usize] as usize]
            .remove_entry(self.positions[v as usize] as usize, &self.neighborhoods);
        self.positions[v as usize] = NOT_EXISTING;
    }

    /// Clears (and deletes) the InsertionTree of u.
    /// Panics if u has no InsertionTree
    pub fn clear(&mut self, u: Node) {
        debug_assert!(self.index[u as usize] != NOT_EXISTING);

        let pos = self.index[u as usize] as usize;
        self.trees.swap_remove(pos);
        self.index[u as usize] = NumNodes::MAX;

        if self.trees.len() > pos {
            self.index[self.trees[pos].owner as usize] = pos as NumNodes;
        }
    }

    /// Transfers ownership of an InsertionTree to another node
    /// Panics if u as now InsertionTree
    pub fn transfer(&mut self, u: Node, v: Node) {
        debug_assert!(self.index[u as usize] != NOT_EXISTING);

        self.index[v as usize] = self.index[u as usize];
        self.index[u as usize] = NOT_EXISTING;
        self.trees[self.index[v as usize] as usize].owner = v;
    }
}
