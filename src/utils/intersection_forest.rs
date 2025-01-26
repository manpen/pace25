use crate::{
    graph::{Node, NumNodes},
    prelude::{NeighborsSlice, SortedNeighborhoods},
};

/// An IntersectionEntry assigns a list of nodes (neighbors) to a specific node.
/// See IntersectionTree for more explanation
#[derive(Debug, Clone)]
struct IntersectionEntry {
    /// Entries
    data: Vec<Node>,
    /// Node-Representative
    value: Node,
}

const NOT_SET: Node = Node::MAX;

/// IntersectionTree - Constant time queries for common neighbors
///
/// An IntersectionTree is a binary tree used to store common neighbors of all nodes inserted into the
/// tree. A parent node stores the common entries of its two child node entries as well as its own
/// original entries, ie. the intersection of all three IntersectionEntries.
///
/// A leaf is either empty as no node was yet assigned or stores a node and all its neighbors as entries.
///
/// The root of the tree thus stores all nodes that are incident to all nodes inserted into the
/// tree as entries.
///
/// The tree is constructed in a way that
/// (I4) is either a leaf or has two children.
///
/// See IntersectionForest for more important invariants (I1 - I3).
#[derive(Debug, Clone)]
struct IntersectionTree {
    /// Binary tree compacted into a list
    tree_nodes: Vec<Option<IntersectionEntry>>,
    /// Indexes in `tree_nodes` that can be (re-)assigned
    free_nodes: Vec<Node>,
    /// Owner of the tree (see IntersectionTree for explanation)
    owner: Node,
}

impl IntersectionTree {
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
                let result = other[ptrr..].binary_search_by(|x| x.cmp(&item));
                dest[ptrw] = item;
                ptrw += result.is_ok() as usize;
                ptrr += result.unwrap_or_else(|x| x);

                if ptrr == other.len() {
                    break;
                }
            }
        } else {
            for item in other {
                let result = dest[ptrr..].binary_search_by(|x| x.cmp(item));
                if result.is_ok() {
                    dest[ptrw] = *item;
                    ptrw += 1;
                }
                ptrr += result.unwrap_or_else(|x| x);

                if ptrr == dest.len() {
                    break;
                }
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

            dest[ptrw] = item1;
            ptrw += (item1 == item2) as usize;
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
    /// (I2) `self.owner` is in `neighbors`
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
                let entry = IntersectionEntry {
                    data: neighbors.to_vec(),
                    value: node,
                };
                self.tree_nodes[free_node as usize] = Some(entry);
            }
            free_node as usize
        } else {
            let entry = IntersectionEntry {
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
                let entry = IntersectionEntry {
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
        debug_assert!(neighborhoods.are_all_neighbors_sorted());
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

/// IntersectionForest - Efficient queries into common neighborhoods
///
/// This data structure is responsible for storing and querying common neighbors of a select subset of nodes
/// in multiple instances. That is, every node u can get an IntersectionTree assigned to itself in which neighbors
/// of u can be inserted. Then, one can query the list of all nodes that are incident to all nodes
/// inserted into the IntersectionTree of u.
///
/// It works using the IntersectionTree datastructure: a binary tree that computes intersections of
/// neighborhoods bottom-up to yield an intersection of all neighborhoods in its root.
///
/// See the GreedyReverseSearch heuristic for a more in-depth example.
///
/// The datastructure relies on a series of invariants:
/// (I1) Every node can be inserted in at most one IntersectionTree at a time (or can be an owner of a tree but not both)
/// (I2) The owner of an IntersectionTree must be a neighbor to all nodes inserted into it
/// (I3) Neighborhoods are sorted to allow for faster intersection-computation.
///
#[derive(Debug, Clone)]
pub struct IntersectionForest<Neighbors: NeighborsSlice> {
    /// A list of all current IntersectionTrees
    trees: Vec<IntersectionTree>,
    /// Stores at which position the IntersectionTree of node u is in `self.trees`
    index: Vec<NumNodes>,
    /// Stores at which position lies inside a IntersectionTree
    /// (I1) Every node can be in at most one tree
    positions: Vec<NumNodes>,
    /// Access to neighborhoods as slices
    /// (I3) All slices are sorted
    neighborhoods: Neighbors,
}

const NOT_EXISTING: NumNodes = NumNodes::MAX;

impl<Neighbors: NeighborsSlice> IntersectionForest<Neighbors> {
    /// Create a new instance from neighborhoods
    /// (I3) Neighborhoods will be sorted
    pub fn new(mut neighborhoods: Neighbors) -> Self {
        neighborhoods.sort_all_neighbors_unstable();
        Self::new_sorted(neighborhoods)
    }

    /// Create a new instance from neighborhoods that are already sorted.
    pub fn new_sorted(neighborhoods: Neighbors) -> Self {
        debug_assert!(neighborhoods.are_all_neighbors_sorted());

        let n = neighborhoods.number_of_nodes() as usize;
        Self {
            trees: Vec::new(),
            index: vec![NOT_EXISTING; n],
            positions: vec![NOT_EXISTING; n],
            neighborhoods,
        }
    }

    /// Get the root entries of a specific tree
    /// Panics if u has no IntersectionTree assigned to it
    pub fn get_root_nodes(&self, u: Node) -> &[Node] {
        debug_assert!(self.index[u as usize] < self.trees.len() as NumNodes);
        self.trees[self.index[u as usize] as usize].get_root_nodes()
    }

    /// Returns *true* if u is the owner of some IntersectionTree
    pub fn owns_tree(&self, u: Node) -> bool {
        self.index[u as usize] != NOT_EXISTING
    }

    /// Inserts v into the IntersectionTree of u
    /// (I1) v is in no other IntersectionTree at the time
    pub fn add_entry(&mut self, u: Node, v: Node) {
        debug_assert!(self.positions[v as usize] == NOT_EXISTING);

        if self.index[u as usize] == NOT_EXISTING {
            self.index[u as usize] = self.trees.len() as NumNodes;
            self.trees.push(IntersectionTree::new(u));
        }

        self.positions[v as usize] = self.trees[self.index[u as usize] as usize]
            .add_entry(v, self.neighborhoods.neighbors_slice(v))
            as NumNodes;
    }

    /// Removes v from the IntersectionTree of u.
    /// (I1) The caller is responsible or ensuring that u and v are correct.
    ///
    /// Panics if u has no IntersectionTree  
    pub fn remove_entry(&mut self, u: Node, v: Node) {
        debug_assert!(self.index[u as usize] != NOT_EXISTING);
        self.trees[self.index[u as usize] as usize]
            .remove_entry(self.positions[v as usize] as usize, &self.neighborhoods);
        self.positions[v as usize] = NOT_EXISTING;
    }

    /// Clears (and deletes) the IntersectionTree of u.
    /// Panics if u has no IntersectionTree
    pub fn clear(&mut self, u: Node) {
        debug_assert!(self.index[u as usize] != NOT_EXISTING);

        let pos = self.index[u as usize] as usize;
        self.trees.swap_remove(pos);
        self.index[u as usize] = NumNodes::MAX;

        if self.trees.len() > pos {
            self.index[self.trees[pos].owner as usize] = pos as NumNodes;
        }
    }

    /// Transfers ownership of an IntersectionTree to another node
    /// Panics if u as now IntersectionTree
    pub fn transfer(&mut self, u: Node, v: Node) {
        debug_assert!(self.owns_tree(u) && !self.owns_tree(v));

        self.index[v as usize] = self.index[u as usize];
        self.index[u as usize] = NOT_EXISTING;
        self.trees[self.index[v as usize] as usize].owner = v;
    }
}

#[cfg(test)]
mod tests {
    use rand::{seq::SliceRandom, Rng};

    use crate::{graph::CsrGraph, io::GraphPaceReader};

    use super::*;

    #[test]
    fn test_intersections() {
        let rng = &mut rand::thread_rng();
        for n in [100, 200, 500] {
            for _ in 0..100 {
                let mut base: Vec<Node> = (0..n).collect();

                let size = rng.gen_range(20..((n as usize) / 2));

                base.shuffle(rng);
                let mut list1a = base[..size].to_vec();
                list1a.sort_unstable();
                let mut list1b = list1a.clone();

                base.shuffle(rng);
                let mut list2 = base[..size].to_vec();
                list2.sort_unstable();

                IntersectionTree::intersect_balanced(&mut list1a, &list2);
                IntersectionTree::intersect_unbalanced(&mut list1b, &list2);

                assert_eq!(list1a, list1b);
            }
        }
    }

    #[test]
    fn test_forest() {
        let reader = "p ds 6 8\n1 2\n1 3\n2 4\n2 5\n3 4\n3 6\n4 6\n5 6".as_bytes();
        let graph = CsrGraph::try_read_pace(reader).unwrap();

        let mut forest = IntersectionForest::new(graph);
        forest.add_entry(2, 3);
        assert_eq!(forest.get_root_nodes(2), &[1, 2, 3, 5]);
        forest.add_entry(2, 5);
        assert_eq!(forest.get_root_nodes(2), &[2, 3, 5]);
        forest.add_entry(2, 0);
        assert_eq!(forest.get_root_nodes(2), &[2]);
        forest.remove_entry(2, 5);
        assert_eq!(forest.get_root_nodes(2), &[1, 2]);

        assert!(!forest.owns_tree(1) && forest.owns_tree(2));
        forest.transfer(2, 1);
        assert!(forest.owns_tree(1) && !forest.owns_tree(2));
        forest.add_entry(1, 4);
        assert_eq!(forest.get_root_nodes(1), &[1]);
        forest.remove_entry(1, 0);
        assert_eq!(forest.get_root_nodes(1), &[1, 5]);
    }
}
