use crate::graph::{BitSet, Node, NumEdges, NumNodes};

const NOT_SET: Node = Node::MAX;

#[derive(Debug, Clone, Copy, Default)]
struct NodeInformation {
    pub offset_filtered: NumEdges,
    pub offset_unfiltered: NumEdges,
    pub tree_len: NumNodes,
    pub free_pos: NumNodes,
    pub data_len: NumNodes,
    pub tree_pos: NumNodes,
}

/// Indicator if this spot is a free node that does not point to another free node.
/// Also used as a mask to determine if a node is a free node.
const FREE_SLOT_MASK: NumNodes = NumNodes::MAX >> 1;

/// IntersectionForest - Efficient queries into common neighborhoods
///
/// This data structure supports dynamic queries into common neighborhoods of select subsets of
/// nodes subject to some invariants listed below. That is, if a set S of nodes is inserted into
/// some IntersectionTree T, the associated data of the root of T is the list of all nodes that are
/// incident to *every* node in S.
///
/// It is implemented using an inlined binary tree and supports adding nodes to trees, removing
/// nodes from trees, as well as transferring ownership of a tree to another node. Intersections are
/// computed bottom-up such that the root holds the intersection of all associated data in the
/// tree.
///
/// See the GreedyReverseSearch heuristic for a more in-depth example.
///
/// The data structure relies on a series of invariants listed below; function-related invariants
/// are stated in the function documentation.
/// (I1) Every node can be inserted in at most one tree at a time. Nodes that own trees can only be inserted into their own tree (as a consequence of (I2))
/// (I2) The owner of a tree is a neighbor of all nodes inserted into the tree.
/// (I3) Neighborhoods, i.e. associated data are sorted in increasing order (no multi-edges).
/// (I4) Every node except the root in a tree has a parent (not free-node)
/// (I5) The associated data of node u is the intersection of the neighbors of u, and the data (if existent) of its two (or one) children
///
#[derive(Debug, Clone)]
pub struct InlineIntersectionForest {
    /// List of all edges sorted by source, then by target
    edges: Vec<Node>,

    /// Important information for all nodes u
    /// * offset_filtered := first neighbor of u in edges <=> begin of associated data entries in data
    /// * offset_unfiltered := begin of owned tree representation in forest     
    /// * tree_len := length of the owned tree: Tree[u] = forest[offset_unfiltered..(offset_unfiltered + tree_len)]
    /// * data_len := length of associated data if node in a tree: Data[u] = data[offset_filtered..(offset_filtered + data_len)]
    /// * tree_pos := position in the tree where u is inserted: if u in Tree[v], then forest[offset_unfiltered + TreePos[u]] = u and TreePos[u] < TreeLen[v]
    /// * free_pos := (see self.forest documentation)
    ///
    /// (I2) if u is in Tree[v] => v is in Data[u]
    ///
    /// To allow faster slice access without bound-checks, we store an artificial node (n + 1) with
    /// * offset_filtered = edges.len()
    /// * offset_unfiltered = forest.len()
    /// * .. = 0
    nodes: Vec<NodeInformation>,

    /// Inlined forest: if TreeLen[u] > 0 => Tree[u] = forest[OffsetUnfiltered[u]..(OffsetUnfiltered[u] + TreeLen[u])]
    ///
    /// Every node currently not a tree node but an empty node or a free node has its leftmost bit set to 1.
    /// Free nodes simulate a linked list, i.e. if Tree[u][v] = x is a free node, then Bits[x] = yz where y = 1 indicating that x is a free node and z = (0|1)^(Bits[x].len() - 1)
    /// is the next element in the simulated linked list.
    /// The head of the linked list of Tree[u] is stored in FreePos[u] in nodes.
    /// The tail of the linked lists points to 0111..111 which is considered a null-element.
    /// New elements are pushed to the head (and replace it) instead of the tail.
    ///
    /// Due to order of removal of elements in remove_entry(u, v), the linked list of free nodes is
    /// ordered in such a way that if a free node x is a predecessor of another free node y in
    /// Tree[u], then x was inserted later than y and thus appears earlier in the list.
    forest: Vec<Node>,

    /// Inlined lists of associated data: if DataLen[u] > 0 => Data[u] = data[OffsetFiltered[u]..(OffsetFiltered[u] + DataLen[u])]
    data: Vec<Node>,

    /// Temporary bitset used in compress_tree to repair the tree after compressing
    dirty_nodes: BitSet,
}

impl Default for InlineIntersectionForest {
    fn default() -> Self {
        Self {
            edges: vec![],
            nodes: vec![NodeInformation::default()],
            forest: vec![],
            data: vec![],
            dirty_nodes: BitSet::new(0),
        }
    }
}

macro_rules! node {
    ($self:ident, $node:expr) => {
        $self.nodes[$node as usize]
    };
}

macro_rules! data {
    ($self:ident, $node:expr) => {
        $self.data[(node!($self, $node).offset_filtered as usize)
            ..(node!($self, $node).offset_filtered as usize
                + node!($self, $node).data_len as usize)]
    };
}

macro_rules! data_ref {
    ($self:ident, $node:expr) => {
        &data!($self, $node)
    };
}

impl InlineIntersectionForest {
    /// Creates a new IntersectionForest from a CSR-Representation of the graph as well as two BitSets
    /// indicating which nodes will never appear in any way (owner/tree-node/data-node) in this
    /// data structure (removable_nodes) and which nodes can be ignored in data-lists (ignorable_nodes).
    ///
    /// This assumes that the edges lists are sorted by source and target.
    pub fn new(
        csr_edges: Vec<Node>,
        csr_offsets: Vec<NumEdges>,
        removable_nodes: BitSet,
        ignorable_nodes: BitSet,
    ) -> Self {
        Self::new_inner::<true>(csr_edges, csr_offsets, removable_nodes, ignorable_nodes)
    }

    /// Creates a new IntersectionForest from a CSR-Representation of the graph as well as two BitSets
    /// indicating which nodes will never appear in any way (owner/tree-node/data-node) in this
    /// data structure (removable_nodes) and which nodes can be ignored in data-lists (ignorable_nodes).
    ///
    /// This will sort the edge lists by source and target when initializing.
    pub fn new_unsorted(
        csr_edges: Vec<Node>,
        csr_offsets: Vec<NumEdges>,
        removable_nodes: BitSet,
        ignorable_nodes: BitSet,
    ) -> Self {
        Self::new_inner::<false>(csr_edges, csr_offsets, removable_nodes, ignorable_nodes)
    }

    fn new_inner<const SORTED: bool>(
        mut edges: Vec<Node>,
        csr_offsets: Vec<NumEdges>,
        removable_nodes: BitSet,
        ignorable_nodes: BitSet,
    ) -> Self {
        let n = csr_offsets.len() - 1;

        // We reserve values with the leftmost bit set to 1 as free nodes and use the 0111..111 as
        // a marker for a null-free node (see self.forest documentation). Thus the number of nodes
        // is restricted to values less than 0111..111
        debug_assert!(n < FREE_SLOT_MASK as usize);

        let mut nodes = Vec::with_capacity(n + 1);

        let mut max_degree = 0;

        let mut read_ptr = 0usize;
        let mut write_ptr = 0usize;
        let mut ignorable_offset = 0;
        for u in 0..n {
            nodes.push(NodeInformation {
                offset_filtered: write_ptr as NumEdges,
                offset_unfiltered: ignorable_offset,
                free_pos: FREE_SLOT_MASK,
                ..Default::default()
            });

            while read_ptr < csr_offsets[u + 1] as usize {
                let node = edges[read_ptr];
                edges[write_ptr] = node;

                let not_removable = !removable_nodes.get_bit(node);
                write_ptr += (not_removable && !ignorable_nodes.get_bit(node)) as usize;
                ignorable_offset += not_removable as NumEdges;
                read_ptr += 1;
            }

            if !SORTED {
                edges[(nodes[u].offset_filtered as usize)..write_ptr].sort_unstable();
            }

            max_degree = max_degree.max(ignorable_offset - nodes[u].offset_unfiltered);
        }

        // Node (n + 1) does not exist, but to prevent unnecessary branching, we store an
        // additional NodeInformation to keep track of the final offsets
        nodes.push(NodeInformation {
            offset_filtered: write_ptr as NumEdges,
            offset_unfiltered: ignorable_offset,
            free_pos: FREE_SLOT_MASK,
            ..Default::default()
        });

        edges.truncate(write_ptr);

        let forest = vec![FREE_SLOT_MASK; ignorable_offset as usize];
        let data = edges.clone();

        let dirty_nodes = BitSet::new(max_degree as Node);

        Self {
            edges,
            nodes,
            forest,
            data,
            dirty_nodes,
        }
    }

    /// Returns the number of nodes in the underlying graph
    fn number_of_nodes(&self) -> NumNodes {
        (self.nodes.len() - 1) as NumNodes
    }

    /// Returns `true` if u owns a tree
    pub fn owns_tree(&self, u: Node) -> bool {
        node!(self, u).tree_len > 0
    }

    /// Returns `true` if u is inserted into a tree
    pub fn is_tree_node(&self, u: Node) -> bool {
        node!(self, u).data_len > 0
    }

    /// Returns `true` if u is neither the owner of a tree nor inserted into one
    pub fn is_unassigned(&self, u: Node) -> bool {
        !self.owns_tree(u) && !self.is_tree_node(u)
    }

    /// Returns the node at position pos in `Tree[u]`
    fn node_at(&self, u: Node, pos: usize) -> Node {
        debug_assert!(self.owns_tree(u));
        self.forest[node!(self, u).offset_unfiltered as usize + pos]
    }

    /// Returns `true` if u is a free node
    fn is_free_node(u: Node) -> bool {
        (u & !FREE_SLOT_MASK) > 0
    }

    /// Returns the associated data of the root node of `Tree[u]`
    ///
    /// If u does not own a tree, an empty slice is returned to prevent requiring additional checks.
    /// By (I2), if `Tree[u]` exists, `u` must be stored in the root and `Root\[Tree\[u\]\].len() >= 1`
    pub fn get_root_nodes(&self, u: Node) -> &[Node] {
        if node!(self, u).tree_len == 0 {
            return &[];
        }

        let root = self.forest[node!(self, u).offset_unfiltered as usize];
        data_ref!(self, root)
    }

    /// Costly function to find the owner of the tree where u is inserted.
    /// Only meant for Debug-Purposes and/or checks.
    fn find_owner(&self, u: Node) -> Node {
        debug_assert!(self.is_tree_node(u));
        let pos = node!(self, u).tree_pos as usize;
        (0..self.number_of_nodes())
            .find(|&v| {
                self.owns_tree(v)
                    && self.forest[node!(self, v).offset_unfiltered as usize + pos] == u
            })
            .unwrap()
    }

    /// Unbalanced case of intersect(dest, node), where one data-list is significantly longer than the other.
    fn intersect_unbalanced<const DEST: bool>(&mut self, dest: Node, other: Node) {
        // (I2) and (I3)
        debug_assert!(!data!(self, dest).is_empty() && !data!(self, other).is_empty());
        debug_assert!(data!(self, dest).is_sorted() && data!(self, other).is_sorted());

        let dest_len = node!(self, dest).data_len as usize;
        let other_len = node!(self, other).data_len as usize;

        let mut ptrw = 0usize;
        let mut ptrr = 0usize;

        if DEST {
            // dest_len < other_len
            let off_dest = node!(self, dest).offset_filtered as usize;

            for idx in 0..dest_len {
                let item = self.data[off_dest + idx];
                let result = data!(self, other)[ptrr..].binary_search(&item);

                // Branchless update of value
                data!(self, dest)[ptrw] = item;
                ptrw += result.is_ok() as usize;
                ptrr += result.unwrap_or_else(|x| x);

                if ptrr == other_len {
                    break;
                }
            }
        } else {
            // other_len > dest_len
            let off_other = node!(self, other).offset_filtered as usize;

            for idx in 0..other_len {
                let item = self.data[off_other + idx];
                let result = data!(self, dest)[ptrr..].binary_search(&item);

                // Branchless update of value
                // If Err(_) = result, then Data[dest][ptrw] <- 0 * item + 1 * Data[dest][ptrw] = Data[dest][ptrw]
                let is_ok = result.is_ok() as Node;
                data!(self, dest)[ptrw] = is_ok * item + (1 - is_ok) * data!(self, dest)[ptrw];
                ptrw += is_ok as usize;
                ptrr += result.unwrap_or_else(|x| x);

                if ptrr == dest_len {
                    break;
                }
            }
        }

        node!(self, dest).data_len = ptrw as NumNodes;
    }

    /// Balanced case of intersect(dest, node) where both data lists are roughly the same size
    fn intersect_balanced(&mut self, dest: Node, other: Node) {
        // (I2) and (I3)
        debug_assert!(!data!(self, dest).is_empty() && !data!(self, other).is_empty());
        debug_assert!(data!(self, dest).is_sorted() && data!(self, other).is_sorted());

        let off_dest = node!(self, dest).offset_filtered as usize;
        let off_other = node!(self, other).offset_filtered as usize;

        let dest_len = node!(self, dest).data_len as usize;
        let other_len = node!(self, other).data_len as usize;

        let mut ptrw = 0usize;
        let mut ptr1 = 0usize;
        let mut ptr2 = 0usize;

        while ptr1 < dest_len && ptr2 < other_len {
            let item1 = self.data[off_dest + ptr1];
            let item2 = self.data[off_other + ptr2];

            // Avoid branching
            ptr1 += (item1 <= item2) as usize;
            ptr2 += (item1 >= item2) as usize;

            self.data[off_dest + ptrw] = item1;
            ptrw += (item1 == item2) as usize;
        }

        node!(self, dest).data_len = ptrw as NumNodes;
    }

    /// Intersects associated data of dest and other and writes them into the associated data of dest.
    /// Returns `true` if the data in dest was modified by intersect.
    ///
    /// (I2, I3) Data[dest] and Data[other] are non-empty and sorted.
    fn intersect(&mut self, dest: Node, other: Node) -> bool {
        // (I2) and (I3)
        debug_assert!(!data!(self, dest).is_empty() && !data!(self, other).is_empty());
        debug_assert!(data!(self, dest).is_sorted() && data!(self, other).is_sorted());

        // Application of (I2): if any list has length 1, the element must be the owner of the
        // tree, which is common
        if node!(self, dest).data_len == 1 {
            return false;
        }
        if node!(self, other).data_len == 1 {
            data!(self, dest)[0] = data!(self, other)[0];
            node!(self, dest).data_len = 1;

            return true;
        }

        const SIZE_DIFF: NumNodes = 4;

        let dest_len = node!(self, dest).data_len;
        let other_len = node!(self, other).data_len;
        if dest_len << SIZE_DIFF <= other_len {
            self.intersect_unbalanced::<true>(dest, other);
        } else if other_len << SIZE_DIFF <= dest_len {
            self.intersect_unbalanced::<false>(dest, other);
        } else {
            self.intersect_balanced(dest, other);
        }

        // Intersections can only remove elements from dest and not add new ones.
        // Thus, if lengths remain unchanged, no elements were removed (or added) and dest remained
        // unchanged.
        dest_len != node!(self, dest).data_len
    }

    /// 'Restores' the associated data of a node by assigning its original neighborhood as data
    ///
    /// Warning: this breaks (I5)
    fn restore_node(&mut self, u: Node) {
        debug_assert!(!self.owns_tree(u));
        debug_assert!(!Self::is_free_node(u));

        let beg = node!(self, u).offset_filtered as usize;
        let len = node!(self, u + 1).offset_filtered as usize - beg;

        // SAFETY: by definition, intervals in edges and data are non-overlapping and are spaced
        // the same (i.e. by offset_filtered).
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.edges.as_ptr().add(beg),
                self.data.as_mut_ptr().add(beg),
                len,
            );
        }
        node!(self, u).data_len = len as NumNodes;
    }

    /// 'Repairs' the associated data of the node at position pos in Tree[u] by computing the intersection of its original data
    /// as well as those of its two (or one, or zero) children.
    ///
    /// Returns `true` if the length of the data was changed by repairing the node.
    ///
    /// Warning: this can break (I5) for the parent of u.
    fn repair_node(&mut self, u: Node, pos: usize) -> bool {
        debug_assert!(self.owns_tree(u));
        debug_assert!(node!(self, u).tree_len > pos as NumNodes);

        let node = self.node_at(u, pos);
        if Self::is_free_node(node) {
            return true;
        }
        let len = node!(self, node).data_len;
        self.restore_node(node);

        // Right child
        let right_child_pos = (pos << 1) + 2;
        if right_child_pos < node!(self, u).tree_len as usize {
            let right_child = self.node_at(u, right_child_pos);
            if !Self::is_free_node(right_child) {
                self.intersect(node, right_child);
            }
        }

        // Left child
        if right_child_pos <= node!(self, u).tree_len as usize {
            let left_child = self.node_at(u, right_child_pos - 1);
            if !Self::is_free_node(left_child) {
                self.intersect(node, left_child);
            }
        }

        len != node!(self, node).data_len
    }

    /// 'Repairs' Tree[u] upwards from node at position pos.
    /// Stops when no change was detected.
    ///
    /// Restores (I5) in the path from pos to root
    fn repair_up(&mut self, u: Node, mut pos: usize) {
        // We call repair_node when changing data at position pos.
        // Even if no length-change was detected, data can still differ from before and we thus
        // need to at least update its parent.
        self.repair_node(u, pos);
        while pos > 0 {
            pos = (pos - 1) >> 1;
            // Early returns when no change in data length was done.
            // This assumes that every node in the path from pos->root has unchanged children
            // (except at pos).
            if !self.repair_node(u, pos) {
                return;
            }
        }
    }

    /// 'Repairs' Tree[u] upwards from node at position pos.
    /// Stops when no change was detected.
    ///
    /// Returns `true` if at any point, position must_see_pos was repaired as well as its parent
    /// (if existent).
    fn repair_up_ensure_pos(&mut self, u: Node, mut pos: usize, must_see_pos: usize) -> bool {
        let mut pos_seen = pos == must_see_pos;
        self.repair_node(u, pos);
        while pos > 0 {
            pos = (pos - 1) >> 1;
            // If pos == must_see_pos, we return early before marking must_see_pos as true as we
            // need to update its parent (if existent) as well.
            if !self.repair_node(u, pos) {
                break;
            }

            pos_seen = pos_seen || pos == must_see_pos;
        }

        pos_seen
    }

    /// Inserts node `v` into the `Tree[u]`
    pub fn add_entry(&mut self, u: Node, mut v: Node) {
        debug_assert!(self.owns_tree(u) && !self.is_tree_node(v));

        let u_offset = node!(self, u).offset_unfiltered as usize;

        let pos;
        if node!(self, u).free_pos != FREE_SLOT_MASK {
            pos = node!(self, u).free_pos;
            node!(self, u).free_pos = self.forest[u_offset + pos as usize] & FREE_SLOT_MASK;
        } else {
            pos = node!(self, u).tree_len;
            node!(self, u).tree_len += 1;
        };

        node!(self, v).tree_pos = pos;

        let mut pos = pos as usize;
        // In both above cases, pos is the position of a leaf and we do not need to consider
        // future 'children' of v when updating the data as they do not exist
        self.forest[u_offset + pos] = v;
        self.restore_node(v);

        // Since we assume that (I5) is true beforehand, we do not resort to repair_up as
        // this resets every node and its path and re-computes its data from scratch
        while pos > 0 {
            pos = (pos - 1) >> 1;

            let parent = self.forest[u_offset + pos];
            // If the data did not change from before, no further elements must be removed in
            // cascading updates and we can stop early.
            if !self.intersect(parent, v) {
                break;
            }

            // Propagate node
            v = parent;
        }
    }

    /// Removes node `v` from `Tree[u]`
    pub fn remove_entry(&mut self, u: Node, v: Node) {
        debug_assert!(self.owns_tree(u) && self.is_tree_node(v));
        debug_assert_eq!(self.find_owner(v), u);

        let pos = node!(self, v).tree_pos;
        let tree_len = node!(self, u).tree_len;

        debug_assert_eq!(
            self.forest[node!(self, u).offset_unfiltered as usize + pos as usize],
            v
        );

        // Trees without nodes are considered empty and can be cleared
        if tree_len == 1 {
            self.clear_tree(u);
            return;
        }

        // If v is the last node in the tree, i.e. its rightmost leaf, remove it and repair upwards
        if pos == tree_len - 1 {
            node!(self, u).tree_len -= 1;
            self.repair_up(u, ((pos - 1) >> 1) as usize);
            return;
        }

        let u_offset = node!(self, u).offset_unfiltered as usize;
        let leaf_pos = self.find_leaf_pos(u, pos as usize);

        // If v is an inner leaf, i.e. a leaf, but not the rightmost leaf,
        // mark it as a free node and repair upwards
        if leaf_pos == pos as usize {
            if pos == 0 {
                self.clear_tree(u);
                return;
            }
            self.forest[u_offset + pos as usize] = node!(self, u).free_pos | !FREE_SLOT_MASK;
            node!(self, u).free_pos = pos;
            self.repair_up(u, ((pos - 1) >> 1) as usize);
            return;
        }

        // v is an inner node: swap with leaf and repair upwards from parent of original leaf.
        // The original position of v lies in the path from the leaf to the root and will thus
        // also be repaired on the way.
        let leaf = self.forest[u_offset + leaf_pos];
        self.forest[u_offset + pos as usize] = leaf;
        node!(self, leaf).tree_pos = pos;

        // If the leaf was the rightmost leaf in the tree, it does not have to be marked as free
        // and can be safely removed from the tree. Otherwise, mark it as free.
        if leaf_pos == (tree_len - 1) as usize {
            node!(self, u).tree_len -= 1;
        } else {
            self.forest[u_offset + leaf_pos] = node!(self, u).free_pos | !FREE_SLOT_MASK;
            node!(self, u).free_pos = leaf_pos as NumNodes;
        }

        // Repair from the leaf position. If the original position of the removed node was also
        // repaired (as well as its) parent, we can safely stop computing cascading intersections.
        // Otherwise, we need to start another repair_up procedure from the original position.
        if !self.repair_up_ensure_pos(u, (leaf_pos - 1) >> 1, pos as usize) {
            self.repair_up(u, pos as usize);
        }
    }

    /// Clears a tree, i.e. removes it from the forest.
    pub fn clear_tree(&mut self, u: Node) {
        debug_assert!(self.owns_tree(u));
        node!(self, u).tree_len = 0;
        node!(self, u).free_pos = FREE_SLOT_MASK;
    }

    /// Returns the position of a leaf in the subtree rooted at Tree[u][pos].
    /// This leaf must be a non-free node
    fn find_leaf_pos(&self, u: Node, mut pos: usize) -> usize {
        let tree_len = node!(self, u).tree_len as usize;
        let offset = node!(self, u).offset_unfiltered as usize;
        loop {
            let right_child_pos = (pos << 1) + 2;

            // No child exists
            if right_child_pos > tree_len {
                return pos;
            }

            // Left child exists but not right child
            if right_child_pos == tree_len {
                if self.forest[offset + right_child_pos - 1] < self.number_of_nodes() {
                    return right_child_pos - 1;
                }
                return pos;
            }

            // Right child exists and is set
            if self.forest[offset + right_child_pos] < self.number_of_nodes() {
                pos = right_child_pos;
                continue;
            }

            // Left child exists and is set
            if self.forest[offset + right_child_pos - 1] < self.number_of_nodes() {
                pos = right_child_pos - 1;
                continue;
            }

            return pos;
        }
    }

    /// Transfers ownership of `Tree[u]` to `v`, i.e. `Tree[v] = Tree[u]`, followed by `Tree[u].clear()`
    ///
    /// (I2) Every inserted node in `Tree[u]` is a neighbor of `v`, i.e. `Root\[Tree\[u\]\]` contains `v`.
    pub fn transfer_tree(&mut self, u: Node, v: Node) {
        // (I2)
        debug_assert!(self.get_root_nodes(u).contains(&v));

        let u_offset = node!(self, u).offset_unfiltered as usize;
        let v_offset = node!(self, v).offset_unfiltered as usize;

        // If Tree[u] does not fit into the reserved space of v, compress it to make it fit:
        // By (I2), every inserted node in `Tree[u]` is a neighbor of v and v has reserved space for
        // all its neighbors: if u had more neighbors that now are free nodes, it is possibly too
        // big counting all its free nodes
        let target_size = node!(self, v + 1).offset_unfiltered as NumNodes - v_offset as NumNodes;
        if target_size < node!(self, u).tree_len {
            self.compress_tree(u, target_size);
        }

        // Important: compute after compress_tree, as they both changed possibly
        let tree_len = node!(self, u).tree_len as usize;

        // SAFETY: by (I2), tree_len <= Neighbors[u].len(), Neighbors[v].len();
        // thus, intervals are non-overlapping and belong to u and v respectively.
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.forest.as_ptr().add(u_offset),
                self.forest.as_mut_ptr().add(v_offset),
                tree_len,
            );
        }

        node!(self, v).tree_len = node!(self, u).tree_len;
        node!(self, v).free_pos = node!(self, u).free_pos;
        self.clear_tree(u);
    }

    /// Compresses a tree by swapping nodes in lower levels with free nodes in higher levels such
    /// that the whole tree (including remaining free nodes) does not exceed a size of threshold in
    /// array representation.
    ///
    /// Requires threshold >= Tree[u].len() - Free[u].len()
    fn compress_tree(&mut self, u: Node, threshold: NumNodes) {
        let offset = node!(self, u).offset_unfiltered as usize;

        // Currently, BitSet only provides `get_first_set_index_atleast` to search the next set bit
        // from left to right. A similar funtion for the reverse case does not exist. Hence, we
        // simply flip the order of the bits to get the desired output
        let init_tree_len = node!(self, u).tree_len - 1;

        let mut last_free_pos = node!(self, u).free_pos;
        while node!(self, u).tree_len > threshold {
            let leaf = self.forest[offset + node!(self, u).tree_len as usize - 1];

            // Remove last element if empty
            if Self::is_free_node(leaf) {
                node!(self, u).tree_len -= 1;
                continue;
            }

            // Find free node position inside threshold
            while last_free_pos >= threshold {
                last_free_pos = self.forest[offset + last_free_pos as usize] & FREE_SLOT_MASK;
            }
            let prev_free_slot = self.forest[offset + last_free_pos as usize] & FREE_SLOT_MASK;

            // Move leaf further up the tree
            self.forest[offset + last_free_pos as usize] = leaf;
            node!(self, leaf).tree_pos = last_free_pos;

            node!(self, u).tree_len -= 1;
            self.dirty_nodes.set_bit(init_tree_len - last_free_pos);

            // Also add the parent of the added tree node into the bitset as we will definitely
            // need to update it later on.
            self.dirty_nodes
                .set_bit(init_tree_len - ((last_free_pos - 1) >> 1));
            self.dirty_nodes
                .set_bit(init_tree_len - ((node!(self, u).tree_len - 1) >> 1));

            last_free_pos = prev_free_slot;
        }

        // Repair tree
        let mut last_bit_pos = 0;
        while let Some(bit_pos) = self.dirty_nodes.get_first_set_index_atleast(last_bit_pos) {
            if bit_pos >= init_tree_len {
                break;
            }
            last_bit_pos = bit_pos;
            self.dirty_nodes.clear_bit(last_bit_pos);

            let bit_pos = init_tree_len - bit_pos;
            if bit_pos >= threshold {
                continue;
            }

            // If a the data of a node did not change, do not cause cascading updates.
            // Only for a node with a position change, we need to update its parent which is
            // ensured by inserting them in the previous part preemptively.
            if self.repair_node(u, bit_pos as usize) {
                let parent = (bit_pos - 1) >> 1;
                self.dirty_nodes.set_bit(init_tree_len - parent);
            }
        }

        // If remaining free nodes inside Tree[u] point to (previous) free node positions outside
        // of threshold, further compact the free node list to fit.
        while last_free_pos >= threshold && last_free_pos != FREE_SLOT_MASK {
            last_free_pos = self.forest[offset + last_free_pos as usize] & FREE_SLOT_MASK;
        }
        node!(self, u).free_pos = last_free_pos;

        if last_free_pos != FREE_SLOT_MASK {
            let mut current_head = self.forest[offset + last_free_pos as usize] & FREE_SLOT_MASK;
            while current_head != FREE_SLOT_MASK {
                if current_head < threshold {
                    self.forest[offset + last_free_pos as usize] = current_head | !FREE_SLOT_MASK;
                    last_free_pos = current_head;
                }
                current_head = self.forest[offset + current_head as usize] & FREE_SLOT_MASK;
            }
            self.forest[offset + last_free_pos as usize] = current_head | !FREE_SLOT_MASK;
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct NodeInfo {
    pub offset_unfiltered: NumEdges,
    pub offset_filtered: NumEdges,
    pub list_idx: NumNodes,
    pub list_len: NumNodes,
    pub list_pos: NumNodes,
}

/// IntersectionLists - Efficient queries into common neighborhoods with higher memory consumption
///
/// This data structure supports dynamic queries into common neighborhoods of select subsets of
/// nodes subject to some invariants listed below. That is, if a set S of nodes is inserted into
/// some IntersectionList T, the associated data of the highest bucket of T is the list of all nodes that are
/// incident to *every* node in S.
///
/// It is implemented by storing a list of all nodes partitioned by number of occurrences in
/// neighborhoods of inserted nodes.
///
/// See the GreedyReverseSearch heuristic for a more in-depth example.
///
/// The data structure relies on a series of invariants listed below
/// (I1) Every node can be inserted in at most one list at a time.
/// (I2) The owner of a tree is a neighbor of all nodes inserted into the tree.
/// (I3) Neighborhoods, i.e. associated data are sorted in increasing order (no multi-edges).
/// (I4) The number of trees is non-increasing over time
///
#[derive(Debug, Clone, Default)]
pub struct IntersectionLists {
    /// Number of Nodes
    n: usize,

    /// MaxDegree of Graph + 2: represents the maximum number of buckets (i.e. occurrences) that are
    /// possible. (+2) because we need a bucket 0 as well as a bucket MaxDegree + 1 to allow for
    /// branchless slice-access.
    delta: usize,

    /// List of all edges sorted by source, then by target
    edges: Vec<Node>,

    /// Important information for all nodes u
    /// * offset_filtered := first neighbor of u in edges <=> begin of associated data entries in data
    /// * offset_unfiltered := begin of owned list representation in forest   
    /// * list_idx := Index of List owned by node in self.buckets
    /// * list_len := length of the owned list: List[u] = lists[offset_unfiltered..(offset_unfiltered + list_len)]
    /// * list_pos := position in the list where u is inserted: if u in List[v], then lists[offset_unfiltered + ListPos[u]] = u and ListPos[u] < ListLen[v]
    ///
    /// To allow faster slice access without bound-checks, we store an artificial node (n + 1) with
    /// * offset_filtered = edges.len()
    /// * offset_unfiltered = lists.len()
    /// * .. = 0
    nodes: Vec<NodeInfo>,

    /// Inlined list of all inserted nodes
    ///
    /// Indexed by offset_unfiltered in self.nodes.
    lists: Vec<Node>,

    /// Nodes partitioned by
    /// (1) Associated List L
    /// (2) Number of occurrences in neighborhoods of inserted nodes of L
    buckets: Vec<Node>,

    /// Stores in which bucket and position a node is stored in the respective list.
    /// Partitioned by associated list.
    ///
    /// The position (second value) is independent of the bucket (although it has to be between the
    /// corresponding offsets).
    pointer: Vec<(NumNodes, NumNodes)>,

    /// Stores the offsets of the buckets for each list inlined into a single Vec.
    /// Partitioned by associated list. Here, every list only gets delta-entries as the number of
    /// buckets is upper bounded by delta.
    offsets: Vec<NumNodes>,
}

macro_rules! neighbors {
    ($self:ident, $node:ident) => {
        &$self.edges[($self.nodes[$node as usize].offset_filtered as usize)
            ..($self.nodes[$node as usize + 1].offset_filtered as usize)]
    };
}

impl IntersectionLists {
    /// Creates a new IntersectionLists from a CSR-Representation of the graph as well as two BitSets
    /// indicating which nodes will never appear in any way (owner/list-node/data-node) in this
    /// data structure (removable_nodes) and which nodes can be ignored in data-lists (ignorable_nodes).
    /// Also requires an initial list of all list owners.
    ///
    /// This assumes that the edges lists are sorted by source and target.
    pub fn new(
        csr_edges: Vec<Node>,
        csr_offsets: Vec<NumEdges>,
        removable_nodes: BitSet,
        ignorable_nodes: BitSet,
        initial_tree_owners: Vec<Node>,
    ) -> Self {
        Self::new_inner::<true>(
            csr_edges,
            csr_offsets,
            removable_nodes,
            ignorable_nodes,
            initial_tree_owners,
        )
    }

    /// Creates a new IntersectionLists from a Csr-Representation of the graph as well as two BitSets
    /// indicating which nodes will never appear in any way (owner/list-node/data-node) in this
    /// data structure (removable_nodes) and which nodes can be ignored in data-lists (ignorable_nodes).
    /// Also requires an initial list of all list owners.
    ///
    /// This will sort the edge lists by source and target when initializing.
    pub fn new_unsorted(
        csr_edges: Vec<Node>,
        csr_offsets: Vec<NumEdges>,
        removable_nodes: BitSet,
        ignorable_nodes: BitSet,
        initial_tree_owners: Vec<Node>,
    ) -> Self {
        Self::new_inner::<false>(
            csr_edges,
            csr_offsets,
            removable_nodes,
            ignorable_nodes,
            initial_tree_owners,
        )
    }

    fn new_inner<const SORTED: bool>(
        mut edges: Vec<Node>,
        csr_offsets: Vec<NumEdges>,
        removable_nodes: BitSet,
        ignorable_nodes: BitSet,
        initial_tree_owners: Vec<Node>,
    ) -> Self {
        let n = csr_offsets.len() - 1;

        // We reserve values with the leftmost bit set to 1 as free nodes and use the 0111..111 as
        // a marker for a null-free node (see self.forest documentation). Thus the number of nodes
        // is restricted to values less than 0111..111
        debug_assert!(n < FREE_SLOT_MASK as usize);

        let mut nodes = Vec::with_capacity(n + 1);

        let mut max_degree = 0;

        let mut read_ptr = 0usize;
        let mut write_ptr = 0usize;
        let mut ignorable_offset = 0;
        for u in 0..n {
            nodes.push(NodeInfo {
                offset_filtered: write_ptr as NumEdges,
                offset_unfiltered: ignorable_offset,
                list_idx: NOT_SET,
                list_len: 0,
                list_pos: 0,
            });

            while read_ptr < csr_offsets[u + 1] as usize {
                let node = edges[read_ptr];
                edges[write_ptr] = node;

                let not_removable = !removable_nodes.get_bit(node);
                write_ptr += (not_removable && !ignorable_nodes.get_bit(node)) as usize;
                ignorable_offset += not_removable as NumEdges;
                read_ptr += 1;
            }

            if !SORTED {
                edges[(nodes[u].offset_filtered as usize)..write_ptr].sort_unstable();
            }

            max_degree = max_degree.max(ignorable_offset - nodes[u].offset_unfiltered);
        }

        // Node (n + 1) does not exist, but to prevent unnecessary branching, we store an
        // additional NodeInformation to keep track of the final offsets
        nodes.push(NodeInfo {
            offset_filtered: write_ptr as NumEdges,
            offset_unfiltered: ignorable_offset,
            list_idx: NOT_SET,
            list_len: 0,
            list_pos: 0,
        });

        edges.truncate(write_ptr);

        let num_owners = initial_tree_owners.len();

        let lists = vec![NOT_SET; ignorable_offset as usize];
        let buckets = (0..(n * num_owners))
            .map(|x| (x % n) as Node)
            .collect::<Vec<Node>>();
        let pointer = (0..(n * num_owners))
            .map(|x| (0, (x % n) as NumNodes))
            .collect();
        let mut offsets = vec![n as NumNodes; (max_degree as usize + 2) * num_owners];

        // Assign initial lists
        for (i, u) in initial_tree_owners.into_iter().enumerate() {
            nodes[u as usize].list_idx = i as NumNodes;
            offsets[i * (max_degree as usize + 2)] = 0;
        }

        Self {
            n,
            delta: max_degree as usize + 2,
            edges,
            nodes,
            lists,
            buckets,
            pointer,
            offsets,
        }
    }

    /// Returns the associated data of `List[u]`, i.e. the set of nodes that are neighbored by all
    /// inserted nodes of `List[u]`.
    ///
    /// If u does not own a list, an empty slice is returned to prevent requiring additional checks.    
    pub fn get_root_nodes(&self, u: Node) -> &[Node] {
        let len = self.nodes[u as usize].list_len as usize;
        if len == 0 {
            return &[];
        }

        let list = self.nodes[u as usize].list_idx as usize;
        debug_assert!(list != NOT_SET as usize);

        &self.buckets[(list * self.n + self.offsets[list * self.delta + len] as usize)
            ..(list * self.n + self.offsets[list * self.delta + len + 1] as usize)]
    }

    /// Inserts node v into `List[u]`
    pub fn add_entry(&mut self, u: Node, v: Node) {
        debug_assert!(self.nodes[u as usize].list_idx != NOT_SET);

        // Updates lists
        self.lists[self.nodes[u as usize].offset_unfiltered as usize
            + self.nodes[u as usize].list_len as usize] = v;
        self.nodes[v as usize].list_pos = self.nodes[u as usize].list_len;
        self.nodes[u as usize].list_len += 1;

        // Faster indexing
        let list = self.nodes[u as usize].list_idx as usize;
        let buckets = &mut self.buckets[(list * self.n)..((list + 1) * self.n)];
        let pointer = &mut self.pointer[(list * self.n)..((list + 1) * self.n)];
        let offsets = &mut self.offsets[(list * self.delta)..((list + 1) * self.delta)];

        // Move every neighbor of v one bucket up
        for &w in neighbors!(self, v) {
            let (bucket, position) = pointer[w as usize];
            debug_assert!(bucket <= self.nodes[u as usize].list_len);
            debug_assert_eq!(buckets[position as usize], w);
            debug_assert!(
                offsets[bucket as usize] <= position && position < offsets[bucket as usize + 1]
            );

            offsets[bucket as usize + 1] -= 1;

            let swap_position = offsets[bucket as usize + 1];
            buckets.swap(position as usize, swap_position as usize);

            pointer[buckets[position as usize] as usize].1 = position;
            pointer[w as usize] = (bucket + 1, swap_position);
        }
    }

    /// Removes node v from `List[u]`
    pub fn remove_entry(&mut self, u: Node, v: Node) {
        debug_assert!(self.nodes[u as usize].list_idx != NOT_SET);

        // Updates lists
        self.nodes[u as usize].list_len -= 1;
        let orig_pos = self.nodes[v as usize].list_pos;
        let swap_node = self.lists[self.nodes[u as usize].offset_unfiltered as usize
            + self.nodes[u as usize].list_len as usize];
        self.lists[self.nodes[u as usize].offset_unfiltered as usize + orig_pos as usize] =
            swap_node;
        self.nodes[swap_node as usize].list_pos = orig_pos;
        self.nodes[v as usize].list_pos = NOT_SET;

        // Faster indexing
        let list = self.nodes[u as usize].list_idx as usize;
        let buckets = &mut self.buckets[(list * self.n)..((list + 1) * self.n)];
        let pointer = &mut self.pointer[(list * self.n)..((list + 1) * self.n)];
        let offsets = &mut self.offsets[(list * self.delta)..((list + 1) * self.delta)];

        // Move every neighbor of v one bucket down
        for &w in neighbors!(self, v) {
            let (bucket, position) = pointer[w as usize];
            debug_assert!(bucket > 0);
            debug_assert_eq!(buckets[position as usize], w);
            debug_assert!(
                offsets[bucket as usize] <= position && position < offsets[bucket as usize + 1]
            );

            let swap_position = offsets[bucket as usize];
            buckets.swap(position as usize, swap_position as usize);

            offsets[bucket as usize] += 1;

            pointer[buckets[position as usize] as usize].1 = position;
            pointer[w as usize] = (bucket - 1, swap_position);
        }
    }

    /// Transfer ownership of `List[u]` to `v`
    pub fn transfer_tree(&mut self, u: Node, v: Node) {
        debug_assert!(
            self.nodes[u as usize].list_idx != NOT_SET
                && self.nodes[v as usize].list_idx == NOT_SET
        );

        let list_len = self.nodes[u as usize].list_len as usize;

        // SAFETY: by (I2), `tree_len <= Neighbors[u].len()`, `Neighbors[v].len()`;
        // thus, intervals are non-overlapping and belong to u and v respectively.
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.lists
                    .as_ptr()
                    .add(self.nodes[u as usize].offset_unfiltered as usize),
                self.lists
                    .as_mut_ptr()
                    .add(self.nodes[v as usize].offset_unfiltered as usize),
                list_len,
            );
        }

        self.nodes[u as usize].list_len = 0;
        self.nodes[v as usize].list_len = list_len as NumNodes;
        self.nodes[v as usize].list_idx = self.nodes[u as usize].list_idx;
        self.nodes[u as usize].list_idx = NOT_SET;
    }

    /// Clears `List[u]`, i.e. removes it from tracking.
    /// This essentially deletes a list forever.
    pub fn clear_tree(&mut self, u: Node) {
        self.nodes[u as usize].list_idx = NOT_SET;
    }
}
