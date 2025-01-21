use crate::graph::{Node, NumEdges, NumNodes};

#[derive(Debug, Clone)]
struct MergeEntry {
    data: Vec<Node>,
    value: Node,
}

#[derive(Debug, Clone)]
pub struct MergeTree {
    tree_nodes: Vec<Option<MergeEntry>>,
    free_nodes: Vec<Node>,
    owner: Node,
}

impl MergeTree {
    pub fn new(owner: Node) -> Self {
        Self {
            tree_nodes: Vec::new(),
            free_nodes: Vec::new(),
            owner,
        }
    }

    pub fn clear(&mut self) {
        self.free_nodes.clear();
        self.tree_nodes.clear();
    }

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

    pub fn get_root_nodes(&self) -> &[Node] {
        if self.tree_nodes.is_empty() {
            return &[];
        }

        match &self.tree_nodes[0] {
            Some(entry) => &entry.data,
            None => &[],
        }
    }

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

#[derive(Debug, Clone)]
pub struct MergeTrees {
    trees: Vec<MergeTree>,
    edges: Vec<Node>,
    index: Vec<NumNodes>,
    offsets: Vec<NumEdges>,
    positions: Vec<NumNodes>,
}

impl MergeTrees {
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

    pub fn get_root_nodes(&self, u: Node) -> &[Node] {
        if self.index[u as usize] == NumNodes::MAX {
            return &[];
        }
        self.trees[self.index[u as usize] as usize].get_root_nodes()
    }

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

    pub fn transfer(&mut self, u: Node, v: Node) {
        self.index[v as usize] = self.index[u as usize];
        self.index[u as usize] = NumNodes::MAX;
        self.trees[self.index[v as usize] as usize].owner = v;
    }
}
