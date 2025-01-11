use super::*;
use itertools::Itertools;

pub type PartitionClass = Node;

/// A partition splits a graph into node-disjoint substructures (think SCCs, bipartite classes, etc)
pub struct Partition {
    // Remark on the encoding: in a perfect world `classes` should contain `Option<PartitionClass>`
    // to encode "unassigned" nodes. As of writing, this is extremely wasteful since `PartitionClass`
    // requires 4 bytes, while `Option<PartitionClass>` takes 8 bytes (due to padding for alignment).
    // We hence treat class 0 as unassigned and hide that from the user. Partition class `i` is
    // then mapped to the internal class `i+1`; we use this mapping (e.g. instead of encoding
    // unassigned with MAXINT) to simplify the interplay between `classes` and `class_sizes`.
    classes: Vec<PartitionClass>,
    class_sizes: Vec<Node>,
}

impl Partition {
    /// Creates a partition for `nodes` nodes which are initially all unassigned
    ///
    /// # Example
    /// ```
    /// use dss::graph::partition::*;
    /// let partition = Partition::new(10);
    /// assert_eq!(partition.number_of_unassigned(), 10);
    /// ```
    pub fn new(nodes: Node) -> Self {
        Self {
            classes: vec![0; nodes as usize],
            class_sizes: vec![nodes],
        }
    }

    /// Creates a new partition class and assigns all provided nodes to it; we require that these
    /// nodes were previously unassigned.
    ///
    /// # Example
    /// ```
    /// use dss::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// let class_id = partition.add_class([2,4]);
    /// assert_eq!(partition.number_of_unassigned(), 8);
    /// assert_eq!(partition.number_in_class(class_id), 2);
    /// assert_eq!(partition.class_of_edge(1, 2), None);
    /// assert_eq!(partition.class_of_edge(2, 4), Some(class_id));
    /// ```
    pub fn add_class<I: IntoIterator<Item = Node>>(&mut self, nodes: I) -> PartitionClass {
        let class_id = self.class_sizes.len() as PartitionClass;
        self.class_sizes.push(0);

        let size = &mut self.class_sizes[class_id as usize];
        for u in nodes {
            assert_eq!(self.classes[u as usize], 0); // check that node is unassigned
            self.classes[u as usize] = class_id;
            *size += 1;
        }

        self.class_sizes[0] -= *size;

        class_id - 1
    }

    /// Moves node into an existing partition class. The node may or may not have been previously assigned.
    ///
    /// # Example
    /// ```
    /// use dss::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// let class_id = partition.add_class([2,4]);
    /// partition.move_node(1, class_id);
    /// assert_eq!(partition.number_of_unassigned(), 7);
    /// assert_eq!(partition.number_in_class(class_id), 3);
    /// assert_eq!(partition.class_of_edge(1, 2), Some(class_id));
    /// ```
    pub fn move_node(&mut self, node: Node, new_class: PartitionClass) {
        self.class_sizes[self.classes[node as usize] as usize] -= 1;
        self.classes[node as usize] = new_class + 1;
        self.class_sizes[self.classes[node as usize] as usize] += 1;
    }

    /// Returns the class identifier of node `node` or `None` if `node` is unassigned
    ///
    /// # Example
    /// ```
    /// use dss::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// let class_id = partition.add_class([2,4]);
    /// assert_eq!(partition.class_of_node(1), None);
    /// assert_eq!(partition.class_of_node(2), Some(class_id));
    /// ```
    pub fn class_of_node(&self, node: Node) -> Option<PartitionClass> {
        let class_id = self.classes[node as usize];
        if class_id == 0 {
            None
        } else {
            Some(class_id - 1)
        }
    }

    /// Returns the class identifier if both nodes `u` and `v` are assigned to the same class
    /// and `None` otherwise.
    ///
    /// # Example
    /// ```
    /// use dss::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// let c1 = partition.add_class([2,4]);
    /// let c2 = partition.add_class([6,8]);
    /// assert_eq!(partition.class_of_edge(0, 1), None); // both unassigned
    /// assert_eq!(partition.class_of_edge(0, 2), None); // 0 unassigned
    /// assert_eq!(partition.class_of_edge(4, 6), None); // assigned to different classes
    /// assert_eq!(partition.class_of_edge(2, 4), Some(c1));
    /// assert_eq!(partition.class_of_edge(4, 2), Some(c1));
    /// assert_eq!(partition.class_of_edge(6, 8), Some(c2));
    /// ```
    pub fn class_of_edge(&self, u: Node, v: Node) -> Option<PartitionClass> {
        let cu = self.class_of_node(u)?;
        let cv = self.class_of_node(v)?;
        if cu == cv {
            Some(cu)
        } else {
            None
        }
    }

    /// Returns the number of unassigned nodes
    ///
    /// # Example
    /// ```
    /// use dss::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// assert_eq!(partition.number_of_unassigned(), 10);
    /// partition.add_class([2,4]);
    /// assert_eq!(partition.number_of_unassigned(), 8);
    /// ```
    pub fn number_of_unassigned(&self) -> Node {
        self.class_sizes[0]
    }

    /// Returns the number of nodes in class `class_id`
    ///
    /// # Example
    /// ```
    /// use dss::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// let class_id = partition.add_class([2,4]);
    /// assert_eq!(partition.number_in_class(class_id), 2);
    /// ```
    pub fn number_in_class(&self, class_id: PartitionClass) -> Node {
        self.class_sizes[class_id as usize + 1]
    }

    /// Returns the number of partition classes (0 if all nodes are unassigned)
    ///
    /// # Example
    /// ```
    /// use dss::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// assert_eq!(partition.number_of_classes(), 0);
    /// partition.add_class([2,4]);
    /// assert_eq!(partition.number_of_classes(), 1);
    /// ```
    pub fn number_of_classes(&self) -> Node {
        self.class_sizes.len() as Node - 1
    }

    /// Returns the members of a partition class in order.
    ///
    /// # Warning
    /// This operation is expensive and requires time linear in the total number of nodes, i.e. it
    /// is roughly independent of the actual size of partition class `class_id`.
    ///
    /// # Example
    /// ```
    /// use dss::graph::partition::*;
    /// use itertools::Itertools;;
    /// let mut partition = Partition::new(10);
    /// let class_id = partition.add_class([2,5,4]);
    /// assert_eq!(partition.members_of_class(class_id).collect_vec(), vec![2,4,5]);
    /// ```
    pub fn members_of_class(&self, class_id: Node) -> impl Iterator<Item = Node> + '_ {
        let class_id = class_id + 1;
        assert!(self.class_sizes.len() > class_id as usize);
        self.classes.iter().enumerate().filter_map(move |(i, &c)| {
            if c == class_id {
                Some(i as Node)
            } else {
                None
            }
        })
    }

    /// Splits the input graph `graph` (has to have the same number of nodes as `self`) into
    /// one subgraph per partition class; the `result[i]` corresponds to partition class `i`.
    pub fn split_into_subgraphs_as<GI, GO, M>(&self, graph: &GI) -> Vec<(GO, M)>
    where
        GI: ColoredAdjacencyList,
        GO: GraphNew + GraphEdgeEditing,
        M: node_mapper::Setter + node_mapper::Getter,
    {
        assert_eq!(graph.len(), self.classes.len());

        // Create an empty graph and mapper with the capacity for each partition class
        let mut result = (0..self.number_of_classes())
            .map(|class_id| {
                let n = self.number_in_class(class_id);
                (GO::new(n as NumNodes), M::with_capacity(n))
            })
            .collect_vec();

        // Iterator over all (assigned) nodes and map them into their respective subgraph
        let mut nodes_mapped_in_class = vec![0; self.number_of_classes() as usize];
        for (u, &class_id) in self.classes.iter().enumerate() {
            if class_id == 0 {
                // u is unassigned
                continue;
            }

            let class_id = (class_id - 1) as usize;

            result[class_id]
                .1
                .map_node_to(u as Node, nodes_mapped_in_class[class_id]);
            nodes_mapped_in_class[class_id] += 1;
        }

        // Iterate over all edges incident to assigned nodes
        for (u, &class_id) in self.classes.iter().enumerate().filter(|(_, &c)| c > 0) {
            let u = u as Node;
            let result_containg_u = &mut result[class_id as usize - 1];

            let mapped_u = result_containg_u.1.new_id_of(u).unwrap();

            // Iterate over all out-neighbors of u that are in the same partition class
            for ColoredEdge(_, v, color) in graph
                .colored_edges_of(u, true)
                .filter(|ColoredEdge(_, v, _)| self.classes[*v as usize] == class_id)
            {
                let mapped_v = result_containg_u.1.new_id_of(v).unwrap();
                result_containg_u.0.add_edge(mapped_u, mapped_v, color);
            }
        }

        result
    }

    /// Shorthand for [`Partition::split_into_subgraphs_as`]
    pub fn split_into_subgraphs<G>(&self, graph: &G) -> Vec<(G, NodeMapper)>
    where
        G: ColoredAdjacencyList + GraphNew + GraphEdgeEditing,
    {
        self.split_into_subgraphs_as(graph)
    }
}

impl From<BitSet> for Partition {
    fn from(set: BitSet) -> Self {
        let mut part = Partition::new(set.number_of_bits());

        assert_eq!(part.add_class(std::iter::empty()), 0);
        assert_eq!(part.add_class(std::iter::empty()), 1);

        for i in 0..set.number_of_bits() {
            part.move_node(i, set.get_bit(i) as u32);
        }

        part
    }
}
