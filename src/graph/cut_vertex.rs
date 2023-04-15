use super::*;

pub trait ArticluationPoint {
    fn compute_articulation_points(&self) -> BitSet;
}

impl<G> ArticluationPoint for G
where
    G: AdjacencyList,
{
    fn compute_articulation_points(&self) -> BitSet {
        ArticulationPointSearch::new(self).compute()
    }
}

pub struct ArticulationPointSearch<'a, T: AdjacencyList> {
    graph: &'a T,
    low_point: Vec<Node>,
    dfs_num: Vec<Node>,
    visited: BitSet,
    articulation_points: BitSet,
    current_dfs_num: Node,
    parent: Vec<Option<Node>>,
}

impl<'a, T: AdjacencyList> ArticulationPointSearch<'a, T> {
    /// Assumes the graph is connected, and for each edge (u, v) the edge (v, u) exists
    pub fn new(graph: &'a T) -> Self {
        let n = graph.number_of_nodes();
        Self {
            graph,
            low_point: vec![0; n as usize],
            dfs_num: vec![0; n as usize],
            visited: BitSet::new(n),
            parent: vec![None; n as usize],
            articulation_points: BitSet::new(n),
            current_dfs_num: 0,
        }
    }

    pub fn compute(mut self) -> BitSet {
        let _ = self.compute_recursive(0, 0);
        self.articulation_points
    }

    fn compute_recursive(&mut self, u: Node, depth: Node) -> Result<(), ()> {
        if depth > 10000 {
            return Err(());
        }

        self.visited.set_bit(u);
        self.current_dfs_num += 1;
        self.dfs_num[u as usize] = self.current_dfs_num;
        self.low_point[u as usize] = self.current_dfs_num;

        // counts number of tree neighbors
        let mut tree_neighbors = 0;
        for v in self.graph.neighbors_of(u) {
            // tree edge
            if !self.visited.get_bit(v) {
                tree_neighbors += 1;
                self.parent[v as usize] = Some(u);
                self.compute_recursive(v, depth + 1)?;
                self.low_point[u as usize] =
                    self.low_point[u as usize].min(self.low_point[v as usize]);

                if self.parent[u as usize].is_some()
                    && self.low_point[v as usize] >= self.dfs_num[u as usize]
                {
                    self.articulation_points.set_bit(u);
                }
            } else {
                // back edge, update value if v is not the parent
                if self.parent[u as usize].is_none() || self.parent[u as usize].unwrap() != v {
                    self.low_point[u as usize] =
                        self.low_point[u as usize].min(self.dfs_num[v as usize]);
                }
            }
        }

        if self.parent[u as usize].is_none() && tree_neighbors > 1 {
            self.articulation_points.set_bit(u);
        }

        Ok(())
    }
}
