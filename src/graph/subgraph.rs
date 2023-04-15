use super::*;

pub trait InducedSubgraph: Sized {
    /// Returns a new graph instance containing all nodes i with vertices\[i\] == true
    fn vertex_induced_as<M, Gout>(&self, vertices: &BitSet) -> (Gout, M)
    where
        M: node_mapper::Getter + node_mapper::Setter,
        Gout: GraphNew + GraphEdgeEditing;

    fn vertex_induced(&self, vertices: &BitSet) -> (Self, NodeMapper)
    where
        Self: GraphEdgeEditing,
    {
        self.vertex_induced_as(vertices)
    }

    /// Creates a subgraph where all nodes without edges are removed
    fn remove_disconnected_verts(&self) -> (Self, NodeMapper)
    where
        Self: GraphNew + GraphEdgeEditing + AdjacencyList + ColoredAdjacencyList,
    {
        self.vertex_induced(&BitSet::new_with_bits_cleared(
            self.number_of_nodes(),
            self.vertices().filter(|&u| self.degree_of(u) == 0),
        ))
    }
}

impl<G: GraphNew + GraphEdgeEditing + ColoredAdjacencyList + Sized> InducedSubgraph for G {
    fn vertex_induced_as<M, Gout>(&self, vertices: &BitSet) -> (Gout, M)
    where
        M: node_mapper::Getter + node_mapper::Setter,
        Gout: GraphNew + GraphEdgeEditing,
    {
        let new_n = vertices.cardinality();
        let mut result = Gout::new(new_n);

        // compute new node ids
        let mut mapping = M::with_capacity(new_n);
        for (new, old) in vertices.iter_set_bits().enumerate() {
            mapping.map_node_to(old, new as Node);
            assert!(new_n > new as Node);
        }

        for u in self.vertices() {
            if let Some(new_u) = mapping.new_id_of(u) {
                result.add_colored_edges(self.colored_edges_of(u, true).filter_map(
                    |ColoredEdge(_, v, c)| Some(ColoredEdge(new_u, mapping.new_id_of(v)?, c)),
                ));
            }
        }

        (result, mapping)
    }
}

pub trait SubGraph {
    fn sub_graph<H>(&self, vertices: &BitSet) -> H
    where
        H: ColoredAdjacencyList + GraphEdgeEditing + Clone + GraphNew;
}

impl<G: ColoredAdjacencyList + GraphEdgeEditing + Clone + GraphNew> SubGraph for G {
    /// creates a subgraph of a graph induced by some nodes (vertices)
    fn sub_graph<H: AdjacencyList + GraphEdgeEditing + Clone + GraphNew>(
        &self,
        vertices: &BitSet,
    ) -> H {
        let mut sub_graph: H = H::new(self.number_of_nodes());

        sub_graph.add_colored_edges(vertices.iter_set_bits().flat_map(|u| {
            self.colored_edges_of(u, true)
                .filter(|ColoredEdge(_, v, _)| vertices.get_bit(*v))
        }));

        sub_graph
    }
}
