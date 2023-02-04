use super::*;

pub trait ColorFilter: Sized {
    fn red_only(&self) -> ColorFiltered<'_, Self> {
        ColorFiltered {
            filter: EdgeColorFilter::RedOnly,
            graph: self,
        }
    }

    fn black_only(&self) -> ColorFiltered<'_, Self> {
        ColorFiltered {
            filter: EdgeColorFilter::BlackOnly,
            graph: self,
        }
    }
}

pub struct ColorFiltered<'a, G> {
    filter: EdgeColorFilter,
    graph: &'a G,
}

macro_rules! forward {
    ($func : ident, $type : ty) => {
        fn $func(&self, node: Node) -> $type {
            paste::paste! {
                match self.filter {
                EdgeColorFilter::BlackOnly => self.graph.[<black_ $func>](node),
                EdgeColorFilter::RedOnly => self.graph.[<red_ $func>](node),
                EdgeColorFilter::BlackAndRed => self.graph.$func(node),
                }
            }
        }
    };
}

impl<'a, G> GraphNodeOrder for ColorFiltered<'a, G>
where
    G: GraphNodeOrder,
{
    type VertexIter<'b> = G::VertexIter<'b>
    where
        Self: 'b;

    fn number_of_nodes(&self) -> Node {
        self.graph.number_of_nodes()
    }

    fn vertices(&self) -> Self::VertexIter<'_> {
        self.graph.vertices()
    }

    fn len(&self) -> usize {
        self.graph.len()
    }

    fn vertices_range(&self) -> Range<Node> {
        self.graph.vertices_range()
    }

    fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }
}

impl<'a, G> AdjacencyList for ColorFiltered<'a, G>
where
    G: ColoredAdjacencyList,
{
    forward!(neighbors_of, &[Node]);
    forward!(degree_of, NumNodes);
}

impl<'a, G> AdjacencyTest for ColorFiltered<'a, G>
where
    G: ColoredAdjacencyTest,
{
    fn has_edge(&self, u: Node, v: Node) -> bool {
        match self.filter {
            EdgeColorFilter::BlackOnly => self.graph.has_black_edge(u, v),
            EdgeColorFilter::RedOnly => self.graph.has_red_edge(u, v),
            EdgeColorFilter::BlackAndRed => self.graph.has_edge(u, v),
        }
    }
}
