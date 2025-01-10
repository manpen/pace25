use std::io::Write;

use super::super::graph::*;

/// produces a minimalistic DOT representation of the graph
pub trait DotWriter {

    fn try_write_dot<W: Write>(&self, writer: W) -> Result<(), std::io::Error>;
}

impl<T> DotWriter for T
where
    T: ColoredAdjacencyList,
{
    fn try_write_dot<W: Write>(&self, mut writer: W) -> Result<(), std::io::Error> {
        write!(writer, "graph G {{")?;
        for ColoredEdge(u, v, c) in self.ordered_colored_edges(true) {
            if c.is_red() {
                write!(writer, "v{u}--v{v}[color=red]; ")?;
            } else {
                write!(writer, "v{u}--v{v}; ")?;
            }
        }
        write!(writer, r"}}")
    }
}
