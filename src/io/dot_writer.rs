use std::io::Write;

use super::super::graph::*;

/// produces a minimalistic DOT representation of the graph
pub trait DotWriter {
    fn try_write_dot<W: Write>(&self, writer: W) -> Result<(), std::io::Error>;
}

impl<T> DotWriter for T
where
    T: AdjacencyList,
{
    fn try_write_dot<W: Write>(&self, mut writer: W) -> Result<(), std::io::Error> {
        write!(writer, "graph G {{")?;
        for Edge(u, v) in self.ordered_edges(true) {
            write!(writer, "v{u}--v{v}; ")?;
        }
        write!(writer, r"}}")
    }
}

pub trait DotNodeWriter {
    fn try_write_dot_nodes<W: Write>(
        &mut self,
        writer: W,
        node_prefix: &str,
        color: &str,
    ) -> Result<(), std::io::Error>;
}

impl<T> DotNodeWriter for T
where
    T: Iterator<Item = Node>,
{
    fn try_write_dot_nodes<W: Write>(
        &mut self,
        mut writer: W,
        node_prefix: &str,
        color: &str,
    ) -> Result<(), std::io::Error> {
        for u in self {
            write!(
                writer,
                "{node_prefix}{u} [style=filled, fillcolor={color}];"
            )?;
        }
        writeln!(writer)
    }
}
