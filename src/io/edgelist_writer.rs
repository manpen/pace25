use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use super::super::graph::*;

pub trait EdgelistWriter {
    fn try_write_edgelist<W: Write>(&self, writer: W) -> Result<(), std::io::Error>;
    fn try_write_edgelist_file<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error>;
}

impl<T> EdgelistWriter for T
where
    T: ColoredAdjacencyList + GraphEdgeOrder,
{
    fn try_write_edgelist<W: Write>(&self, mut writer: W) -> Result<(), std::io::Error> {
        writeln!(writer, "Source,Target,Color\n")?;
        for ColoredEdge(u, v, c) in self.ordered_colored_edges(true) {
            writeln!(
                writer,
                "{u},{v},{}",
                if c.is_black() { "black" } else { "red" }
            )?; //, )?;
        }

        Ok(())
    }

    fn try_write_edgelist_file<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let writer = BufWriter::new(File::create(path)?);
        self.try_write_edgelist(writer)
    }
}
