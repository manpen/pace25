use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use super::super::graph::*;

pub trait PaceWriter {
    fn try_write_pace<W: Write>(&self, writer: W) -> Result<(), std::io::Error>;
    fn try_write_pace_file<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error>;
}

impl<T> PaceWriter for T
where
    T: AdjacencyList + GraphEdgeOrder,
{
    fn try_write_pace<W: Write>(&self, mut writer: W) -> Result<(), std::io::Error> {
        writeln!(
            writer,
            "p ds {} {}",
            self.number_of_nodes(),
            self.number_of_edges()
        )?;

        for Edge(u, v) in self.ordered_edges(true) {
            writeln!(writer, "{} {}", u + 1, v + 1)?;
        }

        Ok(())
    }

    fn try_write_pace_file<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let writer = BufWriter::new(File::create(path)?);
        self.try_write_pace(writer)
    }
}

#[cfg(test)]
mod test {
    use crate::io::GraphPaceReader;

    use super::*;
    use itertools::Itertools;
    use rand::{Rng, SeedableRng};
    use regex::Regex;

    #[test]
    fn hard_coded() {
        let mut graph = AdjArray::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(3, 2);

        let output = {
            let mut buffer: Vec<u8> = Vec::new();
            graph.try_write_pace(&mut buffer).expect("Failed to write");
            String::from_utf8(buffer).unwrap()
        };

        assert!(
            Regex::new(r"p\sds\s4\s2")
                .unwrap()
                .is_match(output.as_str())
        );
        assert!(
            Regex::new(r"1\s2").unwrap().is_match(output.as_str()),
            "Output: {output}"
        );
        assert!(
            Regex::new(r"3\s4").unwrap().is_match(output.as_str()),
            "Output: {output}"
        );
    }

    #[test]
    fn transcribe() {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(1234);
        for n in 0..100 {
            let p = rng.gen_range(0.01..0.99);

            let org = AdjArray::random_gnp(&mut rng, n, p);

            let mut buffer: Vec<u8> = Vec::new();
            org.try_write_pace(&mut buffer).expect("Failed to write");

            let read = AdjArray::try_read_pace(buffer.as_slice()).expect("Failed to read");

            assert_eq!(org.number_of_nodes(), read.number_of_nodes());
            assert_eq!(
                org.ordered_edges(true).collect_vec(),
                read.ordered_edges(true).collect_vec()
            );
        }
    }
}
