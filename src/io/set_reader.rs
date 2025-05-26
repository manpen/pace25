use std::{
    fs::File,
    io::{BufRead, BufReader, ErrorKind, Lines},
    path::Path,
};

use crate::graph::{Edge, GraphFromReader, NumEdges, NumNodes};

pub type Result<T> = std::io::Result<T>;

pub trait SetPaceReader: Sized {
    fn try_read_set_pace<R: BufRead>(reader: R) -> Result<(Self, NumNodes)>;
    fn try_read_set_pace_file<P: AsRef<Path>>(path: P) -> Result<(Self, NumNodes)>;
}

impl<G> SetPaceReader for G
where
    G: GraphFromReader,
{
    fn try_read_set_pace<R: BufRead>(reader: R) -> Result<(Self, NumNodes)> {
        let pace_reader = SetReader::try_new(reader)?;
        let n = pace_reader.number_of_nodes();
        let m = pace_reader.number_of_edges();
        Ok((G::from_edges(n + m as NumNodes, pace_reader), n))
    }

    fn try_read_set_pace_file<P: AsRef<Path>>(path: P) -> Result<(Self, NumNodes)> {
        let reader = File::open(path)?;
        let buf_reader = BufReader::new(reader);
        Self::try_read_set_pace(buf_reader)
    }
}

pub struct SetReader<R> {
    lines: Lines<R>,
    number_of_nodes: NumNodes,
    number_of_edges: NumEdges,
    current_set_idx: NumEdges,
    current_edges: Vec<Edge>,
}

#[allow(dead_code)]
impl<R: BufRead> SetReader<R> {
    pub fn try_new(reader: R) -> Result<Self> {
        let mut pace_reader = Self {
            lines: reader.lines(),
            number_of_nodes: 0,
            number_of_edges: 0,
            current_set_idx: 0,
            current_edges: Vec::new(),
        };

        (pace_reader.number_of_nodes, pace_reader.number_of_edges) = pace_reader.parse_header()?;
        pace_reader.current_set_idx = pace_reader.number_of_nodes;
        Ok(pace_reader)
    }

    pub fn number_of_edges(&self) -> NumEdges {
        self.number_of_edges
    }

    pub fn number_of_nodes(&self) -> NumNodes {
        self.number_of_nodes
    }
}

impl<R: BufRead> Iterator for SetReader<R> {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        self.current_edges.pop().or_else(|| {
            self.current_edges = self.parse_edge_line().unwrap()?;
            self.current_set_idx += 1;
            self.current_edges.pop()
        })
    }
}

macro_rules! raise_error_unless {
    ($cond : expr, $kind : expr, $info : expr) => {
        if !($cond) {
            return Err(std::io::Error::new($kind, $info));
        }
    };
}

macro_rules! parse_next_value {
    ($iterator : expr, $name : expr) => {{
        let next = $iterator.next();
        raise_error_unless!(
            next.is_some(),
            ErrorKind::InvalidData,
            format!("Premature end of line when parsing {}.", $name)
        );

        let parsed = next.unwrap().parse();
        raise_error_unless!(
            parsed.is_ok(),
            ErrorKind::InvalidData,
            format!("Invalid value found. Cannot parse {}.", $name)
        );

        parsed.unwrap()
    }};
}

impl<R: BufRead> SetReader<R> {
    fn next_non_comment_line(&mut self) -> Result<Option<String>> {
        loop {
            let line = self.lines.next();
            match line {
                None => return Ok(None),
                Some(Err(x)) => return Err(x),
                Some(Ok(line)) if line.starts_with('c') => continue,
                Some(Ok(line)) => return Ok(Some(line)),
            }
        }
    }

    fn parse_header(&mut self) -> Result<(NumNodes, NumEdges)> {
        let line = self.next_non_comment_line()?;

        raise_error_unless!(line.is_some(), ErrorKind::InvalidData, "No header found");
        let line = line.unwrap();

        let mut parts = line.split(' ').filter(|t| !t.is_empty());

        raise_error_unless!(
            parts.next().is_some_and(|t| t.starts_with('p')),
            ErrorKind::InvalidData,
            "Invalid header found; line should start with p"
        );

        raise_error_unless!(
            parts.next() == Some("hs"),
            ErrorKind::InvalidData,
            "Invalid header found; file type should be \"hs\""
        );

        let number_of_nodes = parse_next_value!(parts, "Header>Number of nodes");
        let number_of_edges = parse_next_value!(parts, "Header>Number of edges");

        raise_error_unless!(
            parts.next().is_none(),
            ErrorKind::InvalidData,
            "Invalid header found; expected end of line"
        );

        Ok((number_of_nodes, number_of_edges))
    }

    fn parse_edge_line(&mut self) -> Result<Option<Vec<Edge>>> {
        let line = self.next_non_comment_line()?;
        if let Some(line) = line {
            let edges = line
                .trim()
                .split(' ')
                .filter(|t| !t.is_empty())
                .map(|s| {
                    let node = s.parse().unwrap();
                    debug_assert!((1..=self.number_of_nodes).contains(&node));

                    Edge(node - 1, self.current_set_idx)
                })
                .collect();

            Ok(Some(edges))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::*;

    #[test]
    fn test_success() {
        const DEMO_FILE: &str = "c TEST\np hs 6 5\n1 2\nc TEST\n2 3 4\n5 6\n1 3 6\n2 4 5\n";
        let buf_reader = std::io::BufReader::new(DEMO_FILE.as_bytes());
        let pace_reader = SetReader::try_new(buf_reader).unwrap();

        assert_eq!(pace_reader.number_of_nodes(), 6);
        assert_eq!(pace_reader.number_of_edges(), 5);

        let mut edges: Vec<_> = pace_reader.collect();
        edges.sort_unstable();

        // Nodes => Sets
        // 6  => {0, 1}
        // 7  => {1, 2, 3}
        // 8  => {4, 5}
        // 9  => {0, 2, 5}
        // 10 => {1, 3, 4}
        assert_eq!(
            edges,
            vec![
                Edge(0, 6),
                Edge(0, 9),
                Edge(1, 6),
                Edge(1, 7),
                Edge(1, 10),
                Edge(2, 7),
                Edge(2, 9),
                Edge(3, 7),
                Edge(3, 10),
                Edge(4, 8),
                Edge(4, 10),
                Edge(5, 8),
                Edge(5, 9),
            ]
        );
    }
}
