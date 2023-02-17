use super::{graph::*, io::*};
use regex::Regex;

pub fn get_test_graphs_with_tww(pattern: &str) -> impl Iterator<Item = (String, AdjArray, Node)> {
    let files = glob::glob(pattern).expect("Invalid pattern");
    let regex_pattern = Regex::new(r"_tww(\d+)_").expect("Invalid regexp");

    files.filter_map(move |filename| {
        let filename = filename.ok()?;
        let filename_string = String::from(filename.as_os_str().to_str()?);

        let tww = regex_pattern
            .captures_iter(filename_string.as_str())
            .next()?[1]
            .parse()
            .ok()?;

        let graph = AdjArray::try_read_pace_file(filename).ok()?;

        Some((filename_string, graph, tww))
    })
}
