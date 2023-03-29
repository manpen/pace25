use super::{graph::*, io::*};
use rayon::prelude::*;
use regex::Regex;

#[allow(dead_code)]
pub fn get_test_graphs_with_tww(pattern: &str) -> impl Iterator<Item = (String, AdjArray, Node)> {
    let files = glob::glob(pattern).expect("Invalid pattern");
    let regex_pattern = Regex::new(r"_tww(\d+)_").expect("Invalid regexp");

    files.filter_map(move |filename| process_file(filename, &regex_pattern))
}

#[allow(dead_code)]
pub fn get_test_graphs_with_tww_in_par(
    pattern: &str,
) -> impl ParallelIterator<Item = (String, AdjArray, Node)> {
    let files: Vec<_> = glob::glob(pattern).expect("Invalid pattern").collect();
    let regex_pattern = Regex::new(r"_tww(\d+)_").expect("Invalid regexp");

    files
        .into_par_iter()
        .filter_map(move |filename| process_file(filename, &regex_pattern))
}

fn process_file(
    filename: std::result::Result<std::path::PathBuf, glob::GlobError>,
    regex_pattern: &Regex,
) -> Option<(String, AdjArray, u32)> {
    let filename = filename.ok()?;
    let filename_string = String::from(filename.as_os_str().to_str()?);

    let tww = regex_pattern
        .captures_iter(filename_string.as_str())
        .next()?[1]
        .parse()
        .ok()?;

    let graph = AdjArray::try_read_pace_file(filename).ok()?;

    Some((filename_string, graph, tww))
}
