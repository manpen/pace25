use dss::exact::highs_sub::*;
use std::io::Read;

fn main() -> anyhow::Result<()> {
    let mut byte_buffer = Vec::new();
    std::io::stdin().lock().read_to_end(&mut byte_buffer)?;
    let problem: HighsSubprocessProblem = serde_json::from_slice(byte_buffer.as_slice())?;
    let resp = problem.solve();
    serde_json::to_writer(std::io::stdout(), &resp)?;
    Ok(())
}
