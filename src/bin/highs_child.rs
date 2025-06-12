use dss::exact::highs_sub::*;
use std::{
    io::Read,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::sleep,
    time::{Duration, Instant},
};

fn main() -> anyhow::Result<()> {
    let mut byte_buffer = Vec::new();
    std::io::stdin().lock().read_to_end(&mut byte_buffer)?;
    let problem: HighsSubprocessProblem = serde_json::from_slice(byte_buffer.as_slice())?;

    let processing = Arc::new(AtomicBool::from(true));

    let handler = {
        let processing_clone = processing.clone();
        std::thread::spawn(move || {
            let start = Instant::now();
            while processing_clone.load(Ordering::Acquire) {
                if start.elapsed().as_secs() > problem.timeout + 3 {
                    panic!("Timeout");
                }
                sleep(Duration::from_millis(500));
            }
        })
    };

    let resp = problem.solve();
    processing.store(false, Ordering::Release);
    serde_json::to_writer(std::io::stdout(), &resp)?;
    let _ = handler.join();

    Ok(())
}
