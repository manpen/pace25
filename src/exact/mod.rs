pub mod ext_maxsat;
pub mod naive;

pub mod highs;
use std::path::{Path, PathBuf};

pub use highs::highs_solver as default_exact_solver;

pub mod highs_advanced;
pub mod highs_sub;

pub const DEFAULT_SOLVER_IS_FAST: bool = true;

use thiserror::Error;

pub use crate::utils::DominatingSet;

#[derive(Debug, Error)]
pub enum ExactError {
    #[error("upper bound infeasible")]
    Infeasible,
    #[error("timeout")]
    Timeout,
    #[error("timeout_with_solution")]
    TimeoutWithSolution(DominatingSet),
}

pub type Result<T> = std::result::Result<T, ExactError>;

pub fn search_binary_path(bin: &Path) -> anyhow::Result<PathBuf> {
    // search in the same directory where this binary lies
    if let Ok(path) = std::env::current_exe() {
        let cand = path.with_file_name(bin);
        if cand.is_file() {
            return Ok(cand);
        }
    }

    // search in the current working dir
    if let Ok(path) = std::env::current_dir() {
        let cand = path.join(bin);
        if cand.is_file() {
            return Ok(cand);
        }
    }

    anyhow::bail!("Binary {bin:?} not found");
}
