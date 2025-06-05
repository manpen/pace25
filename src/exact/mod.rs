pub mod naive;
pub mod sat_solver;

pub mod highs;
pub use highs::highs_solver as default_exact_solver;

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
