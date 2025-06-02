pub mod naive;
pub mod sat_solver;

#[cfg(feature = "highs")]
pub mod highs;

#[cfg(feature = "highs")]
pub use highs::highs_solver as default_exact_solver;

#[cfg(not(feature = "highs"))]
pub use naive::naive_solver as default_exact_solver;

#[cfg(not(feature = "highs"))]
pub const DEFAULT_SOLVER_IS_FAST: bool = false;

#[cfg(feature = "highs")]
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
