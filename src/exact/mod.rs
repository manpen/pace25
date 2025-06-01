pub mod naive;
pub mod sat_solver;

#[cfg(feature = "highs")]
pub mod highs;

#[cfg(feature = "highs")]
pub use highs::highs_solver as default_exact_solver;

#[cfg(not(feature = "highs"))]
pub use naive::naive_solver as default_exact_solver;
use thiserror::Error;

#[derive(Debug, PartialEq, PartialOrd, Error)]
pub enum ExactError {
    #[error("upper bound infeasible")]
    Infeasible,
    #[error("timeout")]
    Timeout,
}

pub type Result<T> = std::result::Result<T, ExactError>;
