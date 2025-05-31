pub mod naive;
pub mod sat_solver;

#[cfg(feature = "highs")]
pub mod highs;

#[cfg(feature = "highs")]
pub use highs::highs_solver as default_exact_solver;

#[cfg(not(feature = "highs"))]
pub use naive::naive_solver as default_exact_solver;
