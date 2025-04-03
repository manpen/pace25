use std::error::Error;

/// Trait for checking invariants in datastructures
pub trait InvariantCheck<E: Error> {
    fn is_correct(&self) -> Result<(), E>;
}
