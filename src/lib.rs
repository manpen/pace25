#![feature(type_alias_impl_trait)]
#![allow(incomplete_features)]
#![feature(let_chains)]
#![feature(linked_list_cursors)]
#![feature(impl_trait_in_assoc_type)]

pub mod algorithm;
pub mod exact;
pub mod graph;
pub mod heuristic;
pub mod io;
pub mod kernelization;
pub mod log;
pub mod utils;

pub mod prelude {
    pub use super::algorithm::*;
    pub use super::graph::*;
    pub use super::io::*;
    pub use super::utils::*;
}

#[cfg(test)]
mod testing;
