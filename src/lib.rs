#![feature(type_alias_impl_trait)]
#![allow(incomplete_features)]
#![feature(let_chains)]
#![feature(linked_list_cursors)]
#![feature(impl_trait_in_assoc_type)]
#![feature(vec_push_within_capacity)]
#![feature(iter_collect_into)]

pub mod algorithm;
pub mod errors;
pub mod exact;
pub mod graph;
pub mod heuristic;
pub mod io;
pub mod log;
pub mod reduction;
pub mod utils;

pub mod prelude {
    pub use super::algorithm::*;
    pub use super::graph::*;
    pub use super::io::*;
    pub use super::utils::*;
    pub use log::{debug, info};
}

#[cfg(test)]
mod testing;
