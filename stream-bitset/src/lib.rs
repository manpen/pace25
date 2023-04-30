#![allow(incomplete_features)]
#![feature(inherent_associated_types)]

pub mod bitmask;
pub mod bitmask_stream;
pub mod bitmask_stream_consumer;
pub mod bitset;
pub mod bitset_array;
mod bitset_macros;
pub mod bitset_shard;

use bitmask::*;
use num::{FromPrimitive, PrimInt, ToPrimitive, Unsigned};
pub trait PrimIndex: PrimInt + ToPrimitive + FromPrimitive + Unsigned {}
impl<T: PrimInt + ToPrimitive + FromPrimitive + Unsigned> PrimIndex for T {}

pub mod prelude {
    pub use super::{bitmask_stream::*, bitmask_stream_consumer::*, bitset::*, bitset_shard::*};
}
