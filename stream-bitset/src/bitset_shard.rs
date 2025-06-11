use std::{
    borrow::Borrow,
    marker::PhantomData,
    ops::*,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use super::{prelude::*, *};

pub type BitSetShard32 = BitSetShardImpl<u32>;
pub type BitSetShard64 = BitSetShardImpl<u64>;

#[derive(PartialEq, Clone)]
pub struct BitSetShardImpl<Index>
where
    Index: PrimIndex,
{
    cardinality: usize,
    number_of_bits: usize,
    data: *mut Bitmask,
    _index: PhantomData<Index>,
}

unsafe impl<Index: PrimIndex> Send for BitSetShardImpl<Index> {}
unsafe impl<Index: PrimIndex> Sync for BitSetShardImpl<Index> {}

impl<Index: PrimIndex> BitSetShardImpl<Index> {
    pub(crate) fn new(data: *mut Bitmask, number_of_bits: Index) -> Self {
        let number_of_bits = number_of_bits.to_usize().unwrap();
        let res = Self {
            number_of_bits,
            cardinality: 0,
            data,
            _index: Default::default(),
        };
        debug_assert_eq!(res.cardinality, res._get_cardinality());
        res
    }

    pub(crate) fn switch_to_new_pointer(&mut self, data: *mut Bitmask) {
        self.data = data;
        debug_assert_eq!(self.cardinality, self._get_cardinality());
    }

    #[inline(always)]
    fn _get_number_of_bits(&self) -> usize {
        self.number_of_bits
    }

    #[inline(always)]
    fn _set_number_of_bits(&mut self, number_of_bits: usize) {
        self.number_of_bits = number_of_bits;
    }

    #[inline(always)]
    fn _get_cardinality(&self) -> usize {
        self.cardinality
    }

    #[inline(always)]
    fn _set_cardinality(&mut self, cardinality: usize) {
        self.cardinality = cardinality;
    }

    #[inline(always)]
    fn _as_slice(&self) -> &[Bitmask] {
        let n = self._len();
        unsafe { from_raw_parts(self.data, n) }
    }

    #[inline(always)]
    fn _as_mut_slice(&mut self) -> &mut [Bitmask] {
        let n = self._len();
        unsafe { from_raw_parts_mut(self.data, n) }
    }

    #[inline(always)]
    pub(crate) fn _len(&self) -> usize {
        (self.number_of_bits + BITS_IN_MASK - 1) / BITS_IN_MASK
    }

    bitset_macros::impl_bitset!();
}

impl<I> ToBitmaskStream for BitSetShardImpl<I>
where
    I: PrimIndex,
{
    type ToStream<'a>
        = BitmaskSliceStream<'a>
    where
        Self: 'a;

    fn bitmask_stream(&self) -> Self::ToStream<'_> {
        BitmaskSliceStream::new(self._as_slice(), self.number_of_bits)
    }
}

impl<I> IntoBitmaskStream for BitSetShardImpl<I>
where
    I: PrimIndex,
{
    type IntoStream = BitsetStreamShard<I>;

    fn into_bitmask_stream(self) -> Self::IntoStream {
        BitsetStreamShard { set: self, idx: 0 }
    }
}

impl<I: PrimIndex> std::fmt::Debug for BitSetShardImpl<I> {
    bitset_macros::impl_bitset_fmt_debug!();
}

impl<I: PrimIndex> IntoBitset for &BitSetShardImpl<I> {
    fn into_bitset<J: PrimIndex>(self) -> BitSetImpl<J> {
        BitSetImpl {
            cardinality: self.cardinality,
            number_of_bits: self.number_of_bits,
            data: Vec::from(self._as_slice()),
            _index: Default::default(),
        }
    }
}

macro_rules! impl_assign_ops {
    ($trait:ident, $op_assign:ident, $op:ident) => {
        // Take Bitstream by value
        impl<I, B> $trait<&B> for BitSetShardImpl<I>
        where
            I: PrimIndex,
            B: ToBitmaskStream,
        {
            fn $op_assign(&mut self, rhs: &B) {
                let rhs = rhs.bitmask_stream();
                assert_eq!(self.number_of_bits, rhs.number_of_bits());
                for (l, StreamElement(r)) in self._as_mut_slice().iter_mut().zip(rhs) {
                    *l = <$op>::op(*l, r);
                }
                self.recompute_cardinality();
            }
        }
    };
}

impl_assign_ops!(BitOrAssign, bitor_assign, BitmaskOr);
impl_assign_ops!(BitAndAssign, bitand_assign, BitmaskAnd);
impl_assign_ops!(BitXorAssign, bitxor_assign, BitmaskXor);
impl_assign_ops!(SubAssign, sub_assign, BitmaskSub);

pub struct BitsetStreamShard<Index>
where
    Index: PrimIndex,
{
    set: BitSetShardImpl<Index>,
    idx: usize,
}

impl<Index: PrimIndex> Iterator for BitsetStreamShard<Index> {
    type Item = StreamElement;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = self.set._as_slice();
        if self.idx >= slice.len() {
            return None;
        }
        let word = slice[self.idx];
        self.idx += 1;
        Some(StreamElement(word))
    }
}

impl<Index: PrimIndex> BitmaskStream for BitsetStreamShard<Index> {
    fn number_of_bits(&self) -> usize {
        self.set.number_of_bits
    }
}

impl<Index: PrimIndex> IntoBitmaskStream for BitsetStreamShard<Index> {
    type IntoStream = Self;

    fn into_bitmask_stream(self) -> Self::IntoStream {
        self
    }
}

impl<Index: PrimIndex> ToBitmaskStream for BitsetStreamShard<Index> {
    type ToStream<'b>
        = BitmaskSliceStream<'b>
    where
        Self: 'b;

    fn bitmask_stream(&self) -> Self::ToStream<'_> {
        BitmaskSliceStream::new(self.set._as_slice(), self.set.number_of_bits)
    }
}
