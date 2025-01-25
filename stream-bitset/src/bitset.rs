use std::{borrow::Borrow, marker::PhantomData, ops::*};

use super::{prelude::*, *};

pub type BitSet32 = BitSetImpl<u32>;
pub type BitSet64 = BitSetImpl<u64>;

#[derive(Default, PartialEq, Clone)]
pub struct BitSetImpl<Index>
where
    Index: PrimIndex,
{
    pub(crate) cardinality: usize,
    pub(crate) number_of_bits: usize,
    pub(crate) data: Vec<Bitmask>,
    pub(crate) _index: PhantomData<Index>,
}

impl<Index: PrimIndex> BitSetImpl<Index> {
    /// Creates a new BitSet with a universe size of `number_of_bits` bits.
    /// ```
    /// use stream_bitset::prelude::*;
    /// let set = BitSet32::new(12);
    /// assert_eq!(set.cardinality(), 0);
    /// assert_eq!(set.number_of_bits(), 12);
    /// ```
    pub fn new(number_of_bits: Index) -> Self {
        let number_of_bits = number_of_bits.to_usize().unwrap();
        Self {
            number_of_bits,
            cardinality: 0,
            data: vec![0; (BITS_IN_MASK - 1 + number_of_bits) / BITS_IN_MASK],
            _index: Default::default(),
        }
    }

    /// Creates a new BitSet with a universe size of `number_of_bits` bits
    /// where the data values come from an callable
    /// ```
    /// use stream_bitset::prelude::*;
    /// let set = BitSet32::new_from_bitmasks(128, || 1);
    /// assert_eq!(set.cardinality(), 2);
    /// assert_eq!(set.number_of_bits(), 128);
    /// ```
    pub fn new_from_bitmasks<F: FnMut() -> Bitmask>(number_of_bits: Index, mut source: F) -> Self {
        let number_of_bits = number_of_bits.to_usize().unwrap();
        let mut res = Self {
            number_of_bits,
            cardinality: 0,
            data: (0..(BITS_IN_MASK - 1 + number_of_bits) / BITS_IN_MASK)
                .map(|_| source())
                .collect(),
            _index: Default::default(),
        };

        res.mask_last_element();
        res.recompute_cardinality();
        res
    }

    /// Creates a new BitSet with a universe size of `number_of_bits` bits
    /// that are all set.
    /// ```
    /// use stream_bitset::prelude::*;
    /// let set = BitSet32::new_all_set(12);
    /// assert_eq!(set.cardinality(), 12);
    /// assert_eq!(set.number_of_bits(), 12);
    /// ```
    pub fn new_all_set(number_of_bits: Index) -> Self {
        let number_of_bits = number_of_bits.to_usize().unwrap();
        let mut res = Self {
            number_of_bits,
            cardinality: number_of_bits,
            data: vec![!0; (BITS_IN_MASK - 1 + number_of_bits) / BITS_IN_MASK],
            _index: Default::default(),
        };
        res.mask_last_element();
        res
    }

    /// Creates a new BitSet with a universe size of `number_of_bits` bits and
    /// sets the bits provided via the iterator `set_bits`.
    ///
    /// # Panics
    /// Panics if any of the bits in `set_bits` is larger than `number_of_bits`.
    ///
    /// # Example
    /// ```
    /// use stream_bitset::prelude::*;
    /// let set = BitSet32::new_with_bits_set(12, vec![1u32, 3, 5]);
    /// assert_eq!(set.cardinality(), 3);
    /// assert!(set.get_bit(1));
    /// assert!(!set.get_bit(2));
    /// assert!(set.get_bit(3));
    /// ```
    pub fn new_with_bits_set<I, B, T>(number_of_bits: Index, set_bits: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: Borrow<T>,
        T: Unsigned + ToPrimitive + Copy,
    {
        let mut set = Self::new(number_of_bits);
        set.set_bits(set_bits);
        set
    }

    /// Creates a new BitSet with a universe size of `number_of_bits` bits, where
    /// all bits but the ones provided via the iterator `clear_bits` are set.
    ///
    /// # Panics
    /// Panics if any of the bits in `clear_bits` is larger than `number_of_bits`.
    ///
    /// # Example
    /// ```
    /// use stream_bitset::prelude::*;
    /// let set = BitSet32::new_with_bits_cleared(12, vec![1u32, 3, 5]);
    /// assert_eq!(set.cardinality(), 9);
    /// assert!(!set.get_bit(1));
    /// assert!(set.get_bit(2));
    /// assert!(!set.get_bit(3));
    /// ```
    pub fn new_with_bits_cleared<I, B, T>(number_of_bits: Index, clear_bits: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: Borrow<T>,
        T: Unsigned + ToPrimitive + Copy,
    {
        let mut set = Self::new(number_of_bits);
        set.set_all();
        set.clear_bits(clear_bits);
        set
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
        self.data.as_slice()
    }

    #[inline(always)]
    fn _as_mut_slice(&mut self) -> &mut [Bitmask] {
        self.data.as_mut_slice()
    }

    bitset_macros::impl_bitset!();
}

impl<I> ToBitmaskStream for BitSetImpl<I>
where
    I: PrimIndex,
{
    type ToStream<'a> = BitmaskSliceStream<'a>
    where
        Self: 'a;

    fn bitmask_stream(&self) -> Self::ToStream<'_> {
        BitmaskSliceStream::new(self.data.as_slice(), self.number_of_bits)
    }
}

impl<I> IntoBitmaskStream for BitSetImpl<I>
where
    I: PrimIndex,
{
    type IntoStream = BitsetStream<I>;

    fn into_bitmask_stream(self) -> Self::IntoStream {
        BitsetStream { set: self, idx: 0 }
    }
}

pub trait IntoBitset {
    fn into_bitset<I: PrimIndex>(self) -> BitSetImpl<I>;
}

/// Converts a `BitmaskStream` into a `BitSet`.
///
/// # Example
/// ```
/// use stream_bitset::prelude::*;
/// let org = BitSet32::new_with_bits_set(40, [0u32, 5, 9, 12]);
/// let set: BitSet32 = org.bitmask_stream().into_bitset();
/// assert_eq!(set, org);
/// ```
impl<T: BitmaskStream> IntoBitset for T {
    fn into_bitset<I: PrimIndex>(self) -> BitSetImpl<I> {
        let mut set = BitSetImpl {
            number_of_bits: self.number_of_bits(),
            data: self.map(|StreamElement(x)| x).collect(),
            cardinality: 0,
            _index: Default::default(),
        };
        set.recompute_cardinality();
        set
    }
}

impl<I: PrimIndex> std::fmt::Debug for BitSetImpl<I> {
    bitset_macros::impl_bitset_fmt_debug!();
}

macro_rules! impl_assign_ops {
    ($trait:ident, $op_assign:ident, $op:ident) => {
        // Take Bitstream by value
        impl<I, B> $trait<&B> for BitSetImpl<I>
        where
            I: PrimIndex,
            B: ToBitmaskStream,
        {
            fn $op_assign(&mut self, rhs: &B) {
                let rhs = rhs.bitmask_stream();
                assert_eq!(self.number_of_bits, rhs.number_of_bits());
                for (l, StreamElement(r)) in self.data.iter_mut().zip(rhs) {
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

pub struct BitsetStream<Index>
where
    Index: PrimIndex,
{
    set: BitSetImpl<Index>,
    idx: usize,
}

impl<Index: PrimIndex> Iterator for BitsetStream<Index> {
    type Item = StreamElement;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.set.data.len() {
            return None;
        }
        let word = self.set.data[self.idx];
        self.idx += 1;
        Some(StreamElement(word))
    }
}

impl<Index: PrimIndex> BitmaskStream for BitsetStream<Index> {
    fn number_of_bits(&self) -> usize {
        self.set.number_of_bits
    }
}

impl<Index: PrimIndex> IntoBitmaskStream for BitsetStream<Index> {
    type IntoStream = Self;

    fn into_bitmask_stream(self) -> Self::IntoStream {
        self
    }
}

impl<Index: PrimIndex> ToBitmaskStream for BitsetStream<Index> {
    type ToStream<'b> = BitmaskSliceStream<'b>
    where
        Self: 'b;

    fn bitmask_stream(&self) -> Self::ToStream<'_> {
        BitmaskSliceStream::new(self.set.data.as_slice(), self.set.number_of_bits)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bitor_assign() {
        let mut set1 = BitSet32::new(32);
        let mut set2 = BitSet64::new(32);
        set1.set_bits(0u32..10);
        set2.set_bits(5u32..15);
        set1 |= &set2;
        assert_eq!(set1.cardinality(), 15);
    }

    #[test]
    fn bitand_assign() {
        let mut set1 = BitSet32::new(32);
        let mut set2 = BitSet64::new(32);
        set1.set_bits(0u32..10);
        set2.set_bits(5u32..15);
        set1 &= &set2;
        assert_eq!(set1.cardinality(), 5);
    }

    #[test]
    fn bitxor_assign() {
        let mut set1 = BitSet32::new(32);
        let mut set2 = BitSet64::new(32);
        set1.set_bits(0u32..10);
        set2.set_bits(5u32..15);
        set1 ^= &set2;
        assert_eq!(set1.cardinality(), 10);
    }

    #[test]
    fn bitsub_assign() {
        let mut set1 = BitSet32::new(32);
        let mut set2 = BitSet64::new(32);
        set1.set_bits(0u32..10);
        set2.set_bits(5u32..15);
        set1 -= &set2;
        assert_eq!(set1.cardinality(), 5);
        assert!(set1.get_bit(4));
    }

    #[test]
    fn debug() {
        let bitset = BitSet32::new_with_bits_set(8, [0u32, 1, 7]);
        let result = format!("{bitset:?}");
        assert_eq!(result, "BitSet(num_bits=8 card=3 [11000001])");
    }
}
