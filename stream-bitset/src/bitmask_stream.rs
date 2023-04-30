use super::*;

use std::{
    marker::PhantomData,
    ops::{BitAnd, BitOr, BitXor, Sub},
};

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct StreamElement(pub Bitmask);

impl std::fmt::Debug for StreamElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:064b}", self.0)
    }
}

pub trait BitmaskStream:
    Iterator<Item = StreamElement> + ToBitmaskStream + IntoBitmaskStream
{
    fn number_of_bits(&self) -> usize;
}

pub trait IntoBitmaskStream {
    type IntoStream: BitmaskStream;
    fn into_bitmask_stream(self) -> Self::IntoStream;
}

pub trait ToBitmaskStream {
    type ToStream<'a>: BitmaskStream
    where
        Self: 'a;
    fn bitmask_stream(&self) -> Self::ToStream<'_>;
}

#[derive(Clone)]
pub struct BitmaskSliceStream<'a> {
    number_of_bits: usize,
    data: &'a [Bitmask],
}

impl<'a> BitmaskSliceStream<'a> {
    pub fn new(data: &'a [Bitmask], number_of_bits: usize) -> Self {
        assert_eq!(
            data.len(),
            ((number_of_bits + BITS_IN_MASK - 1) / BITS_IN_MASK)
        );

        let remainder = number_of_bits % BITS_IN_MASK;
        assert!(remainder == 0 || (*data.last().unwrap() >> remainder) == 0);

        Self {
            data,
            number_of_bits,
        }
    }
}

impl<'a> Iterator for BitmaskSliceStream<'a> {
    type Item = StreamElement;

    fn next(&mut self) -> Option<Self::Item> {
        let (&first, data) = self.data.split_first()?;
        self.data = data;
        Some(StreamElement(first))
    }
}

impl<'a> BitmaskStream for BitmaskSliceStream<'a> {
    fn number_of_bits(&self) -> usize {
        self.number_of_bits
    }
}

impl<'a> IntoBitmaskStream for BitmaskSliceStream<'a> {
    type IntoStream = Self;

    fn into_bitmask_stream(self) -> Self::IntoStream {
        self
    }
}

impl<'a> ToBitmaskStream for BitmaskSliceStream<'a> {
    type ToStream<'b> = Self
    where
        Self: 'b;

    fn bitmask_stream(&self) -> Self::ToStream<'_> {
        self.clone()
    }
}

pub struct BitmaskStreamBinop<L, R, Op> {
    left: L,
    right: R,
    number_of_bits: usize,
    bits_remaining: usize,
    _op: PhantomData<Op>,
}

impl<L, R, Op> Iterator for BitmaskStreamBinop<L, R, Op>
where
    L: BitmaskStream,
    R: BitmaskStream,
    Op: BitmaskBinop,
{
    type Item = StreamElement;

    fn next(&mut self) -> Option<Self::Item> {
        let l = self.left.next()?.0;
        let r = self.right.next()?.0;

        let mask = if self.bits_remaining < BITS_IN_MASK {
            assert!(self.bits_remaining > 0);
            (1 << self.bits_remaining) - 1
        } else {
            Bitmask::MAX
        };

        self.bits_remaining = self.bits_remaining.saturating_sub(BITS_IN_MASK);

        Some(StreamElement(Op::op(l, r) & mask))
    }
}

impl<L, R, Op> BitmaskStream for BitmaskStreamBinop<L, R, Op>
where
    L: BitmaskStream,
    R: BitmaskStream,
    Op: BitmaskBinop,
{
    fn number_of_bits(&self) -> usize {
        self.number_of_bits
    }
}

impl<L, R, Op> IntoBitmaskStream for BitmaskStreamBinop<L, R, Op>
where
    L: BitmaskStream,
    R: BitmaskStream,
    Op: BitmaskBinop,
{
    type IntoStream = Self;

    fn into_bitmask_stream(self) -> Self::IntoStream {
        self
    }
}

impl<L, R, Op> ToBitmaskStream for BitmaskStreamBinop<L, R, Op>
where
    L: BitmaskStream,
    R: BitmaskStream,
    Op: BitmaskBinop,
{
    type ToStream<'a> = BitmaskStreamBinop<L::ToStream<'a>, R::ToStream<'a>, Op>
    where
        Self: 'a;

    fn bitmask_stream(&self) -> Self::ToStream<'_> {
        BitmaskStreamBinop::<_, _, Op> {
            left: self.left.bitmask_stream(),
            right: self.right.bitmask_stream(),
            number_of_bits: self.number_of_bits,
            bits_remaining: self.bits_remaining,
            _op: Default::default(),
        }
    }
}

macro_rules! impl_bitmask_binop {
    ($name : ident, $op_trait : ident, $op_func : ident) => {
        impl<'a> $op_trait for BitmaskSliceStream<'a> {
            type Output = BitmaskStreamBinop<Self, Self, $name>;

            fn $op_func(self, rhs: Self) -> Self::Output {
                let number_of_bits = self.number_of_bits().min(rhs.number_of_bits());

                Self::Output {
                    left: self,
                    right: rhs,
                    number_of_bits,
                    bits_remaining: number_of_bits,
                    _op: Default::default(),
                }
            }
        }

        impl<'a, T> $op_trait<&'a T> for BitmaskSliceStream<'a>
        where
            T: ToBitmaskStream,
        {
            type Output = BitmaskStreamBinop<Self, T::ToStream<'a>, $name>;

            fn $op_func(self, rhs: &'a T) -> Self::Output {
                let rhs = rhs.bitmask_stream();
                let number_of_bits = self.number_of_bits().min(rhs.number_of_bits());

                Self::Output {
                    left: self,
                    right: rhs,
                    number_of_bits,
                    bits_remaining: number_of_bits,
                    _op: Default::default(),
                }
            }
        }
    };
}

impl_bitmask_binop!(BitmaskOr, BitOr, bitor);
impl_bitmask_binop!(BitmaskAnd, BitAnd, bitand);
impl_bitmask_binop!(BitmaskXor, BitXor, bitxor);
impl_bitmask_binop!(BitmaskSub, Sub, sub);

pub struct BitmaskStreamNeg<I> {
    inner: I,
    next: Option<Bitmask>,
}

impl<I: BitmaskStream> From<I> for BitmaskStreamNeg<I> {
    fn from(mut value: I) -> Self {
        let next = value.next().map(|x| x.0);
        Self { inner: value, next }
    }
}

impl<I: BitmaskStream> Iterator for BitmaskStreamNeg<I> {
    type Item = StreamElement;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(value) = self.next {
            self.next = self.inner.next().map(|x| x.0);

            let mut value = !value;
            if self.next.is_none() {
                // value is not the last element; mask out unused bits
                let remainder = self.inner.number_of_bits() % BITS_IN_MASK;
                value &= !(Bitmask::MAX << remainder);
            }

            Some(StreamElement(value))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn bitmask_slice_stream() {
        let data = [0x123, 0x456];
        let mut stream = BitmaskSliceStream::new(&data, 128);

        assert_eq!(stream.next().unwrap(), StreamElement(0x123));
        assert_eq!(stream.next().unwrap(), StreamElement(0x456));
        assert!(stream.next().is_none());
    }

    #[test]
    #[should_panic]
    fn bitmask_slice_stream_panic() {
        let data = [0b0, 0b11]; // one bit too many
        BitmaskSliceStream::new(&data, BITS_IN_MASK + 1);
    }

    #[test]
    #[should_panic]
    fn bitmask_slice_stream_panic1() {
        let data = [0b100]; // one bit too many
        BitmaskSliceStream::new(&data, 2);
    }

    #[test]
    fn bitset_explicit_stream() {
        let bs0 = BitSet32::new_with_bits_set(96, (0u32..96).filter(|x| *x % 2 == 0));
        let bs1 = BitSet32::new_with_bits_set(96, (0u32..96).filter(|x| *x % 3 == 0));
        let bs2 = BitSet32::new_with_bits_set(96, (0u32..96).filter(|x| *x % 6 == 0));

        let bi0 = bs0.bitmask_stream();
        assert_eq!(bi0.number_of_bits(), 96);

        let bi1 = bs1.bitmask_stream();

        for (and, rf) in (bi0 & bi1).zip(bs2.bitmask_stream()) {
            assert_eq!(and, rf);
        }
    }

    #[test]
    fn bitset_implicit_stream() {
        let bs0 = BitSet32::new_with_bits_set(96, (0u32..96).filter(|x| *x % 2 == 0));
        let bs1 = BitSet32::new_with_bits_set(96, (0u32..96).filter(|x| *x % 3 == 0));
        let bs2 = BitSet32::new_with_bits_set(96, (0u32..96).filter(|x| *x % 6 == 0));

        for (and, rf) in (bs0.bitmask_stream() & &bs1).zip(bs2.bitmask_stream()) {
            assert_eq!(and, rf);
        }
    }
}
