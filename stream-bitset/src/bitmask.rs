use std::ops::{BitAnd, BitOr, BitXor};

pub type Bitmask = u64;
pub const BITS_IN_MASK: usize = 8 * std::mem::size_of::<Bitmask>();

pub trait BitmaskBinop {
    fn op(lhs: Bitmask, rhs: Bitmask) -> Bitmask;
}

pub struct BitmaskSub();

impl BitmaskBinop for BitmaskSub {
    fn op(lhs: Bitmask, rhs: Bitmask) -> Bitmask {
        lhs & !rhs
    }
}

macro_rules! impl_bitmask_binop {
    ($name : ident, $op_func : ident) => {
        pub struct $name();

        impl BitmaskBinop for $name {
            fn op(lhs: Bitmask, rhs: Bitmask) -> Bitmask {
                lhs.$op_func(rhs)
            }
        }
    };
}

impl_bitmask_binop!(BitmaskOr, bitor);
impl_bitmask_binop!(BitmaskAnd, bitand);
impl_bitmask_binop!(BitmaskXor, bitxor);

pub(crate) trait BitManip {
    fn is_ith_bit_set(&self, id: usize) -> bool;
    fn ith_bit_set(id: usize) -> Self;
    fn assign_ith_bit(&mut self, id: usize, value: bool) -> bool;

    fn set_ith_bit(&mut self, id: usize) -> bool {
        self.assign_ith_bit(id, true)
    }

    fn clear_ith_bit(&mut self, id: usize) -> bool {
        self.assign_ith_bit(id, false)
    }
}

impl BitManip for Bitmask {
    fn is_ith_bit_set(&self, id: usize) -> bool {
        (Self::ith_bit_set(id) & *self) != 0
    }

    fn ith_bit_set(id: usize) -> Self {
        1 << id
    }

    fn assign_ith_bit(&mut self, id: usize, value: bool) -> bool {
        let mask = Self::ith_bit_set(id);
        let prev = *self & mask;
        *self = (*self & !mask) | ((value as Bitmask) << id);
        prev != 0
    }
}
