use super::*;
use bitvec::prelude::*;
use num::{NumCast, ToPrimitive, Unsigned};
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{AddAssign, Div, Index};
use std::{fmt, iter};

#[derive(Default)]
pub struct BitSet {
    cardinality: NumNodes,
    bit_vec: BitVec,
}

impl Clone for BitSet {
    fn clone(&self) -> Self {
        // it's quite common to write vec![BitSet::new(n), n] which is quite expensive
        // if done by actually copying the BitSet. The following heuristic causes a massive
        // speed-up in these situations.
        if self.empty() {
            Self::new(self.len())
        } else {
            Self {
                cardinality: self.cardinality,
                bit_vec: self.bit_vec.clone(),
            }
        }
    }
}

impl Ord for BitSet {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bit_vec.cmp(&other.bit_vec)
    }
}

impl PartialOrd for BitSet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.bit_vec.partial_cmp(&other.bit_vec)
    }
}

impl Debug for BitSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let values: Vec<_> = self.iter().map(|i| i.to_string()).collect();
        write!(
            f,
            "BitSet {{ cardinality: {}, bit_vec: [{}]}}",
            self.cardinality,
            values.join(", "),
        )
    }
}

impl PartialEq for BitSet {
    fn eq(&self, other: &Self) -> bool {
        self.cardinality == other.cardinality && self.bit_vec == other.bit_vec
    }
}
impl Eq for BitSet {}

impl Hash for BitSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bit_vec.hash(state)
    }
}

#[inline]
fn subset_helper(a: &[usize], b: &[usize]) -> bool {
    if a.len() > b.len() {
        !a.iter()
            .zip(b.iter().chain(iter::repeat(&0)))
            .any(|(a, b)| (*a | *b) != *b)
    } else {
        !a.iter()
            .chain(iter::repeat(&0))
            .zip(b.iter())
            .any(|(a, b)| (*a | *b) != *b)
    }
}

const fn block_size() -> usize {
    std::mem::size_of::<usize>() * 8
}

impl BitSet {
    #[inline]
    pub fn new(size: NumNodes) -> Self {
        let mut bit_vec: BitVec = BitVec::with_capacity(size as usize);
        unsafe {
            bit_vec.set_len(size as usize);
        }
        for i in bit_vec.as_raw_mut_slice() {
            *i = 0;
        }
        Self {
            cardinality: 0,
            bit_vec,
        }
    }

    pub fn from_bitvec(bit_vec: BitVec) -> Self {
        let cardinality = bit_vec.iter().filter(|b| **b).count() as NumNodes;
        Self {
            cardinality,
            bit_vec,
        }
    }

    pub fn from_slice<T: Div<Output = T> + ToPrimitive + AddAssign + Default + Copy + Display>(
        size: NumNodes,
        slice: &[T],
    ) -> Self {
        let mut bit_vec: BitVec = BitVec::with_capacity(size as usize);
        unsafe {
            bit_vec.set_len(size as usize);
        }
        slice.iter().for_each(|i| {
            bit_vec.set(NumCast::from(*i).unwrap(), true);
        });
        let cardinality = slice.len() as NumNodes;
        Self {
            cardinality,
            bit_vec,
        }
    }

    #[inline]
    pub fn empty(&self) -> bool {
        self.cardinality == 0
    }

    #[inline]
    pub fn full(&self) -> bool {
        self.cardinality as usize == self.bit_vec.len()
    }

    pub fn new_all_set(size: NumNodes) -> Self {
        let mut bit_vec: BitVec = BitVec::with_capacity(size as usize);
        unsafe {
            bit_vec.set_len(size as usize);
        }
        for i in bit_vec.as_raw_mut_slice() {
            *i = usize::MAX;
        }
        Self {
            cardinality: size,
            bit_vec,
        }
    }

    pub fn new_all_set_but<T, I>(size: NumNodes, bits_unset: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Unsigned + ToPrimitive,
    {
        let mut bs = BitSet::new_all_set(size);
        for i in bits_unset {
            bs.unset_bit(i.to_usize().unwrap() as NumNodes);
        }
        bs
    }

    pub fn new_all_unset_but<T, I>(size: NumNodes, bits_set: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Unsigned + ToPrimitive,
    {
        let mut bs = BitSet::new(size);
        for i in bits_set {
            bs.set_bit(i.to_usize().unwrap() as NumNodes);
        }
        bs
    }

    #[inline]
    pub fn is_disjoint_with(&self, other: &BitSet) -> bool {
        !self
            .bit_vec
            .as_raw_slice()
            .iter()
            .zip(other.as_slice().iter())
            .any(|(x, y)| *x ^ *y != *x | *y)
    }

    #[inline]
    pub fn intersects_with(&self, other: &BitSet) -> bool {
        !self.is_disjoint_with(other)
    }

    #[inline]
    pub fn is_subset_of(&self, other: &BitSet) -> bool {
        self.cardinality <= other.cardinality
            && subset_helper(self.bit_vec.as_raw_slice(), other.as_slice())
    }

    #[inline]
    pub fn is_superset_of(&self, other: &BitSet) -> bool {
        other.is_subset_of(self)
    }

    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        self.bit_vec.as_raw_slice()
    }

    #[inline]
    pub fn as_bitslice(&self) -> &BitSlice {
        self.bit_vec.as_bitslice()
    }

    #[inline]
    pub fn as_bit_vec(&self) -> &BitVec {
        &self.bit_vec
    }

    #[inline]
    pub fn set_bit(&mut self, idx: NumNodes) -> bool {
        if !*self.bit_vec.get(idx as usize).unwrap() {
            self.bit_vec.set(idx as usize, true);
            self.cardinality += 1;
            false
        } else {
            true
        }
    }

    #[inline]
    pub fn flip_bit(&mut self, idx: NumNodes) -> bool {
        let prev = *self.bit_vec.get(idx as usize).unwrap();
        self.bit_vec.set(idx as usize, !prev);
        self.cardinality += 1 - 2 * (prev as NumNodes);
        prev
    }

    #[inline]
    pub fn unset_bit(&mut self, idx: NumNodes) -> bool {
        if *self.bit_vec.get(idx as usize).unwrap() {
            self.bit_vec.set(idx as usize, false);
            self.cardinality -= 1;
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn cardinality(&self) -> NumNodes {
        self.cardinality
    }

    #[inline]
    pub fn len(&self) -> NumNodes {
        self.bit_vec.len() as NumNodes
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bit_vec.is_empty()
    }

    #[inline]
    pub fn or(&mut self, other: &BitSet) {
        if other.len() as usize > self.bit_vec.len() {
            self.bit_vec.resize(other.len() as usize, false);
        }
        for (x, y) in self
            .bit_vec
            .as_raw_mut_slice()
            .iter_mut()
            .zip(other.as_slice().iter())
        {
            *x |= y;
        }
        self.cardinality = self.bit_vec.count_ones() as NumNodes;
    }

    #[inline]
    pub fn resize(&mut self, size: NumNodes) {
        let old_size = self.bit_vec.len();
        self.bit_vec.resize(size as usize, false);
        if (size as usize) < old_size {
            self.cardinality = self.bit_vec.count_ones() as NumNodes;
        }
    }

    #[inline]
    pub fn and(&mut self, other: &BitSet) {
        for (x, y) in self
            .bit_vec
            .as_raw_mut_slice()
            .iter_mut()
            .zip(other.as_slice().iter())
        {
            *x &= y;
        }
        self.cardinality = self.bit_vec.count_ones() as NumNodes;
    }

    #[inline]
    pub fn and_not(&mut self, other: &BitSet) {
        for (x, y) in self
            .bit_vec
            .as_raw_mut_slice()
            .iter_mut()
            .zip(other.as_slice().iter())
        {
            *x &= !y;
        }
        self.cardinality = self.bit_vec.count_ones() as NumNodes;
    }

    #[inline]
    pub fn not(&mut self) {
        self.bit_vec
            .as_raw_mut_slice()
            .iter_mut()
            .for_each(|x| *x = !*x);
        self.cardinality = self.bit_vec.count_ones() as NumNodes;
    }

    #[inline]
    pub fn unset_all(&mut self) {
        self.bit_vec
            .as_raw_mut_slice()
            .iter_mut()
            .for_each(|x| *x = 0);
        self.cardinality = 0;
    }

    #[inline]
    pub fn set_all(&mut self) {
        self.bit_vec
            .as_raw_mut_slice()
            .iter_mut()
            .for_each(|x| *x = std::usize::MAX);
        self.cardinality = self.bit_vec.len() as NumNodes;
    }

    #[inline]
    pub fn has_smaller(&mut self, other: &BitSet) -> Option<bool> {
        let self_idx = self.get_first_set()?;
        let other_idx = other.get_first_set()?;
        Some(self_idx < other_idx)
    }

    #[inline]
    pub fn get_first_set(&self) -> Option<NumNodes> {
        if self.cardinality != 0 {
            return self.get_next_set(0);
        }
        None
    }

    #[inline]
    pub fn get_next_set(&self, idx: NumNodes) -> Option<NumNodes> {
        let idx = idx as usize;
        if idx >= self.bit_vec.len() {
            return None;
        }
        let mut block_idx = idx / block_size();
        let word_idx = idx % block_size();
        let mut block = self.bit_vec.as_raw_slice()[block_idx];
        let max = self.bit_vec.as_raw_slice().len();
        block &= usize::MAX << word_idx;
        while block == 0usize {
            block_idx += 1;
            if block_idx >= max {
                return None;
            }
            block = self.bit_vec.as_raw_slice()[block_idx];
        }
        let v = block_idx * block_size() + block.trailing_zeros() as usize;
        if v >= self.bit_vec.len() {
            None
        } else {
            Some(v as NumNodes)
        }
    }

    #[inline]
    pub fn get_first_unset(&self) -> Option<NumNodes> {
        if self.cardinality != self.len() {
            return self.get_next_unset(0);
        }
        None
    }

    #[inline]
    pub fn get_next_unset(&self, idx: NumNodes) -> Option<NumNodes> {
        let idx = idx as usize;
        if idx >= self.bit_vec.len() {
            return None;
        }
        let mut block_idx = idx / block_size();
        let word_idx = idx % block_size();
        let mut block = self.bit_vec.as_raw_slice()[block_idx];
        let max = self.bit_vec.as_raw_slice().len();
        block |= (1 << word_idx) - 1;
        while block == usize::MAX {
            block_idx += 1;
            if block_idx >= max {
                return None;
            }
            block = self.bit_vec.as_raw_slice()[block_idx];
        }
        let v = block_idx * block_size() + block.trailing_ones() as usize;
        if v >= self.bit_vec.len() {
            None
        } else {
            Some(v as NumNodes)
        }
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<u32> {
        let mut tmp = Vec::with_capacity(self.cardinality as usize);
        for (i, _) in self
            .bit_vec
            .as_bitslice()
            .iter()
            .enumerate()
            .filter(|(_, x)| **x)
        {
            tmp.push(i as u32);
        }
        tmp
    }

    #[inline]
    pub fn at(&self, idx: NumNodes) -> bool {
        self.bit_vec[idx as usize]
    }

    #[inline]
    pub fn iter(&self) -> BitSetIterator {
        BitSetIterator {
            iter: self.bit_vec.as_raw_slice().iter(),
            block: 0,
            idx: 0,
            size: self.bit_vec.len(),
        }
    }
}

pub struct BitSetIterator<'a> {
    iter: ::std::slice::Iter<'a, usize>,
    block: usize,
    idx: usize,
    size: usize,
}

impl<'a> Iterator for BitSetIterator<'a> {
    type Item = NumNodes;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.block == 0 {
            self.block = if let Some(&i) = self.iter.next() {
                if i == 0 {
                    self.idx += block_size();
                    continue;
                } else {
                    self.idx = ((self.idx + block_size() - 1) / block_size()) * block_size();
                    i
                }
            } else {
                return None;
            }
        }
        let offset = self.block.trailing_zeros() as usize;
        self.block >>= offset;
        self.block >>= 1;
        self.idx += offset + 1;
        if self.idx > self.size {
            return None;
        }
        Some((self.idx - 1) as NumNodes)
    }
}

impl Index<NumNodes> for BitSet {
    type Output = bool;

    #[inline]
    fn index(&self, index: NumNodes) -> &Self::Output {
        self.bit_vec.index(index as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iter() {
        let mut bs = BitSet::new(256);

        let a: Vec<NumNodes> = (0..256).filter(|i| i % 2 == 0).collect();
        for i in &a {
            bs.set_bit(*i);
        }

        let b: Vec<NumNodes> = bs.iter().collect();
        assert_eq!(a, b);
        {
            let mut c = Vec::new();
            let mut v = bs.get_next_set(0);
            while v.is_some() {
                c.push(v.unwrap());
                v = bs.get_next_set(v.unwrap() + 1);
            }
            assert_eq!(a, c);
        }

        {
            let odds: Vec<NumNodes> = (0..256).filter(|i| i % 2 == 1).collect();
            let mut d = Vec::new();
            let mut v = bs.get_next_unset(0);
            while v.is_some() {
                d.push(v.unwrap());
                v = bs.get_next_unset(v.unwrap() + 1);
            }
            assert_eq!(odds, d);
        }
    }

    #[test]
    fn get_set() {
        let n = 257;
        let mut bs = BitSet::new(n);
        for i in 0..n {
            assert!(!bs[i]);
        }
        for i in 0..n {
            bs.set_bit(i);
            assert!(bs[i]);
        }

        for i in 0..n {
            bs.unset_bit(i);
            assert!(!bs[i]);
        }

        for i in 0..n {
            bs.flip_bit(i);
            assert!(bs[i]);
        }
    }

    #[test]
    fn logic() {
        let n = 257;
        let mut bs1 = BitSet::new_all_set(n);

        for i in 0..n {
            assert!(bs1[i]);
        }

        let mut bs2 = BitSet::new(n);

        for i in 0..n {
            assert!(!bs2[i]);
        }
        for i in (0..n).filter(|i| i % 2 == 0) {
            bs2.set_bit(i);
            bs1.unset_bit(i);
        }

        let mut tmp = bs1.clone();
        tmp.and(&bs2);
        for i in 0..n {
            assert!(!tmp[i]);
        }

        let mut tmp = bs1.clone();
        tmp.or(&bs2);
        for i in 0..n {
            assert!(tmp[i]);
        }

        let mut tmp = bs1.clone();
        tmp.and_not(&bs2);
        for i in (0..n).filter(|i| i % 2 == 0) {
            assert!(!tmp[i]);
        }
    }

    #[test]
    fn test_new_all_set_but() {
        // 0123456789
        //  ++ ++ ++
        let bs = BitSet::new_all_set_but(10, (0 as NumNodes..10).filter(|x| x % 3 == 0));
        assert_eq!(bs.cardinality(), 6);
        let out: Vec<NumNodes> = bs.iter().collect();
        assert_eq!(out, vec![1, 2, 4, 5, 7, 8]);
    }

    #[test]
    fn test_new_all_unset_but() {
        // 0123456789
        // +  +  +  +
        let into: Vec<NumNodes> = (0..10).filter(|x| x % 3 == 0).collect();
        let bs = BitSet::new_all_unset_but(10, into.clone().into_iter());
        assert_eq!(bs.cardinality(), 4);
        let out: Vec<NumNodes> = bs.iter().collect();
        assert_eq!(out, into);
    }

    #[test]
    fn test_clone() {
        for n in [0, 1, 100] {
            let empty = BitSet::new(n);
            let copied = empty.clone();
            assert_eq!(copied.len(), n);
            assert_eq!(copied.cardinality(), 0);
        }

        for n in [10, 50, 100] {
            let mut orig = BitSet::new(n);
            for i in 0..n / 5 {
                orig.set_bit(i % 3);
            }

            let copied = orig.clone();
            assert_eq!(copied, orig);
            assert_eq!(copied.len(), orig.len());
            assert_eq!(copied.cardinality(), orig.cardinality());

            for i in 0..n {
                assert_eq!(copied[i], orig[i]);
            }
        }
    }
}
