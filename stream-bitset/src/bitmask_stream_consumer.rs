use std::marker::PhantomData;
use std::ops::Sub;

use super::prelude::*;
use super::*;

/// A collection of methods that consume a bitmask stream.
pub trait BitmaskStreamConsumer {
    /// Returns the number of bits set in the stream.
    ///
    /// # Example
    ///
    /// ```
    /// use stream_bitset::prelude::*;
    /// let set = BitSet32::new_with_bits_set(32, [0u32, 1, 5]);
    /// assert_eq!(set.bitmask_stream().cardinality(), 3);
    /// ```
    fn cardinality(self) -> usize;

    /// Returns the number of bits set in the stream up to `n`.
    /// The returned value is guaranteed to be less than or equal to `n`.
    /// This is useful for early termination of compution.
    ///
    /// # Example
    ///
    /// ```
    /// use stream_bitset::prelude::*;
    /// let set = BitSet32::new_with_bits_set(32, [0u32, 1, 5]);
    /// assert_eq!(set.bitmask_stream().count_ones_upto(2), 2);
    /// ```
    fn count_ones_upto(self, n: usize) -> usize;

    /// Returns true if all bits in the stream are cleared.
    ///
    /// # Example
    /// ```
    /// use stream_bitset::prelude::*;
    /// assert!(BitSet32::new(8).bitmask_stream().are_all_cleared());
    /// assert!(!BitSet32::new_with_bits_set(8, [0u32])
    ///             .bitmask_stream().are_all_cleared());
    /// ```
    fn are_all_cleared(self) -> bool;

    /// Returns true if all bits in the stream are set.
    ///
    /// # Example
    /// ```
    /// use stream_bitset::prelude::*;
    /// assert!(BitSet32::new_with_bits_set(8, 0u32..8).bitmask_stream().are_all_set());
    /// assert!(!BitSet32::new_with_bits_set(8, 0u32..7).bitmask_stream().are_all_set());
    /// ```
    fn are_all_set(self) -> bool;

    type IterSetBits<I>: Iterator<Item = I>
    where
        I: PrimIndex;
    /// Returns an iterator of type [`Self::IterSetBits`] that yields the indices of the bits set in the stream.
    ///
    /// # Example
    ///
    /// ```
    /// use stream_bitset::prelude::*;
    /// let bits_set = vec![0u32, 1, 5, 31];
    /// let set = BitSet32::new_with_bits_set(32, bits_set.clone());
    /// let itered : Vec<u32> = set.bitmask_stream().iter_set_bits().collect();
    /// assert_eq!(itered, bits_set);
    /// ```
    fn iter_set_bits<I: PrimIndex>(self) -> Self::IterSetBits<I>;

    type IterClearedBits<I>: Iterator<Item = I>
    where
        I: PrimIndex;
    /// Returns an iterator of type [`Self::IterClearedBits`] that yields the indices of the bits cleared in the stream.
    ///
    /// # Example
    ///
    /// ```
    /// use stream_bitset::prelude::*;
    /// let bits_set = vec![0u32, 1, 5, 31];
    /// let set = BitSet32::new_with_bits_cleared(32, bits_set.clone());
    /// let itered : Vec<u32> = set.bitmask_stream().iter_cleared_bits().collect();
    /// assert_eq!(itered, bits_set);
    /// ```
    fn iter_cleared_bits<I: PrimIndex>(self) -> Self::IterClearedBits<I>;

    /// Returns true if at least every bit of `self` is also set in `other`.
    ///
    /// # Example
    ///
    /// ```
    /// use stream_bitset::prelude::*;
    ///
    /// let set1 = BitSet32::new_with_bits_set(32, [0u32, 1, 5]);
    /// let set2 = BitSet32::new_with_bits_set(32, [0u32, 1, 5, 31]);
    ///
    /// assert!(set1.bitmask_stream().is_subset_of(set2.bitmask_stream()));
    /// assert!(!set2.bitmask_stream().is_subset_of(set1.bitmask_stream()));
    /// ```
    fn is_subset_of<S, T>(self, other: S) -> bool
    where
        S: ToBitmaskStream,
        T: BitmaskStreamConsumer,
        Self: Sub<S, Output = T>;
}

/// Converts a bitmask stream into an iterator of indices of bits that are set/cleared in the stream.
/// Do not construct this struct directly and instead use [`BitmaskStreamConsumer::iter_set_bits`] or
/// [`BitmaskStreamConsumer::iter_cleared_bits`].
pub struct BitmaskStreamToIndices<S, I, const COUNT_ONES: bool>
where
    S: BitmaskStream,
    I: PrimIndex,
{
    stream: S,
    current_mask: Bitmask,
    index: usize,
    _index: PhantomData<I>,
}

impl<S, I, const COUNT_ONES: bool> BitmaskStreamToIndices<S, I, COUNT_ONES>
where
    S: BitmaskStream,
    I: PrimIndex,
{
    pub fn new(mut stream: S) -> Self {
        let current_mask = stream.next().map(|StreamElement(x)| x).unwrap_or(0);
        BitmaskStreamToIndices {
            stream,
            index: 0,
            current_mask,
            _index: Default::default(),
        }
    }
}

impl<S, I, const COUNT_ONES: bool> Iterator for BitmaskStreamToIndices<S, I, COUNT_ONES>
where
    S: BitmaskStream,
    I: PrimIndex,
{
    type Item = I;

    fn next(&mut self) -> Option<Self::Item> {
        let empty_word = if COUNT_ONES { 0 } else { !0 };

        while self.current_mask == empty_word {
            let StreamElement(x) = self.stream.next()?;
            self.current_mask = x;
            self.index += BITS_IN_MASK;
        }

        let id = self.index
            + if COUNT_ONES {
                let first_one = self.current_mask.trailing_zeros() as usize;
                self.current_mask.clear_ith_bit(first_one);
                first_one
            } else {
                let first_one = self.current_mask.trailing_ones() as usize;
                self.current_mask.set_ith_bit(first_one);
                first_one
            };

        (id < self.stream.number_of_bits()).then(|| I::from(id).unwrap())
    }
}

impl<T: BitmaskStream> BitmaskStreamConsumer for T {
    fn cardinality(self) -> usize {
        self.map(|StreamElement(x)| x.count_ones() as usize).sum()
    }

    fn are_all_cleared(mut self) -> bool {
        self.all(|StreamElement(x)| x == 0)
    }

    fn are_all_set(mut self) -> bool {
        let bits_in_last_word = self.number_of_bits() % BITS_IN_MASK;
        if bits_in_last_word == 0 {
            return self.all(|StreamElement(x)| x == !0);
        }

        let mut prev = self.next();
        for StreamElement(x) in self {
            if prev.unwrap() != StreamElement(!0) {
                return false;
            }

            prev = Some(StreamElement(x));
        }

        if let Some(StreamElement(x)) = prev {
            assert!(bits_in_last_word != 0);
            x == !(!0 << bits_in_last_word)
        } else {
            true
        }
    }

    fn count_ones_upto(self, n: usize) -> usize {
        let mut count = 0;
        for StreamElement(x) in self {
            count += x.count_ones() as usize;
            if count >= n {
                return n;
            }
        }
        count
    }

    type IterSetBits<I> = BitmaskStreamToIndices<Self, I, true> where I: PrimIndex;
    fn iter_set_bits<I: PrimIndex>(self) -> Self::IterSetBits<I> {
        BitmaskStreamToIndices::new(self)
    }

    type IterClearedBits<I> = BitmaskStreamToIndices<Self, I, false> where I: PrimIndex;
    fn iter_cleared_bits<I: PrimIndex>(self) -> Self::IterClearedBits<I> {
        BitmaskStreamToIndices::new(self)
    }

    fn is_subset_of<S, O>(self, other: S) -> bool
    where
        S: ToBitmaskStream,
        O: BitmaskStreamConsumer,
        Self: Sub<S, Output = O>,
    {
        (self - other).are_all_cleared()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    fn random_bitsets<R: Rng>(rng: &mut R) -> impl Iterator<Item = BitSet32> + '_ {
        (1u32..200)
            .map(move |n| BitSet32::new_with_bits_set(n, (0..n / 2).map(|_| rng.gen_range(0..n))))
    }

    #[test]
    fn cardinality() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x1234);
        for bitset in random_bitsets(&mut rng) {
            let stream = bitset.bitmask_stream();
            assert_eq!(stream.cardinality(), bitset.cardinality() as usize);
        }
    }

    #[test]
    fn count_ones_upto() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x12346);
        for bitset in random_bitsets(&mut rng) {
            let stream = bitset.bitmask_stream();
            let max = (bitset.number_of_bits() / 3) as usize;
            let cnt = stream.count_ones_upto(max);
            assert!(cnt <= max);
            assert_eq!(cnt, (bitset.cardinality() as usize).min(max));
        }
    }

    #[test]
    fn are_all_set() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x123467);
        for n in 1u32..300 {
            let mut set = BitSet32::new(n);
            set.set_all();

            assert!(set.bitmask_stream().are_all_set());

            set.clear_bit(rng.gen_range(0..n));

            assert!(!set.bitmask_stream().are_all_set());
        }
    }

    #[test]
    fn are_all_cleared() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x123467);
        for n in 1u32..300 {
            let mut set = BitSet32::new(n);

            assert!(set.bitmask_stream().are_all_cleared());

            set.set_bit(rng.gen_range(0..n));

            assert!(!set.bitmask_stream().are_all_cleared());
        }
    }

    #[test]
    fn iter_set_bits() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x123467);
        for bitset in random_bitsets(&mut rng) {
            let iter = bitset.bitmask_stream().iter_set_bits();
            let mut count = 0;
            for index in iter {
                assert!(bitset.get_bit(index));
                count += 1;
            }
            assert_eq!(count, bitset.cardinality());
        }
    }

    #[test]
    fn iter_cleared_bits() {
        let mut rng = Pcg64Mcg::seed_from_u64(0x123467);
        for bitset in random_bitsets(&mut rng) {
            let iter = bitset.bitmask_stream().iter_cleared_bits();
            let mut count = 0;
            for index in iter {
                assert!(!bitset.get_bit(index));
                count += 1;
            }
            assert_eq!(count, (bitset.number_of_bits() - bitset.cardinality()));
        }
    }
}
