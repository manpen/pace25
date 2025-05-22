/// This macro generates the implementation of the `BitSet` interface for
/// a given stumb implementation. We require the following methods to be
/// present: [`Self::_get_number_of_bits`], [`Self::_set_number_of_bits`],
/// [`Self::_get_cardinality`], [`Self::_set_cardinality`], [`Self::_as_slice`],
/// [`Self::_as_mut_slice`].
macro_rules! impl_bitset {
    () => {
        /// Returns the number of bits set.
        /// ```
        /// use stream_bitset::prelude::*;
        /// let mut set = BitSet32::new(14);
        /// assert_eq!(set.cardinality(), 0);
        ///
        /// set.set_bit(1);
        /// set.set_bit(2);
        /// set.set_bit(2); // no change, already set
        /// assert_eq!(set.cardinality(), 2);
        ///
        /// set.clear_bit(1);
        /// assert_eq!(set.cardinality(), 1);
        /// ```
        pub fn cardinality(&self) -> Index {
            Index::from(self._get_cardinality()).unwrap()
        }

        /// Returns the number of bits accessible in the container.
        /// If `n` is returned, exactly bits `0..n` are available.
        ///
        /// # Examples
        ///
        /// ```
        /// use stream_bitset::prelude::*;
        /// let set = BitSet32::new(13);
        /// assert_eq!(set.number_of_bits(), 13);
        /// ```
        pub fn number_of_bits(&self) -> Index {
            Index::from(self._get_number_of_bits()).unwrap()
        }

        /// Returns true iff all bits are set.
        ///
        /// # Examples
        /// ```
        /// use stream_bitset::prelude::*;
        /// let mut set = BitSet32::new(2);
        /// assert!(!set.are_all_set());
        /// set.set_bit(0);
        /// assert!(!set.are_all_set());
        /// set.set_bit(1);
        /// assert!(set.are_all_set());
        /// ```
        pub fn are_all_set(&self) -> bool {
            self.cardinality() == self.number_of_bits()
        }

        /// Returns true iff all bits are unset.
        ///
        /// # Examples
        /// ```
        /// use stream_bitset::prelude::*;
        /// let mut set = BitSet32::new(2);
        /// assert!(set.are_all_unset());
        /// set.set_bit(0);
        /// assert!(!set.are_all_unset());
        /// ```
        pub fn are_all_unset(&self) -> bool {
            self.cardinality().is_zero()
        }

        /// True if the `id`-th bit is set.
        ///
        /// # Examples
        ///
        /// ```
        /// use stream_bitset::bitset::*;
        /// let mut set = BitSet32::new(14);
        /// assert_eq!(set.get_bit(2), false);
        /// set.set_bit(2);
        /// assert_eq!(set.get_bit(2), true);
        /// ```
        #[inline(always)]
        pub fn get_bit(&self, id: Index) -> bool {
            let (word_idx, bit_idx) = self.word_bit_index(id);
            self._as_slice()[word_idx].is_ith_bit_set(bit_idx)
        }

        /// Assigns the value `value` to the `id`-th bit and returns the previous value.
        ///
        /// # Example
        ///
        /// ```
        /// use stream_bitset::bitset::*;
        /// let mut set = BitSet32::new(14);
        ///
        /// assert_eq!(set.assign_bit(2, true), false);
        /// assert_eq!(set.assign_bit(2, true), true);
        /// assert_eq!(set.cardinality(), 1);
        /// assert_eq!(set.assign_bit(2, false), true);
        /// assert_eq!(set.cardinality(), 0);
        /// ```
        #[inline(always)]
        pub fn assign_bit(&mut self, id: Index, value: bool) -> bool {
            let (word_idx, bit_idx) = self.word_bit_index(id);
            let prev = self._as_mut_slice()[word_idx].assign_ith_bit(bit_idx, value);

            let mut cardinality = self._get_cardinality();
            cardinality -= prev as usize;
            cardinality += value as usize;
            self._set_cardinality(cardinality);

            prev
        }

        /// Sets the `id`-th bit to `true` and returns the previous value.
        ///
        /// # Example
        ///
        /// ```
        /// use stream_bitset::bitset::*;
        ///
        /// let mut set = BitSet32::new(14);
        /// assert_eq!(set.set_bit(3), false);
        /// assert_eq!(set.set_bit(4), false);
        /// assert_eq!(set.cardinality(), 2);
        /// assert_eq!(set.set_bit(3), true);
        /// ```
        #[inline(always)]
        pub fn set_bit(&mut self, id: Index) -> bool {
            self.assign_bit(id, true)
        }

        /// Clears (set to false) the `id`-th bit and returns the previous value.
        ///
        /// # Example
        ///
        /// ```
        /// use stream_bitset::bitset::*;
        ///
        /// let mut set = BitSet32::new(14);
        /// assert_eq!(set.set_bit(3), false);
        /// assert_eq!(set.clear_bit(3), true);
        /// assert_eq!(set.cardinality(), 0);
        /// ```
        #[inline(always)]
        pub fn clear_bit(&mut self, id: Index) -> bool {
            self.assign_bit(id, false)
        }

        /// Flips the `id`-th bit and returns the previous value.
        ///
        /// # Example
        ///
        /// ```
        /// use stream_bitset::bitset::*;
        ///
        /// let mut set = BitSet32::new(14);
        /// assert_eq!(set.set_bit(3), false);
        /// assert_eq!(set.flip_bit(3), true);
        /// assert_eq!(set.cardinality(), 0);
        /// assert_eq!(set.flip_bit(3), false);
        /// assert_eq!(set.cardinality(), 1);
        /// ```
        #[inline(always)]
        pub fn flip_bit(&mut self, id: Index) -> bool {
            self.assign_bit(id, !self.get_bit(id))
        }

        /// Takes an iterator of bit ids and sets them to `common_value`.
        ///
        /// # Example
        /// ```
        /// use stream_bitset::bitset::*;
        /// let mut set = BitSet32::new(14);
        ///
        /// set.assign_bits(0u32..3, true);
        /// assert_eq!(set.cardinality(), 3);
        ///
        /// set.assign_bits([2u32, 5], false);
        /// assert_eq!(set.cardinality(), 2);
        /// ```
        pub fn assign_bits<I, B, T>(&mut self, bits: I, common_value: bool)
        where
            I: IntoIterator<Item = B>,
            B: Borrow<T>,
            T: Unsigned + ToPrimitive + Copy,
        {
            for id in bits {
                self.assign_bit(Index::from(*id.borrow()).unwrap(), common_value);
            }
        }

        /// Takes an iterator of bit ids and sets each bit to true.
        ///
        /// # Example
        /// ```
        /// use stream_bitset::bitset::*;
        /// let mut set = BitSet32::new(14);
        ///
        /// set.set_bits(0u32..3);
        /// assert_eq!(set.cardinality(), 3);
        ///
        /// set.set_bits([2u32, 5]);
        /// assert_eq!(set.cardinality(), 4);
        /// ```
        pub fn set_bits<I, B, T>(&mut self, bits: I)
        where
            I: IntoIterator<Item = B>,
            B: Borrow<T>,
            T: Unsigned + ToPrimitive + Copy,
        {
            for id in bits {
                self.set_bit(Index::from(*id.borrow()).unwrap());
            }
        }

        /// Takes an iterator of bit ids and clears each bit (i.e. set it to false).
        ///
        /// # Example
        /// ```
        /// use stream_bitset::bitset::*;
        /// let mut set = BitSet32::new(14);
        ///
        /// set.set_bits(0u32..10);
        /// assert_eq!(set.cardinality(), 10);
        ///
        /// set.clear_bits([2u32, 5, 11]);
        /// assert_eq!(set.cardinality(), 8);
        /// ```
        pub fn clear_bits<I, B, T>(&mut self, bits: I)
        where
            I: IntoIterator<Item = B>,
            B: Borrow<T>,
            T: Unsigned + ToPrimitive + Copy,
        {
            for id in bits {
                self.clear_bit(Index::from(*id.borrow()).unwrap());
            }
        }

        /// Takes an iterator of bit ids and flips each bit.
        ///
        /// # Example
        /// ```
        /// use stream_bitset::bitset::*;
        /// let mut set = BitSet32::new(14);
        ///
        /// set.set_bits(0u32..10);
        /// assert_eq!(set.cardinality(), 10);
        ///
        /// set.flip_bits([2u32, 5, 11]);
        /// assert_eq!(set.cardinality(), 9);
        /// ```
        pub fn flip_bits<I, B, T>(&mut self, bits: I)
        where
            I: IntoIterator<Item = B>,
            B: Borrow<T>,
            T: Unsigned + ToPrimitive + Copy,
        {
            for id in bits {
                self.flip_bit(Index::from(*id.borrow()).unwrap());
            }
        }

        /// Clears all bits
        ///
        /// # Example
        /// ```
        /// use stream_bitset::bitset::*;
        /// let mut set = BitSet32::new(14);
        /// set.set_bits(0u32..10);
        /// assert_eq!(set.cardinality(), 10);
        /// set.clear_all();
        /// assert_eq!(set.cardinality(), 0);
        /// ```
        pub fn clear_all(&mut self) {
            self._as_mut_slice().fill(0);
            self._set_cardinality(0);
        }

        /// Sets all bits
        ///
        /// # Example
        /// ```
        /// use stream_bitset::bitset::*;
        /// let mut set = BitSet32::new(14);
        /// set.set_bits(0u32..10);
        /// assert_eq!(set.cardinality(), 10);
        /// set.set_all();
        /// assert_eq!(set.cardinality(), 14);
        /// ```
        pub fn set_all(&mut self) {
            self._as_mut_slice().fill(!0);
            self.mask_last_element();

            self._set_cardinality(self._get_number_of_bits());
        }

        /// Flips all bits.
        ///
        /// # Example
        /// ```
        /// use stream_bitset::bitset::*;
        /// let mut set = BitSet32::new(14);
        /// set.set_bits(0u32..10);
        /// set.flip_all();
        /// assert_eq!(set.cardinality(), 4);
        /// ```
        pub fn flip_all(&mut self) {
            for word in self._as_mut_slice() {
                *word = !*word;
            }

            self.mask_last_element();
            self._set_cardinality(self._get_number_of_bits() - self._get_cardinality());
        }

        /// Returns an iterator over the indices of all set bits.
        ///
        /// # Example
        /// ```
        /// use stream_bitset::bitset::*;
        /// let bits_set = vec![0u32, 1, 5, 31];
        /// let set = BitSet32::new_with_bits_set(32, bits_set.clone());
        /// let itered : Vec<_> = set.iter_set_bits().collect();
        /// assert_eq!(itered, bits_set);
        /// ```
        pub fn iter_set_bits(&self) -> impl Iterator<Item = Index> + '_ {
            self.bitmask_stream().iter_set_bits()
        }

        /// Returns an iterator over the indices of all cleared bits.
        ///
        /// # Example
        /// ```
        /// use stream_bitset::bitset::*;
        /// let bits_set = vec![0u32, 1, 5, 31];
        /// let set = BitSet32::new_with_bits_cleared(32, bits_set.clone());
        /// let itered : Vec<_> = set.iter_cleared_bits().collect();
        /// assert_eq!(itered, bits_set);
        /// ```
        pub fn iter_cleared_bits(&self) -> impl Iterator<Item = Index> + '_ {
            self.bitmask_stream().iter_cleared_bits()
        }

        /// Returns true iff at least every element of `self` is also contained in `other`.
        ///
        /// # Example
        /// ```
        /// use stream_bitset::bitset::*;
        /// let set1 = BitSet32::new_with_bits_set(32, vec![0u32, 1, 5]);
        /// let set2 = BitSet32::new_with_bits_set(32, vec![0u32, 1, 5, 31]);
        ///
        /// assert!(set1.is_subset_of(&set2));
        /// assert!(!set2.is_subset_of(&set1));
        /// ```
        pub fn is_subset_of(&self, other: &Self) -> bool {
            let other = other.bitmask_stream();
            self.bitmask_stream().is_subset_of(other)
        }

        /// Returns the index `Some(i)` of the first set bit with `i >= id` or `None` if no such bit exists.
        ///
        /// # Example
        /// ```
        /// use stream_bitset::bitset::*;
        /// let set = BitSet32::new_with_bits_set(210, vec![0u32, 1, 5, 100, 200]);
        ///
        /// assert_eq!(set.get_first_set_index_atleast(0), Some(0));
        /// assert_eq!(set.get_first_set_index_atleast(1), Some(1));
        /// assert_eq!(set.get_first_set_index_atleast(2), Some(5));
        /// assert_eq!(set.get_first_set_index_atleast(6), Some(100));
        /// assert_eq!(set.get_first_set_index_atleast(100), Some(100));
        /// assert_eq!(set.get_first_set_index_atleast(101), Some(200));
        /// ```
        pub fn get_first_set_index_atleast(&self, id: Index) -> Option<Index> {
            if id >= self.number_of_bits() {
                return None;
            }

            let (mut word_idx, bit_idx) = self.word_bit_index(id);
            let mut first_word = self._as_slice()[word_idx];
            first_word &= !(Bitmask::ith_bit_set(bit_idx) - 1);

            if first_word != 0 {
                return Some(
                    Index::from(word_idx * BITS_IN_MASK + first_word.trailing_zeros() as usize)
                        .unwrap(),
                );
            }

            for &word in self._as_slice()[word_idx + 1..].iter() {
                word_idx += 1;
                if word != 0 {
                    let id = Index::from(word_idx * BITS_IN_MASK + word.trailing_zeros() as usize)
                        .unwrap();
                    return (id < self.number_of_bits()).then_some(id);
                }
            }

            None
        }

        pub fn or_streams<I: Iterator<Item = S>, S: ToBitmaskStream>(&mut self, iter: I) {
            for other in iter {
                for (m, StreamElement(o)) in
                    self._as_mut_slice().iter_mut().zip(other.bitmask_stream())
                {
                    *m |= o;
                }
            }
            self.recompute_cardinality();
        }

        #[inline(always)]
        pub fn as_slice(&self) -> &[Bitmask] {
            self._as_slice()
        }

        /// Asserts that the provided `id` is in the range of the container.
        /// If so, it does return the word index (i.e. into `self.data`) and the
        /// bit index (i.e. into `self.data[word_idx]`).
        ///
        /// # Panics
        /// If the provided `id` is out of range.
        fn word_bit_index(&self, id: Index) -> (usize, usize) {
            let id = id.to_usize().unwrap();
            assert!(
                id < self._get_number_of_bits(),
                "id {id} >= number of bits {}",
                self._get_number_of_bits()
            );
            let word_idx = id / BITS_IN_MASK;
            let bit_idx = id % BITS_IN_MASK;
            (word_idx, bit_idx)
        }

        /// Clears all bits in the last element of `self.data` that
        fn mask_last_element(&mut self) {
            let remainder = self._get_number_of_bits() % BITS_IN_MASK;
            if remainder > 0 {
                let mask = (1 << remainder) - 1;
                *self._as_mut_slice().last_mut().unwrap() &= mask;
            }
        }

        /// Recomputes the cardinality of the set and updates `self.cardinality`.
        fn recompute_cardinality(&mut self) {
            self._set_cardinality(
                self._as_slice()
                    .iter()
                    .map(|w| w.count_ones() as usize)
                    .sum(),
            );
        }
    };
}

macro_rules! impl_bitset_fmt_debug {
    () => {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "BitSet(num_bits={} card={} [",
                self._get_number_of_bits(),
                self._get_cardinality()
            )?;
            for i in 0..self._get_number_of_bits() {
                write!(f, "{}", self.get_bit(I::from(i).unwrap()) as u32)?;
                if i % 8 == 7 && i + 1 < self._get_number_of_bits() {
                    write!(f, " ")?;
                }
            }
            write!(f, "])")
        }
    };
}

pub(crate) use impl_bitset;
pub(crate) use impl_bitset_fmt_debug;
