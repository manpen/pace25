use std::{mem, ptr::copy_nonoverlapping};

use prelude::BitSetShardImpl;

use super::*;

pub struct BitSetArray<Index>
where
    Index: PrimIndex,
{
    data: *mut Bitmask,
    len: usize,
    capacity: usize,
    number_of_bits: usize,
    sets: Vec<BitSetShardImpl<Index>>,
}

impl<Index: PrimIndex> Drop for BitSetArray<Index> {
    fn drop(&mut self) {
        unsafe {
            mem::drop(Vec::from_raw_parts(self.data, self.len, self.capacity));
        }
    }
}

impl<Index: PrimIndex> Clone for BitSetArray<Index> {
    fn clone(&self) -> Self {
        let mut buffer = vec![0; self.len];
        let capacity = buffer.capacity();
        let data = buffer.as_mut_ptr();
        mem::forget(buffer);

        unsafe {
            copy_nonoverlapping(self.data, data, self.len);
        }

        let mut clone = Self {
            data,
            capacity,
            len: self.len,
            number_of_bits: self.number_of_bits,
            sets: self.sets.clone(),
        };

        let words_per_set = (self.number_of_bits + BITS_IN_MASK - 1) / BITS_IN_MASK;
        for (i, set) in clone.sets.iter_mut().enumerate() {
            set.switch_to_new_pointer(unsafe { data.add(i * words_per_set) });
        }

        clone
    }
}

impl<Index: PrimIndex> BitSetArray<Index> {
    pub fn new(number_of_sets: Index, number_of_bits: Index) -> Self {
        let usize_num_bits = number_of_bits.to_usize().unwrap();
        let usize_num_sets = number_of_sets.to_usize().unwrap();

        let words_per_set = (usize_num_bits + BITS_IN_MASK - 1) / BITS_IN_MASK;
        let len = usize_num_sets * words_per_set;

        // we offload the allocation to a vector that we `forget` at the end of this scope
        // it will be `rebuild` in our `Drop` implementation
        let mut buffer = vec![0; len];
        let capacity = buffer.capacity();
        let data = buffer.as_mut_ptr();
        mem::forget(buffer);

        Self {
            data,
            capacity,
            len,
            number_of_bits: usize_num_bits,
            sets: (0..usize_num_sets)
                .map(|i| {
                    BitSetShardImpl::<Index>::new(
                        unsafe { data.add(i * words_per_set) },
                        number_of_bits,
                    )
                })
                .collect(),
        }
    }

    pub fn get_set(&self, set: Index) -> &BitSetShardImpl<Index> {
        &self.sets[set.to_usize().unwrap()]
    }

    pub fn get_set_mut(&mut self, set: Index) -> &mut BitSetShardImpl<Index> {
        &mut self.sets[set.to_usize().unwrap()]
    }

    pub fn number_of_sets(&self) -> Index {
        Index::from(self.sets.len()).unwrap()
    }

    pub fn number_of_bits(&self) -> Index {
        Index::from(self.number_of_bits).unwrap()
    }
}

#[cfg(test)]
mod test {
    use bitmask_stream_consumer::BitmaskStreamConsumer;
    use prelude::ToBitmaskStream;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    use super::*;

    #[test]
    fn new() {
        for num_sets in [0u32, 1, 10, 100] {
            for num_bits in [0u32, 1, 10, 100] {
                let array = BitSetArray::new(num_sets, num_bits);

                assert_eq!(array.number_of_bits(), num_bits);
                assert_eq!(array.number_of_sets(), num_sets);
            }
        }
    }

    #[test]
    fn clone_new() {
        for num_sets in [0u32, 1, 10, 100] {
            for num_bits in [0u32, 1, 10, 100] {
                let org = BitSetArray::new(num_sets, num_bits);
                let array = org.clone();

                assert_eq!(array.number_of_bits(), num_bits);
                assert_eq!(array.number_of_sets(), num_sets);
            }
        }
    }

    #[test]
    fn clone_used() {
        let mut rng = Pcg64Mcg::seed_from_u64(1234);
        for num_sets in [1u32, 10, 100] {
            for num_bits in [1u32, 10, 100] {
                let mut org = BitSetArray::new(num_sets, num_bits);

                for _ in 0..(4 * num_sets * num_bits) {
                    let s = rng.gen_range(0..num_sets);
                    let b = rng.gen_range(0..num_bits);
                    org.get_set_mut(s).flip_bit(b);
                }

                let clone = org.clone();
                let mut array = clone.clone();

                for s in 0..num_sets {
                    assert_eq!(org.get_set(s).cardinality(), array.get_set(s).cardinality());
                    assert!(
                        (org.get_set(s).bitmask_stream() ^ array.get_set(s).bitmask_stream())
                            .are_all_cleared()
                    );
                }

                for _ in 0..(2 * num_sets * num_bits) {
                    let s = rng.gen_range(0..num_sets);
                    let b = rng.gen_range(0..num_bits);
                    array.get_set_mut(s).flip_bit(b);
                }

                for s in 0..num_sets {
                    assert_eq!(org.get_set(s).cardinality(), clone.get_set(s).cardinality());
                    assert!(
                        (org.get_set(s).bitmask_stream() ^ clone.get_set(s).bitmask_stream())
                            .are_all_cleared()
                    );
                }
            }
        }
    }
}
