use std::cmp::Ordering;

use num::{FromPrimitive, ToPrimitive};

use crate::graph::Node;

/// Trait for all possible types that can be used as a key for a RadixHeap
pub trait RadixKey: Copy + Default + PartialOrd {
    /// Number of bits of Self
    const NUM_BITS: usize;

    /// Similarity to another instance of Self
    fn radix_similarity(&self, other: &Self) -> usize;
}

macro_rules! radix_key_impl_float {
    ($($t:ty),*) => {
        $(
            impl RadixKey for $t {
                const NUM_BITS: usize = (std::mem::size_of::<$t>() * 8);

                fn radix_similarity(&self, other: &Self) -> usize {
                    (self.to_bits() ^ other.to_bits()).leading_zeros() as usize
                }
            }
        )*
    };
}

macro_rules! radix_key_impl_int {
    ($($t:ty),*) => {
        $(
            impl RadixKey for $t {
                const NUM_BITS: usize = (std::mem::size_of::<$t>() * 8);

                fn radix_similarity(&self, other: &Self) -> usize {
                    (self ^ other).leading_zeros() as usize
                }
            }
        )*
    };
}

radix_key_impl_float!(f32, f64);
radix_key_impl_int!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

/// Inverted radix similarity (=> how far away is other from self)
///
/// Not part of trait to not expose to user
fn radix_distance<K: RadixKey>(lhs: &K, rhs: &K) -> usize {
    let dist = K::NUM_BITS - lhs.radix_similarity(rhs);
    debug_assert!(dist <= K::NUM_BITS);
    dist
}

/// Marker trait for types that can be used as values in IndexedRadixHeap (=> have to be convertible to usize)
pub trait RadixValue: FromPrimitive + ToPrimitive + Default + Copy {}
impl<T: FromPrimitive + ToPrimitive + Default + Copy> RadixValue for T {}

/// Stores all key-value-pairs of the same similarity
type Bucket<K, V> = Vec<(K, V)>;

const NOT_IN_HEAP: usize = usize::MAX;

/// # IndexedRadixHeap: A Radix-MinHeap
///
/// Allows fast insertions/deletions/queries into elements sorted by associated RadixKey-type.
/// Elements have to be convertible to a usize smaller than a given value to allow for fast
/// existence queries.
///
/// ### IMPORTANT
/// NUM_BUCKETS must be equal to K::NUM_BITS + 1! This will be checked when creating the heap.
#[derive(Debug)]
pub struct IndexedRadixHeap<K: RadixKey, V: RadixValue, const NUM_BUCKETS: usize> {
    /// Number of elements in the heap
    len: usize,
    /// Current top element (last element removed)
    top: K,
    /// All buckets
    buckets: [Bucket<K, V>; NUM_BUCKETS],
    /// A pointer for each element to where it is in the heap
    pointer: Vec<(usize, usize)>,
}

/// A heap with nodes as keys and values
pub type NodeHeap = IndexedRadixHeap<Node, Node, 33>;

impl<K: RadixKey, V: RadixValue, const NUM_BUCKETS: usize> IndexedRadixHeap<K, V, NUM_BUCKETS> {
    /// Creates a new heap with a given top element and a maximum number of elements
    pub fn new(n: usize, top: K) -> Self {
        // Only accept heaps with enough buckets
        assert!(NUM_BUCKETS > K::NUM_BITS);
        Self {
            len: 0,
            top,
            buckets: array_init::array_init(|_| Vec::new()),
            pointer: vec![(NOT_IN_HEAP, NOT_IN_HEAP); n],
        }
    }

    /// Removes all elements from the heap and sets the top value to `top`
    pub fn reset(&mut self, top: K) {
        self.len = 0;
        self.top = top;
        self.buckets.iter_mut().for_each(|b| {
            b.drain(..).for_each(|(_, v)| {
                self.pointer[v.to_usize().unwrap()].0 = NOT_IN_HEAP;
            })
        });
    }

    /// Tries to push a (key, value)-pair on the heap. Returns *true* if successfull and *false* if not
    pub fn try_push(&mut self, key: K, value: V) -> bool {
        if self.pointer[value.to_usize().unwrap()].0 != NOT_IN_HEAP {
            return false;
        }

        self.push(key, value);

        true
    }

    /// Force pushes a (key, value)-pair on the heap. The caller is responsible for ensuring bounds
    /// and that value is not already on the heap.
    pub fn push(&mut self, key: K, value: V) {
        debug_assert!(self.pointer[value.to_usize().unwrap()].0 == NOT_IN_HEAP);

        let bucket = radix_distance(&key, &self.top);
        self.pointer[value.to_usize().unwrap()] = (bucket, self.buckets[bucket].len());
        self.buckets[bucket].push((key, value));
        self.len += 1;
    }

    /// Tries to remove a (key, value)-pair identified by the value from the heap.
    /// Returns *None* if no entry was found.
    pub fn try_remove(&mut self, value: V) -> Option<K> {
        let val = value.to_usize().unwrap();

        if val >= self.pointer.len() {
            return None;
        }

        let (bucket, position) = self.pointer[val];
        if bucket == NOT_IN_HEAP {
            return None;
        }

        Some(self._remove_inner(val, bucket, position))
    }

    /// Force removes a (key, value)-pair identified by the value from the heap.
    /// Panics if no value was found.
    pub fn remove(&mut self, value: V) -> K {
        let value = value.to_usize().unwrap();
        debug_assert!(value < self.pointer.len());

        let (bucket, position) = self.pointer[value];
        debug_assert_ne!(bucket, NOT_IN_HEAP);

        self._remove_inner(value, bucket, position)
    }

    /// Private function to remove an element from a specific position in a bucket and update the
    /// pointer. Returns the element. Panics if no entry was found.
    fn _remove_inner(&mut self, value: usize, bucket: usize, position: usize) -> K {
        let res = self.buckets[bucket].swap_remove(position);
        if self.buckets[bucket].len() > position {
            self.pointer[self.buckets[bucket][position].1.to_usize().unwrap()].1 = position;
        }
        self.len -= 1;

        self.pointer[value].0 = NOT_IN_HEAP;

        res.0
    }

    /// Updates the heap to find the new smallest element and re-order buckets accordingly
    fn update_buckets(&mut self) {
        let (buckets, repush) = match self.buckets.iter().position(|bucket| !bucket.is_empty()) {
            None | Some(0) => return,
            Some(index) => {
                let (buckets, rest) = self.buckets.split_at_mut(index);
                (buckets, &mut rest[0])
            }
        };

        self.top = repush
            .iter()
            .min_by(|(k1, _), (k2, _)| k1.partial_cmp(k2).unwrap())
            .unwrap()
            .0;

        repush.drain(..).for_each(|(key, value)| {
            let bucket = radix_distance(&key, &self.top);
            self.pointer[value.to_usize().unwrap()] = (bucket, buckets[bucket].len());
            buckets[bucket].push((key, value));
        });
    }

    /// Pop the smallest element from the heap. If there are multiple, break ties by key in
    /// tiebreaker.
    ///
    /// Returns *None* if no element is left on the heap
    pub fn pop_with_tiebreaker<T: PartialOrd>(&mut self, tiebreaker: &[T]) -> Option<(K, V)> {
        if self.buckets[0].is_empty() {
            self.update_buckets();
        }

        let (pos, (key, val)) =
            self.buckets[0]
                .iter()
                .enumerate()
                .min_by(|(_, (_, v1)), (_, (_, v2))| {
                    tiebreaker[v1.to_usize().unwrap()]
                        .partial_cmp(&tiebreaker[v2.to_usize().unwrap()])
                        .unwrap_or(Ordering::Equal)
                })?;

        let key = *key;
        let val = *val;

        self.top = key;
        self.buckets[0].swap_remove(pos);
        if self.buckets[0].len() > pos {
            self.pointer[self.buckets[0][pos].1.to_usize().unwrap()].1 = pos;
        }

        self.len -= 1;
        self.pointer[val.to_usize().unwrap()].0 = NOT_IN_HEAP;

        Some((key, val))
    }

    /// Pops the smallest element from the heap (no tiebreaker).
    /// Returns *None* if no element is left on the heap.
    pub fn pop(&mut self) -> Option<(K, V)> {
        let (key, val) = self.buckets[0].pop().or_else(|| {
            self.update_buckets();
            self.buckets[0].pop()
        })?;

        self.len -= 1;
        self.pointer[val.to_usize().unwrap()].0 = NOT_IN_HEAP;

        Some((key, val))
    }

    /// Returns the number of elements on the heap.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns *true* if there are no elements on the heap.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::radix::NodeHeap;

    #[test]
    fn test_radix_heap() {
        let mut heap = NodeHeap::new(6, 0);
        heap.push(3, 0);
        heap.push(2, 1);
        heap.push(6, 2);
        heap.push(2, 3);
        heap.push(1, 4);
        heap.push(6, 5);

        assert_eq!(heap.len(), 6);

        assert_eq!(heap.pop(), Some((1, 4)));
        assert_eq!(heap.pop_with_tiebreaker(&[6, 5, 4, 3, 2, 1]), Some((2, 3)));
        assert_eq!(heap.pop(), Some((2, 1)));
        assert_eq!(heap.pop_with_tiebreaker(&[1, 2, 3, 4, 5, 6]), Some((3, 0)));
        assert_eq!(heap.pop_with_tiebreaker(&[1, 2, 3, 4, 5, 6]), Some((6, 2)));
        assert_eq!(heap.pop(), Some((6, 5)));
        assert_eq!(heap.pop(), None);

        assert!(heap.is_empty());

        heap.push(3, 1);
        heap.push(3, 2);
        heap.push(1, 3);

        heap.pop();
        assert_eq!(heap.remove(2), 3);
        assert_eq!(heap.len(), 1);
        assert_eq!(heap.try_remove(8), None);
    }
}
