use std::cmp::Ordering;

use num::{FromPrimitive, ToPrimitive};

use crate::graph::Node;

pub trait RadixKey: Copy + Default + PartialOrd {
    const NUM_BITS: usize;

    fn radix_similarity(&self, other: &Self) -> usize;

    fn radix_distance(&self, other: &Self) -> usize {
        Self::NUM_BITS - self.radix_similarity(other)
    }
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

                #[inline]
                fn radix_similarity(&self, other: &Self) -> usize {
                    (self ^ other).leading_zeros() as usize
                }
            }
        )*
    };
}

radix_key_impl_float!(f32, f64);
radix_key_impl_int!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

pub trait RadixValue: FromPrimitive + ToPrimitive + Default + Copy {}
impl<T: FromPrimitive + ToPrimitive + Default + Copy> RadixValue for T {}

type Bucket<K, V> = Vec<(K, V)>;

#[derive(Debug)]
pub struct IndexedRadixHeap<K: RadixKey, V: RadixValue, const NUM_BUCKETS: usize> {
    len: usize,
    top: K,
    buckets: [Bucket<K, V>; NUM_BUCKETS],
    pointer: Vec<(u8, V)>,
}

pub type NodeHeap = IndexedRadixHeap<Node, Node, 33>;

impl<K: RadixKey, V: RadixValue, const NUM_BUCKETS: usize> IndexedRadixHeap<K, V, NUM_BUCKETS> {
    pub fn new(n: usize, top: K) -> Self {
        Self {
            len: 0,
            top,
            buckets: array_init::array_init(|_| Vec::new()),
            pointer: vec![(u8::MAX, V::default()); n],
        }
    }

    pub fn reset(&mut self, top: K) {
        self.len = 0;
        self.top = top;
        self.buckets.iter_mut().for_each(|b| {
            b.drain(..).for_each(|(_, v)| {
                self.pointer[v.to_usize().unwrap()].0 = u8::MAX;
            })
        });
    }

    pub fn push(&mut self, key: K, value: V) {
        if self.pointer[value.to_usize().unwrap()].0 != u8::MAX {
            return;
        }

        let bucket = key.radix_distance(&self.top);
        self.buckets[bucket].push((key, value));
        self.pointer[value.to_usize().unwrap()] = (
            bucket as u8,
            V::from_usize(self.buckets[bucket].len() - 1).unwrap(),
        );
        self.len += 1;
    }

    pub fn remove(&mut self, value: V) -> Option<K> {
        let value = value.to_usize().unwrap();
        if value >= self.pointer.len() {
            return None;
        }

        let (bucket, position) = self.pointer[value];
        if bucket == u8::MAX {
            return None;
        }

        let bucket = bucket as usize;
        let pos_usize = position.to_usize().unwrap();

        let res = self.buckets[bucket].swap_remove(pos_usize);
        if self.buckets[bucket].len() > pos_usize {
            self.pointer[self.buckets[bucket][pos_usize].1.to_usize().unwrap()].1 = position;
        }
        self.len -= 1;

        self.pointer[value].0 = u8::MAX;

        Some(res.0)
    }

    fn update(&mut self) {
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
            let bucket = key.radix_distance(&self.top);
            buckets[bucket].push((key, value));
            self.pointer[value.to_usize().unwrap()] = (
                bucket as u8,
                V::from_usize(buckets[bucket].len() - 1).unwrap(),
            );
        });
    }

    pub fn pop_with_tiebreaker<T: PartialOrd>(&mut self, tiebreaker: &[T]) -> Option<(K, V)> {
        if self.buckets[0].is_empty() {
            self.update();
        }

        let min_elem = self.buckets[0]
            .iter()
            .enumerate()
            .min_by(|(_, (k1, v1)), (_, (k2, v2))| {
                if k1 < k2
                    || (k1 == k2
                        && tiebreaker[v1.to_usize().unwrap()] < tiebreaker[v2.to_usize().unwrap()])
                {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            });

        if let Some((pos, (key, val))) = min_elem {
            let key = *key;
            let val = *val;

            self.top = key;
            self.buckets[0].swap_remove(pos);
            if self.buckets[0].len() > pos {
                self.pointer[self.buckets[0][pos].1.to_usize().unwrap()].1 =
                    V::from_usize(pos).unwrap();
            }

            self.len -= 1;
            self.pointer[val.to_usize().unwrap()].0 = u8::MAX;

            Some((key, val))
        } else {
            None
        }
    }

    pub fn pop(&mut self) -> Option<(K, V)> {
        let res = self.buckets[0].pop().or_else(|| {
            self.update();
            self.buckets[0].pop()
        });

        if let Some((key, val)) = res {
            self.len -= 1;
            self.pointer[val.to_usize().unwrap()].0 = u8::MAX;

            Some((key, val))
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

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
        assert_eq!(heap.remove(2), Some(3));
        assert_eq!(heap.len(), 1);
        assert_eq!(heap.remove(8), None);
    }
}
