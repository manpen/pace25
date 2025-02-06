use std::cmp::Ordering;

use rand::Rng;
use rand_distr::Distribution;

use crate::graph::Node;

/// Allows fast sampling of nodes where each node has a weight that is a power of 2
///
/// NUM_BUCKETS is the highest power of 2 (minus one) that is allowed
pub struct WeightedPow2Sampler<const NUM_BUCKETS_PLUS_TWO: usize> {
    /// in bucket[i] are all nodes of weight 2^i
    buckets: Vec<Node>,
    /// Pointer for each possible node in which bucket (and position inside the bucket) it is
    pointer: Vec<usize>,
    offsets: [usize; NUM_BUCKETS_PLUS_TWO],
    /// Total weight of all inserted nodes
    total_weight: usize,
}

impl<const NUM_BUCKETS_PLUS_TWO: usize> WeightedPow2Sampler<NUM_BUCKETS_PLUS_TWO> {
    /// Creates a new sampler for a maximum of n nodes and a given number of buckets.
    ///
    /// There can be at most 255 buckets.
    pub fn new(n: usize) -> Self {
        let mut offsets = [n; NUM_BUCKETS_PLUS_TWO];
        offsets[0] = 0;
        Self {
            buckets: (0..(n as Node)).collect(),
            pointer: (0..n).collect(),
            offsets,
            total_weight: 0,
        }
    }

    /// Directly sets the bucket for a given node
    pub fn set_bucket(&mut self, node: Node, weight: usize) {
        let pos = self.pointer[node as usize];

        let new_bucket = weight.min(NUM_BUCKETS_PLUS_TWO - 1);
        let old_bucket = (1..NUM_BUCKETS_PLUS_TWO)
            .find(|&b| pos < self.offsets[b])
            .unwrap()
            - 1;

        // Set weight
        if new_bucket > 0 {
            self.total_weight += 1 << (new_bucket - 1);
        }
        if old_bucket > 0 {
            self.total_weight -= 1 << (old_bucket - 1);
        }

        // Update buckets
        match old_bucket.cmp(&new_bucket) {
            Ordering::Greater => {
                self.pointer[self.buckets[self.offsets[old_bucket]] as usize] = pos;
                self.buckets.swap(pos, self.offsets[old_bucket]);
                for bucket in ((new_bucket + 1)..old_bucket).rev() {
                    self.pointer[self.buckets[self.offsets[bucket]] as usize] =
                        self.offsets[bucket + 1];
                    self.buckets
                        .swap(self.offsets[bucket], self.offsets[bucket + 1]);
                    self.offsets[bucket + 1] += 1;
                }
                self.pointer[node as usize] = self.offsets[new_bucket + 1];
                self.offsets[new_bucket + 1] += 1;
            }
            Ordering::Less => {
                self.pointer[self.buckets[self.offsets[old_bucket + 1] - 1] as usize] = pos;
                self.buckets.swap(pos, self.offsets[old_bucket + 1] - 1);
                for bucket in (old_bucket + 1)..new_bucket {
                    self.pointer[self.buckets[self.offsets[bucket + 1] - 1] as usize] =
                        self.offsets[bucket] - 1;
                    self.buckets
                        .swap(self.offsets[bucket] - 1, self.offsets[bucket + 1] - 1);
                    self.offsets[bucket] -= 1;
                }
                self.pointer[node as usize] = self.offsets[new_bucket] - 1;
                self.offsets[new_bucket] -= 1;
            }
            _ => {}
        };
    }

    /// Moves node into the new_bucket assuming that it was previously in bucket 0
    pub fn add_entry(&mut self, node: Node, new_bucket: usize) {
        self.total_weight += 1 << new_bucket;

        self.pointer[self.buckets[self.offsets[1] - 1] as usize] = self.pointer[node as usize];
        self.buckets
            .swap(self.pointer[node as usize], self.offsets[1] - 1);
        for bucket in 1..=new_bucket {
            self.pointer[self.buckets[self.offsets[bucket + 1] - 1] as usize] =
                self.offsets[bucket] - 1;
            self.buckets
                .swap(self.offsets[bucket] - 1, self.offsets[bucket + 1] - 1);
            self.offsets[bucket] -= 1;
        }
        self.pointer[node as usize] = self.offsets[new_bucket + 1] - 1;
        self.offsets[new_bucket + 1] -= 1;
    }

    /// Moves the node into bucket 0
    pub fn remove_entry(&mut self, node: Node) {
        let old_bucket = (1..NUM_BUCKETS_PLUS_TWO)
            .find(|&b| self.pointer[node as usize] < self.offsets[b])
            .unwrap()
            - 1;
        self.total_weight -= 1 << (old_bucket - 1);

        self.pointer[self.buckets[self.offsets[old_bucket]] as usize] = self.pointer[node as usize];
        self.buckets
            .swap(self.pointer[node as usize], self.offsets[old_bucket]);
        for bucket in (1..old_bucket).rev() {
            self.pointer[self.buckets[self.offsets[bucket]] as usize] = self.offsets[bucket + 1];
            self.buckets
                .swap(self.offsets[bucket], self.offsets[bucket + 1]);
            self.offsets[bucket + 1] += 1;
        }
        self.pointer[node as usize] = self.offsets[1];
        self.offsets[1] += 1;
    }

    /// Returns *true* if there are no elements in the sampler
    pub fn is_empty(&self) -> bool {
        self.offsets[1] == self.buckets.len()
    }

    /// Returns *true* if the node is currently in the sampler
    pub fn is_in_sampler(&self, u: Node) -> bool {
        self.pointer[u as usize] >= self.offsets[1]
    }
}

/// Sample a node according to the given weights from the sampler
///
/// As we use this sampler in an algorithm where we expect the vast majority of elements to be in
/// buckets[1] (and buckets[2]), we iterate bottom-up and not top-down, despite bigger weights in
/// higher buckets.
impl<const NUM_BUCKETS_PLUS_TWO: usize> Distribution<Node>
    for WeightedPow2Sampler<NUM_BUCKETS_PLUS_TWO>
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Node {
        let mut rval = rng.gen_range(0..self.total_weight);
        for i in 1..(NUM_BUCKETS_PLUS_TWO - 1) {
            let weight = (1 << (i - 1)) * (self.offsets[i + 1] - self.offsets[i]);
            if weight > rval {
                return self.buckets[self.offsets[i] + (rval >> (i - 1))];
            } else {
                rval -= weight;
            }
        }
        panic!("The total weight is larger than the stored weight!");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_add_remove_no_panic() {
        let mut sampler = WeightedPow2Sampler::<5>::new(5);

        assert!(sampler.is_empty());

        sampler.add_entry(2, 1);
        sampler.add_entry(3, 2);
        sampler.add_entry(4, 0);
        sampler.add_entry(0, 0);

        assert!(sampler.is_in_sampler(3));

        sampler.set_bucket(4, 1);
        sampler.remove_entry(3);
        sampler.remove_entry(2);
        sampler.remove_entry(0);

        assert_eq!(sampler.sample(&mut rand::thread_rng()), 4);
        sampler.remove_entry(4);

        assert!(sampler.is_empty());
    }

    #[test]
    fn statistical_tests() {
        let rng = &mut rand::thread_rng();

        stat_test::<_, 5>(rng);
        stat_test::<_, 7>(rng);
        stat_test::<_, 9>(rng);
    }

    fn stat_test<R: Rng, const SIZE: usize>(rng: &mut R) {
        let mut sampler = WeightedPow2Sampler::<SIZE>::new(100 * SIZE - 200);

        const NUM_SAMPLES: usize = 100000;

        for i in 0..(100 * (SIZE - 2)) {
            sampler.add_entry(i as Node, i % (SIZE - 2));
        }

        let mut occ = [0; SIZE];
        for _ in 0..((SIZE - 1) * NUM_SAMPLES) {
            occ[(sampler.sample(rng) as usize) % (SIZE - 2)] += 1;
        }

        let frac: Vec<f64> = occ
            .into_iter()
            .take(SIZE - 2)
            .map(|x| (x as f64) / (((SIZE - 1) * NUM_SAMPLES) as f64))
            .collect();

        const DEVIATION: f64 = 0.1;
        for i in 2..frac.len() {
            assert!(
                ((2.0f64 - DEVIATION)..=(2.0f64 + DEVIATION)).contains(&(frac[i] / frac[i - 1])),
                "{}",
                frac[i] / frac[i - 1]
            );
        }
    }
}
