use std::cmp::Ordering;

use rand::Rng;
use rand_distr::Distribution;
use thiserror::Error;

use crate::{errors::InvariantCheck, graph::Node};

/// Allows fast sampling of nodes where each node has a weight that is a power of 2.
/// Nodes are sorted into buckets where nodes u with Bucket[u] = i have a weight of 2^(i - 1) with
/// base weight 0 for nodes in bucket 0.
pub struct WeightedPow2Sampler<const NUM_BUCKETS_PLUS_TWO: usize> {
    /// List of all nodes partitioned by buckets
    buckets: Vec<Node>,
    /// Pointer for each possible node in which bucket (and position inside the bucket) it is
    pointer: Vec<usize>,

    /// We need two additional Offset-Entries: one for Bucket 0 and one to allow branchless
    /// accessing a slice of the last bucket.
    ///
    /// The first entry is thus always 0, and the last is always buckets.len()
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

    /// Computes the weight of a node in a specific bucket.
    ///
    /// As we want weight(0) = 0, we first shift by bucket bits to the left, followed by a right
    /// shift of 1 as this is branchless.
    #[inline(always)]
    fn weight(bucket: usize) -> usize {
        (1 << bucket) >> 1
    }

    /// Returns the bucket in which `pos` lies in
    #[inline(always)]
    fn bucket(&self, pos: usize) -> usize {
        (1..NUM_BUCKETS_PLUS_TWO)
            .find(|&b| pos < self.offsets[b])
            .unwrap()
            - 1
    }

    /// Returns the bucket of node
    #[allow(unused)]
    pub fn bucket_of_node(&self, u: Node) -> usize {
        self.bucket(self.pointer[u as usize])
    }

    /// Panics if the stored total weight does not match the assigned buckets
    #[allow(unused)]
    pub fn assert_total_weight(&self) {
        assert_eq!(
            self.total_weight,
            (1..(NUM_BUCKETS_PLUS_TWO - 1))
                .map(|b| { ((self.offsets[b + 1] - self.offsets[b]) << b) >> 1 })
                .sum()
        );
    }

    /// Panics if any node stores a wrong position
    #[allow(unused)]
    pub fn assert_positions(&self) {
        for u in 0..self.buckets.len() {
            assert_eq!(self.buckets[self.pointer[u]], u as Node);
        }
    }

    /// Directly sets the bucket for a given node
    pub fn set_bucket(&mut self, node: Node, mut new_bucket: usize) {
        let pos = self.pointer[node as usize];

        new_bucket = new_bucket.min(NUM_BUCKETS_PLUS_TWO - 2);
        let old_bucket = self.bucket(pos);

        // Set weight
        self.total_weight += Self::weight(new_bucket);
        self.total_weight -= Self::weight(old_bucket);

        // Update buckets
        match old_bucket.cmp(&new_bucket) {
            // General version of `self.remove_entry`
            //
            // Move element down by swapping it with the first element in its current bucket and
            // updating the offsets recursively.
            //
            // Instead of swapping directly, we copy elements up, and write the original value only
            // once at the end.
            Ordering::Greater => {
                // Update old_bucket
                self.pointer[self.buckets[self.offsets[old_bucket]] as usize] = pos;
                self.buckets[pos] = self.buckets[self.offsets[old_bucket]];

                // Make first bucket element into last recursively
                for bucket in ((new_bucket + 1)..old_bucket).rev() {
                    self.pointer[self.buckets[self.offsets[bucket]] as usize] =
                        self.offsets[bucket + 1];
                    self.buckets[self.offsets[bucket + 1]] = self.buckets[self.offsets[bucket]];
                    self.offsets[bucket + 1] += 1;
                }

                // Insert original node into final position
                self.buckets[self.offsets[new_bucket + 1]] = node;
                self.pointer[node as usize] = self.offsets[new_bucket + 1];
                self.offsets[new_bucket + 1] += 1;
            }
            // General version of `self.add_entry`
            //
            // Move element up by swapping it with the last element in its current bucket and
            // updating the offsets recursively.
            //
            // Instead of swapping directly, we copy elements down, and write the original value only
            // once at the end.
            Ordering::Less => {
                // Update old_bucket
                self.pointer[self.buckets[self.offsets[old_bucket + 1] - 1] as usize] = pos;
                self.buckets[pos] = self.buckets[self.offsets[old_bucket + 1] - 1];

                // Make last element in bucket into first recursively
                for bucket in (old_bucket + 1)..new_bucket {
                    self.pointer[self.buckets[self.offsets[bucket + 1] - 1] as usize] =
                        self.offsets[bucket] - 1;
                    self.buckets[self.offsets[bucket] - 1] =
                        self.buckets[self.offsets[bucket + 1] - 1];
                    self.offsets[bucket] -= 1;
                }

                // Insert original node into final position
                self.offsets[new_bucket] -= 1;
                self.buckets[self.offsets[new_bucket]] = node;
                self.pointer[node as usize] = self.offsets[new_bucket];
            }
            _ => {}
        };
    }

    /// Moves node into the new_bucket assuming that it was previously in bucket 0
    pub fn add_entry(&mut self, node: Node, mut new_bucket: usize) {
        debug_assert_eq!(self.bucket(self.pointer[node as usize]), 0);
        debug_assert_ne!(new_bucket, 0);

        new_bucket = new_bucket.min(NUM_BUCKETS_PLUS_TWO - 1);

        // Set weight
        self.total_weight += Self::weight(new_bucket);

        // Update old_bucket
        self.pointer[self.buckets[self.offsets[1] - 1] as usize] = self.pointer[node as usize];
        self.buckets[self.pointer[node as usize]] = self.buckets[self.offsets[1] - 1];

        // Make last element in bucket into first recursively
        for bucket in 1..new_bucket {
            self.pointer[self.buckets[self.offsets[bucket + 1] - 1] as usize] =
                self.offsets[bucket] - 1;
            self.buckets[self.offsets[bucket] - 1] = self.buckets[self.offsets[bucket + 1] - 1];
            self.offsets[bucket] -= 1;
        }

        // Insert original node into final position
        self.offsets[new_bucket] -= 1;
        self.buckets[self.offsets[new_bucket]] = node;
        self.pointer[node as usize] = self.offsets[new_bucket];
    }

    /// Moves the node into bucket 0
    pub fn remove_entry(&mut self, node: Node) {
        debug_assert!(self.bucket(self.pointer[node as usize]) > 0);
        let old_bucket = self.bucket(self.pointer[node as usize]);

        // Set weight
        self.total_weight -= Self::weight(old_bucket);

        // Update old_bucket
        self.pointer[self.buckets[self.offsets[old_bucket]] as usize] = self.pointer[node as usize];
        self.buckets[self.pointer[node as usize]] = self.buckets[self.offsets[old_bucket]];

        // Make first bucket element into last recursively
        for bucket in (1..old_bucket).rev() {
            if !self.is_bucket_empty(bucket) {
                self.pointer[self.buckets[self.offsets[bucket]] as usize] =
                    self.offsets[bucket + 1];
            }
            self.buckets[self.offsets[bucket + 1]] = self.buckets[self.offsets[bucket]];
            self.offsets[bucket + 1] += 1;
        }

        // Insert original node into final position
        self.buckets[self.offsets[1]] = node;
        self.pointer[node as usize] = self.offsets[1];
        self.offsets[1] += 1;
    }

    /// Returns *true* if there are no elements in the sampler
    pub fn is_empty(&self) -> bool {
        self.offsets[1] == self.buckets.len()
    }

    pub fn is_bucket_empty(&self, bucket: usize) -> bool {
        self.offsets[bucket] == self.offsets[bucket + 1]
    }

    /// Returns *true* if the node is currently in the sampler
    pub fn is_in_sampler(&self, u: Node) -> bool {
        self.pointer[u as usize] >= self.offsets[1]
    }

    /// Samples multiple elements of increasing size (ie. rejecting elements in lower buckets once
    /// a higher bucket element has been sampled). Passes this information along to a given
    /// predicate.
    ///
    /// Useful when looking to sample elements with highest bucket as well as another external
    /// tiebreaker.
    pub fn sample_many<F: FnMut(usize, Node), const NUM_SAMPLES: usize>(
        &self,
        rng: &mut impl Rng,
        mut cb: F,
    ) {
        let mut reject_below = 0;
        'outer: for _ in 0..NUM_SAMPLES {
            let mut rval = rng.gen_range(0..self.total_weight);
            if rval < reject_below {
                continue;
            }

            reject_below = 0;
            for i in 1..(NUM_BUCKETS_PLUS_TWO - 1) {
                let weight = (self.offsets[i + 1] - self.offsets[i]) << (i - 1);
                if weight > rval {
                    let node = self.buckets[self.offsets[i] + (rval >> (i - 1))];
                    cb(i, node);
                    continue 'outer;
                } else {
                    rval -= weight;
                    reject_below += weight;
                }
            }
            unreachable!("The total weight is larger than the stored weight!");
        }
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
            let weight = Self::weight(i) * (self.offsets[i + 1] - self.offsets[i]);
            if weight > rval {
                return self.buckets[self.offsets[i] + (rval >> (i - 1))];
            } else {
                rval -= weight;
            }
        }
        panic!("The total weight is larger than the stored weight!");
    }
}

/// Error type for the sampler invariants
#[derive(Copy, Clone, Debug, Error)]
pub enum SamplerError {
    #[error("{0} is in position {2} but stores {1}")]
    WrongPosition(Node, usize, usize),
    #[error("total weight of sampler should be {1}, not {0}")]
    WrongTotalWeight(usize, usize),
}

impl<const NUM_BUCKETS_PLUS_TWO: usize> InvariantCheck<SamplerError>
    for WeightedPow2Sampler<NUM_BUCKETS_PLUS_TWO>
{
    fn is_correct(&self) -> Result<(), SamplerError> {
        let correct_weight = (1..(NUM_BUCKETS_PLUS_TWO - 1))
            .map(|b| ((self.offsets[b + 1] - self.offsets[b]) << b) >> 1)
            .sum();
        if self.total_weight != correct_weight {
            return Err(SamplerError::WrongTotalWeight(
                self.total_weight,
                correct_weight,
            ));
        }

        for u in 0..self.buckets.len() {
            if u != self.pointer[self.buckets[u] as usize] {
                return Err(SamplerError::WrongPosition(
                    self.buckets[u],
                    self.pointer[self.buckets[u] as usize],
                    u,
                ));
            }
        }

        Ok(())
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
        sampler.add_entry(4, 3);
        sampler.add_entry(0, 3);

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
            sampler.set_bucket(i as Node, i % (SIZE - 2));
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
