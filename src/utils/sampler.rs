use rand::Rng;
use rand_distr::Distribution;

use crate::graph::Node;

/// Allows fast sampling of nodes where each node has a weight that is a power of 2
///
/// NUM_BUCKETS is the highest power of 2 (minus one) that is allowed
pub struct WeightedPow2Sampler<const NUM_BUCKETS: usize> {
    /// in bucket[i] are all nodes of weight 2^i
    buckets: [Vec<Node>; NUM_BUCKETS],
    /// Pointer for each possible node in which bucket (and position inside the bucket) it is
    pointer: Vec<(usize, usize)>,
    /// Total weight of all inserted nodes
    total_weight: usize,
}

const NOT_SET: usize = u8::MAX as usize;

impl<const NUM_BUCKETS: usize> WeightedPow2Sampler<NUM_BUCKETS> {
    /// Creates a new sampler for a maximum of n nodes and a given number of buckets.
    ///
    /// There can be at most 255 buckets.
    pub fn new(n: usize) -> Self {
        debug_assert!(NUM_BUCKETS < u8::MAX as usize);
        Self {
            buckets: array_init::array_init(|_| Vec::new()),
            pointer: vec![(NOT_SET, NOT_SET); n],
            total_weight: 0,
        }
    }

    /// Adds a node to the given bucket.
    /// The node must not be part if the sampler at the moment.
    ///
    /// Checks for overflow and assigns all weights that are too big into the last bucket.  
    pub fn add_entry(&mut self, node: Node, mut bucket: usize) {
        debug_assert!(!self.is_in_sampler(node));
        if bucket >= NUM_BUCKETS {
            bucket = NUM_BUCKETS - 1;
        }

        self.pointer[node as usize] = (bucket, self.buckets[bucket].len());
        self.buckets[bucket].push(node);

        self.total_weight += 1 << bucket;
    }

    /// Removes a node from the sampler.
    /// Panics if node is not in the sampler.
    pub fn remove_entry(&mut self, node: Node) {
        debug_assert!(self.is_in_sampler(node));

        let (bucket, position) = self.pointer[node as usize];

        self.pointer[node as usize].0 = NOT_SET;

        self.buckets[bucket].swap_remove(position);
        if self.buckets[bucket].len() > position {
            self.pointer[self.buckets[bucket][position] as usize].1 = position;
        }

        self.total_weight -= 1 << bucket;
    }

    /// Updates the weight of a node.
    /// Panics if node is not in the sampler.
    pub fn update_entry(&mut self, node: Node, new_bucket: usize) {
        debug_assert!(self.is_in_sampler(node));

        self.remove_entry(node);
        self.add_entry(node, new_bucket);
    }

    /// Returns *true* if there are no elements in the sampler
    pub fn is_empty(&self) -> bool {
        self.total_weight == 0
    }

    /// Returns *true* if the node is currently in the sampler
    pub fn is_in_sampler(&self, u: Node) -> bool {
        self.pointer[u as usize].0 != NOT_SET
    }
}

/// Sample a node according to the given weights from the sampler
///
/// As we use this sampler in an algorithm where we expect the vast majority of elements to be in
/// buckets[0] (and buckets[1]), we iterate bottom-up and not top-down, despite bigger weights in
/// higher buckets.
impl<const NUM_BUCKETS: usize> Distribution<Node> for WeightedPow2Sampler<NUM_BUCKETS> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Node {
        let mut rval = rng.gen_range(0..self.total_weight);
        for (i, bucket) in self.buckets.iter().enumerate() {
            let weight = (1 << i) * bucket.len();
            if weight > rval {
                return bucket[rval >> i];
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
        let mut sampler = WeightedPow2Sampler::<3>::new(5);

        assert!(sampler.is_empty());

        sampler.add_entry(2, 1);
        sampler.add_entry(3, 2);
        sampler.add_entry(4, 0);
        sampler.add_entry(0, 0);

        assert!(sampler.is_in_sampler(3));

        sampler.update_entry(4, 1);
        sampler.remove_entry(3);
        sampler.remove_entry(2);
        sampler.remove_entry(0);

        assert_eq!(sampler.sample(&mut rand::thread_rng()), 4);
        sampler.remove_entry(4);

        assert!(sampler.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_illegal_sampler_op1() {
        let mut sampler = WeightedPow2Sampler::<3>::new(5);
        sampler.add_entry(10, 0);
    }

    #[test]
    #[should_panic]
    fn test_illegal_sampler_op2() {
        let mut sampler = WeightedPow2Sampler::<3>::new(5);
        sampler.add_entry(0, 0);
        sampler.add_entry(0, 2);
    }

    #[test]
    #[should_panic]
    fn test_illegal_sampler_op3() {
        let mut sampler = WeightedPow2Sampler::<3>::new(5);
        sampler.remove_entry(1);
    }

    #[test]
    fn statistical_tests() {
        let rng = &mut rand::thread_rng();

        stat_test::<_, 3>(rng);
        stat_test::<_, 5>(rng);
        stat_test::<_, 7>(rng);
    }

    fn stat_test<R: Rng, const SIZE: usize>(rng: &mut R) {
        let mut sampler = WeightedPow2Sampler::<SIZE>::new(100 * SIZE);

        const NUM_SAMPLES: usize = 100000;

        for i in 0..(100 * SIZE) {
            sampler.add_entry(i as Node, i % SIZE);
        }

        let mut occ = [0; SIZE];
        for _ in 0..(SIZE * NUM_SAMPLES) {
            occ[(sampler.sample(rng) as usize) % SIZE] += 1;
        }

        let frac: Vec<f64> = occ
            .into_iter()
            .map(|x| (x as f64) / ((SIZE * NUM_SAMPLES) as f64))
            .collect();

        const DEVIATION: f64 = 0.1;
        for i in 1..frac.len() {
            assert!(
                ((2.0f64 - DEVIATION)..=(2.0f64 + DEVIATION)).contains(&(frac[i] / frac[i - 1]))
            );
        }
    }
}
