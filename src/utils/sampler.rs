use rand::Rng;
use rand_distr::Distribution;

use crate::graph::{Node, NumNodes};

/// Allows fast sampling of nodes where each node has a weight that is a power of 2
///
/// NUM_BUCKETS is the highest power of 2 (minus one) that is allowed
pub struct WeightedPow2Sampler<const NUM_BUCKETS: usize> {
    /// in bucket[i] are all nodes of weight 2^i
    buckets: [Vec<Node>; NUM_BUCKETS],
    /// Pointer for each possible node in which bucket (and position inside the bucket) it is
    pointer: (Vec<u8>, Vec<NumNodes>),
    /// Total weight of all inserted nodes
    total_weight: usize,
}

impl<const NUM_BUCKETS: usize> WeightedPow2Sampler<NUM_BUCKETS> {
    /// Creates a new sampler for a maximum of n nodes
    pub fn new(n: usize) -> Self {
        debug_assert!(NUM_BUCKETS < u8::MAX as usize);
        Self {
            buckets: array_init::array_init(|_| Vec::new()),
            pointer: (vec![u8::MAX; n], vec![0; n]),
            total_weight: 0,
        }
    }

    /// Adds a node with weight bucket
    pub fn add_entry(&mut self, node: Node, mut bucket: usize) {
        debug_assert!(self.pointer.0[node as usize] == u8::MAX);
        if bucket >= NUM_BUCKETS {
            bucket = NUM_BUCKETS - 1;
        }

        self.pointer.0[node as usize] = bucket as u8;
        self.pointer.1[node as usize] = self.buckets[bucket].len() as NumNodes;

        self.total_weight += 1 << bucket;
    }

    /// Removes a node from the sampler
    pub fn remove_entry(&mut self, node: Node) {
        let (bucket, position) = (self.pointer.0[node as usize], self.pointer.1[node as usize]);
        if bucket == u8::MAX {
            return;
        }

        self.pointer.0[node as usize] = u8::MAX;

        let bucket = bucket as usize;
        let position = position as usize;

        self.buckets[bucket].swap_remove(position);
        if self.buckets[bucket].len() > position {
            self.pointer.1[self.buckets[bucket][position] as usize] = position as NumNodes;
        }

        self.total_weight -= 1 << bucket;
    }

    /// Updates the weight of a node
    pub fn update_entry(&mut self, node: Node, new_bucket: usize) {
        if self.pointer.0[node as usize] != u8::MAX {
            self.remove_entry(node);
        }

        self.add_entry(node, new_bucket);
    }

    /// Is the sampler empty?
    pub fn is_empty(&self) -> bool {
        self.total_weight == 0
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
