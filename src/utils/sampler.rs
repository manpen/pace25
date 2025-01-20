use rand::Rng;
use rand_distr::Distribution;

use crate::graph::{Node, NumNodes};

pub struct WeightedPow2Sampler<const NUM_BUCKETS: usize> {
    buckets: [Vec<Node>; NUM_BUCKETS],
    pointer: Vec<(u8, NumNodes)>,
    total_weight: usize,
}

impl<const NUM_BUCKETS: usize> WeightedPow2Sampler<NUM_BUCKETS> {
    pub fn new(n: usize) -> Self {
        assert!(NUM_BUCKETS < u8::MAX as usize);
        Self {
            buckets: array_init::array_init(|_| Vec::new()),
            pointer: vec![(u8::MAX, 0); n],
            total_weight: 0,
        }
    }

    pub fn add_entry(&mut self, node: Node, mut bucket: usize) {
        assert!(self.pointer[node as usize].0 == u8::MAX);
        if bucket >= NUM_BUCKETS {
            bucket = NUM_BUCKETS - 1;
        }

        self.buckets[bucket].push(node);
        self.pointer[node as usize] = (bucket as u8, (self.buckets[bucket].len() - 1) as NumNodes);
        self.total_weight += 1 << bucket;
    }

    pub fn remove_entry(&mut self, node: Node) {
        let (bucket, position) = self.pointer[node as usize];
        if bucket == u8::MAX {
            return;
        }

        self.pointer[node as usize].0 = u8::MAX;

        let bucket = bucket as usize;
        let position = position as usize;

        self.buckets[bucket].swap_remove(position);
        if self.buckets[bucket].len() > position {
            self.pointer[self.buckets[bucket][position] as usize].1 = position as NumNodes;
        }

        self.total_weight -= 1 << bucket;
    }

    pub fn update_entry(&mut self, node: Node, new_bucket: usize) {
        if self.pointer[node as usize].0 != u8::MAX {
            self.remove_entry(node);
        }

        self.add_entry(node, new_bucket);
    }

    pub fn promote_entry(&mut self, node: Node) {
        let nbucket = (NUM_BUCKETS - 1).min(self.pointer[node as usize].0 as usize + 1);
        self.update_entry(node, nbucket);
    }

    pub fn is_empty(&self) -> bool {
        self.total_weight == 0
    }
}

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
