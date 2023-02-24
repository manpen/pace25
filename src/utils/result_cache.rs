use crate::prelude::*;

use fxhash::{FxBuildHasher, FxHashMap};

const INITIAL_CAPACITY: usize = 10_000;
const DEFAULT_MAX_CAPACITY: usize = 100_000_000;
const EVICITION_SEACH: usize = 100;

pub type Solution = (NumNodes, ContractionSequence);

struct CacheEntry {
    timestamp: u64,
    slack: NumNodes,
    not_above: NumNodes,
    solution: Option<Solution>,
}

pub struct ResultCache<K> {
    cache: FxHashMap<K, CacheEntry>,
    capacity: usize,
    timestamp: u64,
    number_of_misses: u64,
    number_of_accesses: u64,
}

impl<K> Default for ResultCache<K>
where
    K: std::cmp::Eq + std::hash::Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K> ResultCache<K>
where
    K: std::cmp::Eq + std::hash::Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::with_capacity_and_hasher(INITIAL_CAPACITY, FxBuildHasher::default()),
            capacity: DEFAULT_MAX_CAPACITY,
            timestamp: 0,
            number_of_misses: 0,
            number_of_accesses: 0,
        }
    }

    /// Sets new capacity of the cache without reserving actual memory.
    ///
    /// # Warning
    /// Erases cache if the current number of elements exceeds the new cache size
    pub fn set_capacity(&mut self, capacity: usize) {
        if self.cache.len() > capacity {
            self.cache = FxHashMap::with_capacity_and_hasher(capacity + 1, FxBuildHasher::default())
        }
        self.capacity = capacity;
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn number_of_cache_hits(&self) -> u64 {
        self.number_of_accesses - self.number_of_misses
    }

    pub fn number_of_cache_misses(&self) -> u64 {
        self.number_of_misses
    }

    /// Introduces new result into cache
    pub fn add_to_cache(
        &mut self,
        digest: K,
        solution: Option<Solution>,
        slack: NumNodes,
        not_above: NumNodes,
    ) {
        if self.cache.len() > self.capacity {
            self.evict_element()
        }
        self.cache.insert(
            digest,
            CacheEntry {
                timestamp: self.timestamp,
                slack,
                not_above,
                solution,
            },
        );
        self.timestamp += 1;
    }

    /// Looks up result from cache. We need to pass the upper_bound, since a previous entry
    /// obtained for a lower upper bound might be `None` and apply anymore.
    pub fn get(
        &mut self,
        digest: &K,
        slack: NumNodes,
        not_above: NumNodes,
    ) -> Option<Option<&Solution>> {
        self.number_of_accesses += 1;
        if let Some(entry) = self.cache.get_mut(digest) {
            entry.timestamp = self.timestamp;

            if let Some(result) = entry.solution.as_ref() {
                let tww = result.0;

                if tww <= slack || entry.slack <= slack || tww > entry.slack {
                    return Some((tww <= not_above).then_some(result));
                }
            } else if not_above <= entry.not_above {
                return Some(None);
            }
        }

        self.number_of_misses += 1;
        None
    }

    fn evict_element(&mut self) {
        if self.cache.is_empty() {
            return;
        }

        // we assume that the iteration order of the hash table is random
        // then we do not have a considerable bias from only evicting from the beginning
        let key_to_evict = self
            .cache
            .iter()
            .take(EVICITION_SEACH)
            .min_by_key(|(_, entry)| entry.timestamp)
            .unwrap()
            .0
            .clone();
        self.cache.remove(&key_to_evict);
    }
}
