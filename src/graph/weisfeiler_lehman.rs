use super::*;

use fasthash::Murmur3HasherExt;
use std::hash::{Hash, Hasher};

pub trait WeisfeilerLehman {
    fn compute_weisfeiler_lehman(&self) -> WeisfeilerLehmanScoring;
}

impl<G: ColoredAdjacencyList> WeisfeilerLehman for G {
    fn compute_weisfeiler_lehman(&self) -> WeisfeilerLehmanScoring {
        WeisfeilerLehmanScoring::new(self)
    }
}

pub struct WeisfeilerLehmanScoring {
    scores: Vec<u64>,
    scratch: Vec<u64>,
}

impl WeisfeilerLehmanScoring {
    pub fn new<G: ColoredAdjacencyList>(graph: &G) -> Self {
        let mut wl = WeisfeilerLehmanScoring {
            scores: Vec::from_iter(graph.vertices().map(|u| {
                graph.degree_of(u) as u64 + murmur64((graph.red_degree_of(u) as u64) << 32)
            })),

            scratch: vec![0; graph.number_of_nodes() as usize],
        };

        wl.compute(graph);

        wl
    }

    fn compute<G: ColoredAdjacencyList>(&mut self, graph: &G) {
        for _ in graph.vertices() {
            // invariant: at this point `self.scores` contains the hashes
            // from the last iteration; we temporarily store the new
            // results in `self.scratch` and then swap both

            let mut scratch_neighbors = Vec::new();
            for (u, score) in self.scratch.iter_mut().enumerate() {
                let u = u as Node;

                scratch_neighbors.clear();

                scratch_neighbors
                    .extend(graph.black_neighbors_of(u).map(|u| self.scores[u as usize]));

                scratch_neighbors.extend(
                    graph
                        .red_neighbors_of(u)
                        .map(|u| murmur64(self.scores[u as usize])),
                );

                scratch_neighbors.sort();

                let mut s: Murmur3HasherExt = Default::default();
                scratch_neighbors.hash(&mut s);
                *score = s.finish();
            }

            self.scores.copy_from_slice(&self.scratch);

            // scores and scratch have the same contents
            self.scratch.sort();
            if self.scratch.iter().tuple_windows().all(|(x, y)| x != y) {
                break;
            }
        }
    }

    pub fn get_mapper(&self) -> RankingForwardMapper {
        let mut tmp = self
            .scores
            .iter()
            .enumerate()
            .map(|(i, score)| (*score, i as Node))
            .collect_vec();
        tmp.sort();
        let mut ranking = vec![0; tmp.len()];
        for (j, (_, i)) in tmp.iter().enumerate() {
            ranking[*i as usize] = j as Node;
        }
        RankingForwardMapper::from_vec(ranking)
    }
}

impl GraphDigest for WeisfeilerLehmanScoring {
    fn binary_digest<D: digest::Digest>(&self) -> digest::Output<D> {
        let mut hasher = D::new();

        let slice = unsafe {
            std::slice::from_raw_parts(
                self.scores.as_ptr().cast() as *const u8,
                self.scores.len() * std::mem::size_of::<u64>(),
            )
        };

        hasher.update(slice);

        hasher.finalize()
    }
}

fn murmur64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    h
}
