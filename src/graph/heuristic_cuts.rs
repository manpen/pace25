use rand::Rng;

use super::*;

pub trait HeuristicCuts {
    fn compute_cuts(&self) -> Vec<Node>;
}

impl<G> HeuristicCuts for G
where
    G: AdjacencyList,
{
    fn compute_cuts(&self) -> Vec<Node> {
        HeuristicBalancedCut::new(self).compute()
    }
}

pub struct HeuristicBalancedCut<'a, T: AdjacencyList> {
    graph: &'a T,
    p: f64,
    threshold: usize,
}

impl<'a, T: AdjacencyList> HeuristicBalancedCut<'a, T> {
    pub fn new(graph: &'a T) -> Self {
        Self {
            graph,
            p: 0.25,
            threshold: (graph.number_of_nodes() / 2 * 3) as usize,
        }
    }

    pub fn compute(self) -> Vec<Node> {
        let mut rng = rand::thread_rng();
        let n = self.graph.number_of_nodes() as usize;

        let mut a: Vec<bool> = vec![false; n];
        let mut b: Vec<bool> = Default::default();
        let mut c: Vec<bool> = Default::default();

        for u in 0..self.graph.number_of_nodes() {
            if self.p < rng.r#gen() {
                if a.len() < self.threshold {
                    a[u as usize] = true;
                } else if b.len() < self.threshold {
                    b[u as usize] = true;
                } else {
                    c[u as usize] = true;
                }
            } else if b.len() < self.threshold {
                b[u as usize] = true;
            } else if a.len() < self.threshold {
                a[u as usize] = true;
            } else {
                c[u as usize] = true;
            }
        }

        while let Some((u, _)) = b.iter().enumerate().filter(|(_, b)| **b).find(|(u, _)| {
            for v in self.graph.neighbors_of(*u as Node) {
                if a[v as usize] {
                    return true;
                }
            }
            false
        }) {
            b[u] = false;
            c[u] = true;
        }
        b.iter()
            .enumerate()
            .filter(|(_, b)| **b)
            .map(|(u, _)| u as Node)
            .collect()
    }
}
