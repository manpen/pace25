use crate::graph::*;
use rand::Rng;
use rand_distr::Geometric;

pub trait GnpGenerator: Sized {
    /// Generates a Gilbert (also, wrongly, known as Erdos-Reyni) graph
    /// The `G(n,p)` contains n nodes and each of the `n(n-1)/2` edges exists
    /// independently with probability `p`. Each edge is black independently
    /// with probability `prob_black` and red otherwise
    fn random_colored_gnp<R: Rng>(rng: &mut R, n: Node, p: f64, prob_black: f64) -> Self;

    fn random_black_gnp<R: Rng>(rng: &mut R, n: Node, p: f64) -> Self {
        Self::random_colored_gnp(rng, n, p, 1.0)
    }
}

impl<G> GnpGenerator for G
where
    G: GraphNew + GraphEdgeEditing,
{
    fn random_colored_gnp<R: Rng>(rng: &mut R, n: Node, p: f64, _prob_black: f64) -> Self
    where
        G: GraphNew + GraphEdgeEditing,
    {
        let mut result = Self::new(n);

        // indirection via vector as we need a &mut for rng and edge color also needs rng
        let edges: Vec<_> = BernoulliSamplingRange::new(rng, 0, (n as i64) * (n as i64), p)
            .filter_map(|x| {
                let u = x / (n as i64);
                let v = x % (n as i64);
                (u < v).then_some((u as Node, v as Node))
            })
            .collect();

        for (u, v) in edges {
            result.add_edge(u, v);
        }

        result
    }
}

/// Provides an iterator similarly to Range, but
/// includes each element i.i.d. with probability of p
pub struct BernoulliSamplingRange<'a, R: Rng> {
    current: i64,
    end: i64,
    distr: Geometric,
    rng: &'a mut R,
}

impl<'a, R: Rng> BernoulliSamplingRange<'a, R> {
    pub fn new(rng: &'a mut R, begin: i64, end: i64, prob: f64) -> Self {
        debug_assert!(begin <= end);
        debug_assert!((0.0..=1.0).contains(&prob));
        Self {
            rng,
            current: begin - 1,
            end,
            distr: Geometric::new(prob).unwrap(),
        }
    }

    fn try_advance(&mut self) {
        if self.current >= self.end {
            return;
        }

        let skip = self.rng.sample(self.distr);
        if skip > i64::MAX as u64 {
            self.current = self.end;
        } else {
            self.current += 1;
            self.current = match self.current.checked_add(skip as i64) {
                Some(x) => x,
                None => self.end,
            }
        }
    }
}

impl<R: Rng> Iterator for BernoulliSamplingRange<'_, R> {
    type Item = i64;
    fn next(&mut self) -> Option<Self::Item> {
        self.try_advance();

        if self.current >= self.end {
            None
        } else {
            Some(self.current)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bernoulli_range() {
        let rng = &mut rand::thread_rng();

        // empty range
        assert_eq!(BernoulliSamplingRange::new(rng, 0, 0, 1.0).count(), 0);

        // p=1
        assert_eq!(BernoulliSamplingRange::new(rng, 0, 10, 1.0).count(), 10);

        // p=0
        assert_eq!(BernoulliSamplingRange::new(rng, 0, 100, 0.0).count(), 0);

        // test that we see each element ~p*n times
        let min = 3;
        let max = 100;
        let mut counts = vec![0; max as usize];
        for _ in 0..1000 {
            let b = BernoulliSamplingRange::new(rng, min, max, 0.25);
            for x in b {
                assert!(min <= x);
                assert!(x < max);
                counts[x as usize] += 1;
            }
        }

        assert!(counts.iter().enumerate().all(|(i, &c)| {
            if i < min as usize {
                c == 0
            } else {
                (150..350).contains(&c)
            }
        }));
    }

    #[test]
    fn test_gnp() {
        let rng = &mut rand::thread_rng();

        // generate multiple graphs of various densities and verify that the
        // expected number of edges is close to the expected value
        for p in [0.001, 0.01, 0.1] {
            let repeats = 100;
            let n = 100;

            let mean_edges = (0..repeats)
                .map(|_| AdjArray::random_black_gnp(rng, n, p).number_of_edges() as f64)
                .sum::<f64>()
                / repeats as f64;

            let expected = p * (n as f64) * ((n - 1) as f64) / 2.0;

            assert!((0.75 * expected..1.25 * expected).contains(&mean_edges));
        }
    }
}
