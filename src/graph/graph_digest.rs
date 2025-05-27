use super::*;
use ::digest::Digest;

pub trait GraphDigest {
    /// Computes a Hash-Digest of a graph that is independent of the
    /// graph data structure used and returns it as a hex string.
    fn digest<D: Digest>(&self) -> String
    where
        digest::Output<D>: core::fmt::LowerHex,
    {
        format!("{:x}", self.binary_digest::<D>())
    }

    // Computes the Hash-Digest of a graph that is indepenet of the graph
    // data structure ised and returns it as binary data.
    fn binary_digest<D: Digest>(&self) -> digest::Output<D>;

    /// Computes a SHA256 digest using [GraphDigest::digest]. The returned string is 64 characters long.
    fn digest_sha256(&self) -> String {
        self.digest::<sha2::Sha256>()
    }

    /// Computes a SHA256 digest using [GraphDigest::digest] and returns the binary digest.
    fn binary_digest_sha256(&self) -> digest::Output<sha2::Sha256> {
        self.binary_digest::<sha2::Sha256>()
    }
}

impl GraphDigest for AdjArray {
    fn binary_digest<D: Digest>(&self) -> digest::Output<D> {
        let mut hasher = D::new();
        let mut buffer = [0u8; 12];

        let encode = |buf: &mut [u8], u: Node| {
            for (i, c) in buf.iter_mut().enumerate().take(4) {
                *c = (u >> (8 * i)) as u8;
            }
        };

        // first encode the number of nodes in the graph
        encode(&mut buffer[0..4], self.number_of_nodes());
        hasher.update(buffer);

        // then append a sorted edge list
        for ColoredEdge(u, v, col) in self.ordered_colored_edges(true) {
            encode(&mut buffer[0..], u);
            encode(&mut buffer[4..], v);
            encode(&mut buffer[8..], col as u32);
            hasher.update(buffer);
        }

        hasher.finalize()
    }
}

#[cfg(test)]
pub mod test {
    use crate::graph::{AdjArray, EdgeColor, GraphDigest, GraphEdgeEditing, GraphNew};

    #[test]
    fn digest_sha256() {
        let mut graph = AdjArray::new(10);
        graph.add_edge(4, 3, EdgeColor::Black);
        graph.add_edge(1, 2, EdgeColor::Red);
        // computed with https://www.gnu.org/software/coreutils/sha256sum
        assert_eq!(
            graph.digest_sha256(),
            "e941a4197b7ed2a797d7812b0e07e1e575f5f9f0c2963fbef365fc910fbcaf92"
        );
    }
}
