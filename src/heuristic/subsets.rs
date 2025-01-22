use crate::{graph::*, utils::DominatingSet};

/// # Subset-Rule
/// A node is subset-dominated by another node, if its neighborhood is a subset of the others neighborhood.
/// Here, we only consider the restricted neighborhoods in which nodes that are always covered by fixed nodes
/// are left out of consideration.
///
/// Returns a reduced edge lists and offsets that does not contain dominated nodes
pub fn subset_reduction(
    graph: &(impl AdjacencyList + CsrEdgeList + SelfLoop),
    sol: &mut DominatingSet,
) -> (Vec<Node>, Vec<NumEdges>) {
    let n = graph.number_of_nodes();
    let mut is_subset_dominated = BitSet::new(n);

    // Compute permanently covered nodes and degrees
    let mut non_perm_degree: Vec<NumNodes> = (0..n).map(|u| graph.degree_of(u)).collect();
    let mut is_perm_covered = BitSet::new(n);
    for u in sol.iter_fixed() {
        for v in graph.neighbors_of(u) {
            if !is_perm_covered.get_bit(v) {
                is_perm_covered.set_bit(v);
                for w in graph.neighbors_of(v) {
                    non_perm_degree[w as usize] -= 1;
                }
            }
        }
    }

    let (mut copied_edges, mut copied_offsets) = graph.get_csr_edges();

    macro_rules! neighbors {
        ($node:expr) => {
            copied_edges[(copied_offsets[$node as usize] as usize)
                ..(copied_offsets[$node as usize + 1] as usize)]
        };
    }

    // Sort adjacency lists to allow binary searching later on
    for u in 0..n {
        neighbors!(u).sort_unstable();
    }

    let mut candidates = Vec::new();
    let mut offsets = Vec::new();
    for u in 0..n {
        if is_subset_dominated.get_bit(u) || sol.is_fixed_node(u) {
            continue;
        }

        // If every neighbor (including u) is permanently covered, skip this node
        // Otherwise pick node with maximum number of non-permanently covered neighbors
        let min_neighbor = graph
            .neighbors_of(u)
            .filter(|v| !is_perm_covered.get_bit(*v))
            .min_by_key(|v| non_perm_degree[*v as usize]);
        if let Some(node) = min_neighbor {
            for v in &neighbors!(node) {
                if *v == u {
                    continue;
                }

                candidates.push(*v);
                offsets.push(0);
            }
        } else {
            continue;
        }

        // Only consider candidates that are adjacent to all non-permanently covered neighbors of u
        for v in &neighbors!(u) {
            if is_perm_covered.get_bit(*v) {
                continue;
            }

            for i in (0..candidates.len()).rev() {
                let candidate = candidates[i];
                if let Ok(index) = neighbors!(candidate)[offsets[i]..].binary_search(v) {
                    // Since edge-lists are sorted, v is increasing and we can use offsets[i] to
                    // allow for faster binary searches in later iterations
                    offsets[i] += index;
                } else {
                    candidates.swap_remove(i);
                    offsets.swap_remove(i);
                }
            }
        }

        // Consume candidates to find a dominating node of u
        for candidate in candidates.drain(..) {
            // Since possibly min_neighbor = Some(u), we always have
            // non_perm_degree[candidate as usize] <= non_perm_degree[u as usize]
            //
            // Inequality thus implies that the right side is bigger
            if non_perm_degree[candidate as usize] == non_perm_degree[u as usize] {
                if u < candidate {
                    is_subset_dominated.set_bit(u);
                } else {
                    is_subset_dominated.set_bit(candidate);

                    if sol.is_in_domset(candidate) && !is_subset_dominated.get_bit(u) {
                        sol.replace(candidate, u);
                    }
                    continue;
                }
            } else {
                is_subset_dominated.set_bit(u);
            }

            if sol.is_in_domset(u) && !is_subset_dominated.get_bit(candidate) {
                sol.replace(u, candidate);
            }
        }
        offsets.clear();
    }

    // Create a copy of the edge list that only contains nodes that are not subset-dominated
    let mut write_ptr = 0;
    let mut read_ptr = 0;
    for i in 0..(n as usize) {
        copied_offsets[i] = write_ptr as NumEdges;
        let offset = copied_offsets[i + 1];

        while read_ptr < offset as usize {
            if !is_subset_dominated.get_bit(copied_edges[read_ptr]) {
                copied_edges[write_ptr] = copied_edges[read_ptr];
                write_ptr += 1;
            }

            read_ptr += 1;
        }
    }
    copied_edges.truncate(write_ptr);
    copied_offsets[n as usize] = write_ptr as NumEdges;

    (copied_edges, copied_offsets)
}
