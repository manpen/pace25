use crate::{graph::*, utils::DominatingSet};

use super::KernelizationRule;

/// # Subset-Rule
/// A node is subset-dominated by another node, if its neighborhood is a subset of the others neighborhood.
/// Here, we only consider the restricted neighborhoods in which nodes that are always covered by fixed nodes
/// are left out of consideration.
pub struct SubsetRule;

impl KernelizationRule<&mut CsrEdges> for SubsetRule {
    fn apply_rule(graph: &mut CsrEdges, sol: &mut DominatingSet) -> BitSet {
        let n = graph.number_of_nodes();
        let mut is_subset_dominated = graph.vertex_bitset_unset();

        // Compute permanently covered nodes and degrees
        let mut non_perm_degree: Vec<NumNodes> = (0..n).map(|u| graph.degree_of(u)).collect();
        let mut is_perm_covered = graph.vertex_bitset_unset();
        for u in sol.iter_fixed() {
            for &v in &graph[u] {
                if !is_perm_covered.set_bit(v) {
                    for &w in &graph[v] {
                        non_perm_degree[w as usize] -= 1;
                    }
                }
            }
        }

        // Sort adjacency lists to allow binary searching later on
        for u in 0..n {
            graph[u].sort_unstable();
        }

        let mut candidates = Vec::new();
        let mut offsets = Vec::new();
        for u in 0..n {
            if is_subset_dominated.get_bit(u) || sol.is_fixed_node(u) {
                continue;
            }

            // If every neighbor (including u) is permanently covered, skip this node
            // Otherwise pick node with maximum number of non-permanently covered neighbors
            if let Some(node) = graph[u]
                .iter()
                .filter(|&&v| !is_perm_covered.get_bit(v))
                .min_by_key(|&&v| non_perm_degree[v as usize])
            {
                for &v in &graph[*node] {
                    if v == u {
                        continue;
                    }

                    candidates.push(v);
                    offsets.push(0);
                }
            } else {
                continue;
            }

            // Only consider candidates that are adjacent to all non-permanently covered neighbors of u
            for &v in &graph[u] {
                if is_perm_covered.get_bit(v) {
                    continue;
                }

                for i in (0..candidates.len()).rev() {
                    let candidate = candidates[i];
                    if let Ok(index) = &graph[candidate][offsets[i]..].binary_search(&v) {
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

        is_subset_dominated
    }
}
