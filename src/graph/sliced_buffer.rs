use std::ops::{Index, IndexMut};

use super::*;

// Implements the core data structure of CSR (i.e. a data vector `buffer`
// and an non-decreasing index vector offsets, where the start of the
// i-th slice in `buffer` is stored at `offsets[i]`.
//
// This implementation verifies the following invariants at construction
// and to avoid repeated checks during accesses:
//  (0) `offset` has at least two elements
//  (1) `offset` is non-decreasing (i.e. produce a valid range) and
//  (2) `offset` stays within bounds of `buffer`
//
// The implementation is its own module to prevent the CSR data structure
// from manipulating the offsets vector, which may invalidate the aforementioned
// invariants.
#[derive(Debug, Clone)]
pub struct SlicedBuffer<T> {
    buffer: Vec<T>,
    offsets: Vec<NumEdges>,
}

impl<T> SlicedBuffer<T> {
    /// Constructs the SlicedBuffer and panics if one of the three
    /// invariants on offset are violated.
    pub fn new(buffer: Vec<T>, offsets: Vec<NumEdges>) -> Self {
        assert!(offsets.len() > 1);
        assert!(offsets.len() - 1 <= Node::MAX as usize);
        assert!(offsets.is_sorted());
        assert!(*offsets.last().unwrap() as usize <= buffer.len());

        Self { buffer, offsets }
    }

    #[inline(always)]
    pub fn number_of_nodes(&self) -> NumNodes {
        // Cannot underflow since `self.offset` has at least two entries
        unsafe { self.offsets.len().unchecked_sub(1) as NumNodes }
    }

    #[inline(always)]
    pub fn number_of_edges(&self) -> NumEdges {
        self.buffer.len() as NumEdges
    }

    #[inline(always)]
    pub fn degree_of(&self, u: Node) -> NumNodes {
        self.offsets[u as usize + 1] - self.offsets[u as usize]
    }

    #[inline(always)]
    pub fn raw_buffer_slice(&self) -> &[T] {
        &self.buffer
    }

    #[inline(always)]
    pub fn raw_offset_slice(&self) -> &[NumEdges] {
        &self.offsets
    }
}

impl<T> Index<Node> for SlicedBuffer<T> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, node: Node) -> &Self::Output {
        let end = self.offsets[node as usize + 1] as usize;
        let start = self.offsets[node as usize] as usize;

        // using unchecked here is safe, since we established in the
        // constructor that all entries within `self.offsets`` are
        //  (i) non-decreasing (i.e. produce a valid range) and
        //  (ii) are within bounds of `self.buffer`
        unsafe { self.buffer.get_unchecked(start..end) }
    }
}

impl<T> IndexMut<Node> for SlicedBuffer<T> {
    #[inline(always)]
    fn index_mut(&mut self, node: Node) -> &mut Self::Output {
        let end = self.offsets[node as usize + 1] as usize;
        let start = self.offsets[node as usize] as usize;

        // using unchecked here is safe, since we established in the
        // constructor that all entries within `self.offsets`` are
        //  (i) non-decreasing (i.e. produce a valid range) and
        //  (ii) are within bounds of `self.buffer`
        unsafe { self.buffer.get_unchecked_mut(start..end) }
    }
}
