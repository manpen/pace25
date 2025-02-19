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

impl<T> Default for SlicedBuffer<T> {
    fn default() -> Self {
        Self {
            buffer: Vec::new(),
            offsets: vec![0, 0],
        }
    }
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

    #[inline(always)]
    pub fn dissolve(self) -> (Vec<T>, Vec<NumEdges>) {
        (self.buffer, self.offsets)
    }

    #[inline(always)]
    pub fn double_mut(&mut self, u: Node, v: Node) -> (&mut [T], &mut [T]) {
        assert_ne!(u, v);

        if u < v {
            let v_off = self.offsets[v as usize] as usize;
            let (beg, end) = self.buffer.split_at_mut(v_off);

            let u_start = self.offsets[u as usize] as usize;
            let u_end = self.offsets[u as usize + 1] as usize;

            let v_len = self.offsets[v as usize + 1] as usize - v_off;

            // using unchecked here is safe, since we established in the
            // constructor that all entries within `self.offsets`` are
            //  (i) non-decreasing (i.e. produce a valid range) and
            //  (ii) are within bounds of `self.buffer`
            unsafe {
                (
                    beg.get_unchecked_mut(u_start..u_end),
                    end.get_unchecked_mut(0..v_len),
                )
            }
        } else {
            let u_off = self.offsets[u as usize] as usize;
            let (beg, end) = self.buffer.split_at_mut(u_off);

            let v_start = self.offsets[v as usize] as usize;
            let v_end = self.offsets[v as usize + 1] as usize;

            let u_len = self.offsets[u as usize + 1] as usize - u_off;

            // using unchecked here is safe, since we established in the
            // constructor that all entries within `self.offsets`` are
            //  (i) non-decreasing (i.e. produce a valid range) and
            //  (ii) are within bounds of `self.buffer`
            unsafe {
                (
                    end.get_unchecked_mut(0..u_len),
                    beg.get_unchecked_mut(v_start..v_end),
                )
            }
        }
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

// Extends on the core data structure of CSR (i.e. a data vector `buffer`
// and an non-decreasing index vector offsets, where the start of the
// i-th slice in `buffer` is stored at `offsets[i]`.
//
// Stores an additional default-buffer which is immutable and the same length
// (and offsets) as the main buffer. Allows restoring of data by copying the
// default data into the main buffer.
//
// This implementation verifies the following invariants at construction
// and to avoid repeated checks during accesses:
//  (0) `offset` has at least two elements
//  (1) `offset` is non-decreasing (i.e. produce a valid range) and
//  (2) `offset` stays within bounds of `buffer`
//  (3) `buffer` and `default` have the same length
//
// The implementation is its own module to prevent the CSR data structure
// from manipulating the offsets vector, which may invalidate the aforementioned
// invariants.
#[derive(Debug, Clone)]
pub struct SlicedBufferWithDefault<T> {
    buffer: Vec<T>,
    default: Vec<T>,
    offsets: Vec<NumEdges>,
}

impl<T> Default for SlicedBufferWithDefault<T> {
    fn default() -> Self {
        Self {
            buffer: Vec::new(),
            default: Vec::new(),
            offsets: vec![0, 0],
        }
    }
}

impl<T: Default + Clone> SlicedBufferWithDefault<T> {
    /// Constructs the SlicedBuffer and panics if one of the three
    /// invariants on offset are violated.
    pub fn new(default: Vec<T>, offsets: Vec<NumEdges>) -> Self {
        assert!(offsets.len() > 1);
        assert!(offsets.len() - 1 <= Node::MAX as usize);
        assert!(offsets.is_sorted());
        assert!(*offsets.last().unwrap() as usize <= default.len());

        let buffer = vec![T::default(); default.len()];
        Self {
            buffer,
            default,
            offsets,
        }
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

    #[inline(always)]
    pub fn restore_node(&mut self, u: Node) {
        let offset = self.offsets[u as usize] as usize;
        let len = self.offsets[u as usize + 1] as usize - offset;

        // using unchecked here is safe, since we established in the
        // constructor that all entries within `self.offsets`` are
        //  (i) non-decreasing (i.e. produce a valid range) and
        //  (ii) are within bounds of `self.buffer`
        //  (iii) `self.buffer` and `self.default` have the same length
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.default.as_ptr().add(offset),
                self.buffer.as_mut_ptr().add(offset),
                len,
            );
        }
    }

    #[inline(always)]
    pub fn double_mut(&mut self, u: Node, v: Node) -> (&mut [T], &mut [T]) {
        assert_ne!(u, v);

        if u < v {
            let v_off = self.offsets[v as usize] as usize;
            let (beg, end) = self.buffer.split_at_mut(v_off);

            let u_start = self.offsets[u as usize] as usize;
            let u_end = self.offsets[u as usize + 1] as usize;

            let v_len = self.offsets[v as usize + 1] as usize - v_off;

            // using unchecked here is safe, since we established in the
            // constructor that all entries within `self.offsets`` are
            //  (i) non-decreasing (i.e. produce a valid range) and
            //  (ii) are within bounds of `self.buffer`
            unsafe {
                (
                    beg.get_unchecked_mut(u_start..u_end),
                    end.get_unchecked_mut(0..v_len),
                )
            }
        } else {
            let u_off = self.offsets[u as usize] as usize;
            let (beg, end) = self.buffer.split_at_mut(u_off);

            let v_start = self.offsets[v as usize] as usize;
            let v_end = self.offsets[v as usize + 1] as usize;

            let u_len = self.offsets[u as usize + 1] as usize - u_off;

            // using unchecked here is safe, since we established in the
            // constructor that all entries within `self.offsets`` are
            //  (i) non-decreasing (i.e. produce a valid range) and
            //  (ii) are within bounds of `self.buffer`
            unsafe {
                (
                    end.get_unchecked_mut(0..u_len),
                    beg.get_unchecked_mut(v_start..v_end),
                )
            }
        }
    }
}

impl<T: Default + Clone> Index<Node> for SlicedBufferWithDefault<T> {
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

impl<T: Default + Clone> IndexMut<Node> for SlicedBufferWithDefault<T> {
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
