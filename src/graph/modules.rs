use super::*;
use std::{cell::RefCell, rc::Rc};

pub trait Modules {
    fn compute_modules(&self) -> Option<Partition>;
}

impl<G> Modules for G
where
    G: AdjacencyList,
{
    fn compute_modules(&self) -> Option<Partition> {
        panic!("You should consider this implementation broken; not tested!");
        #[allow(unreachable_code)]
        find_modules(self)
    }
}

struct ListEntry {
    elements: BitSet,
    can_pivot: bool,
    is_center: bool,
    next: Option<Rc<RefCell<ListEntry>>>,
}

impl ListEntry {
    fn new(elements: BitSet, can_pivot: bool, is_center: bool) -> Self {
        Self {
            elements,
            can_pivot,
            is_center,
            next: None,
        }
    }

    fn append(&mut self, mut elem: ListEntry) {
        elem.next = std::mem::take(&mut self.next);
        self.next = Some(Rc::new(RefCell::new(elem)));
    }
}

trait Cursor: Sized {
    fn peek_next(&self) -> Option<Self>;

    fn try_move_next(&mut self) -> bool {
        if let Some(next) = self.peek_next() {
            *self = next;
            true
        } else {
            false
        }
    }

    fn split(&self, neighbors: &BitSet, include_first: bool) -> bool;
}

impl Cursor for Rc<RefCell<ListEntry>> {
    fn peek_next(&self) -> Option<Self> {
        self.as_ref().borrow().next.as_ref().cloned()
    }

    fn split(&self, neighbors: &BitSet, include_first: bool) -> bool {
        let (own_elem, next_elem) = {
            let mut own_elements = self.as_ref().borrow().elements.clone();

            let mut s1 = own_elements.clone();
            s1 &= neighbors;

            own_elements -= neighbors;

            if include_first {
                (s1, own_elements)
            } else {
                (own_elements, s1)
            }
        };

        if own_elem.cardinality() == 0 || next_elem.cardinality() == 0 {
            return false;
        }

        let new_entry = ListEntry::new(next_elem, true, false);
        let mut own_entry = self.as_ref().borrow_mut();

        own_entry.elements = own_elem;
        own_entry.append(new_entry);

        true
    }
}

pub fn find_modules<G: AdjacencyList>(graph: &G) -> Option<Partition> {
    for u in graph.vertices_range() {
        let partition = find_modules_impl(graph, u);
        if partition.number_of_classes() < graph.number_of_nodes() {
            return Some(partition);
        }
    }
    None
}

pub fn find_modules_impl<G: AdjacencyList>(graph: &G, pivot: Node) -> Partition {
    let mut neighbors = graph.neighbors_of_as_bitset(pivot);
    neighbors.clear_bit(pivot);

    let mut list = ListEntry::new(neighbors.clone(), true, false);
    list.append(ListEntry::new(
        BitSet::new_with_bits_set(graph.number_of_nodes(), std::iter::once(pivot)),
        false,
        true,
    ));

    neighbors.flip_all();
    neighbors.clear_bit(pivot);

    list.next
        .as_mut()
        .unwrap()
        .borrow_mut()
        .append(ListEntry::new(neighbors, true, false));

    let list = Rc::new(RefCell::new(list));

    let mut c_entry = list.clone();

    loop {
        let mut changed = false;

        loop {
            {
                let elem = c_entry.as_ref().borrow();

                if elem.can_pivot {
                    for u in elem.elements.iter_set_bits() {
                        let mut o_entry = list.clone();

                        let mut passed_other = false;
                        let mut passed_center = false;

                        let neighbors = graph.neighbors_of_as_bitset(u);

                        loop {
                            if o_entry.as_ptr() == c_entry.as_ptr() {
                                passed_other = true;
                            } else if o_entry.as_ref().borrow().is_center {
                                passed_center = true;
                            } else if o_entry
                                .as_ref()
                                .borrow()
                                .elements
                                .iter_set_bits()
                                .any(|v| neighbors.get_bit(v))
                            {
                                let include_first = passed_center != passed_other;
                                changed |= o_entry.split(&neighbors, include_first);
                            }

                            if !o_entry.try_move_next() {
                                break;
                            }
                        }
                    }
                }
            }

            if !c_entry.try_move_next() {
                break;
            }
        }

        if !changed {
            break;
        }
    }

    let mut c_entry = list;

    let mut partition = Partition::new(graph.number_of_nodes());
    loop {
        partition.add_class(c_entry.as_ref().borrow().elements.iter_set_bits());

        if !c_entry.try_move_next() {
            break;
        }
    }

    partition
}
