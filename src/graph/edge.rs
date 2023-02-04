use super::*;

pub trait EdgeOps {
    fn normalized(&self) -> Self;
    fn is_normalized(&self) -> bool;
    fn is_loop(&self) -> bool;
    fn reverse(&self) -> Self;
}

pub trait ColorQuery {
    fn is_red(&self) -> bool;
    fn is_black(&self) -> bool;
    fn is_none(&self) -> bool;
    fn is_some(&self) -> bool {
        !self.is_none()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Edge(pub u32, pub u32);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct ColoredEdge(pub u32, pub u32, pub EdgeColor);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum EdgeColor {
    Black,
    Red,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum EdgeColorFilter {
    BlackOnly,
    RedOnly,
    BlackAndRed,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum EdgeKind {
    Black,
    Red,
    None,
}

impl ColorQuery for EdgeColor {
    fn is_red(&self) -> bool {
        *self == EdgeColor::Red
    }

    fn is_black(&self) -> bool {
        *self == EdgeColor::Black
    }

    fn is_none(&self) -> bool {
        false
    }
}

impl ColorQuery for EdgeKind {
    fn is_red(&self) -> bool {
        *self == EdgeKind::Red
    }

    fn is_black(&self) -> bool {
        *self == EdgeKind::Black
    }

    fn is_none(&self) -> bool {
        *self == EdgeKind::None
    }
}

impl ColorQuery for EdgeColorFilter {
    fn is_red(&self) -> bool {
        *self == EdgeColorFilter::RedOnly || *self == EdgeColorFilter::BlackAndRed
    }

    fn is_black(&self) -> bool {
        *self == EdgeColorFilter::BlackOnly || *self == EdgeColorFilter::BlackAndRed
    }

    fn is_none(&self) -> bool {
        false
    }
}

impl PartialEq<EdgeKind> for EdgeColor {
    fn eq(&self, other: &EdgeKind) -> bool {
        match *self {
            EdgeColor::Black => *other == EdgeKind::Black,
            EdgeColor::Red => *other == EdgeKind::Red,
        }
    }
}

impl PartialEq<EdgeColor> for EdgeKind {
    fn eq(&self, other: &EdgeColor) -> bool {
        match *self {
            EdgeKind::Black => *other == EdgeColor::Black,
            EdgeKind::Red => *other == EdgeColor::Red,
            EdgeKind::None => false,
        }
    }
}

impl EdgeOps for Edge {
    fn normalized(&self) -> Self {
        Edge(self.0.min(self.1), self.0.max(self.1))
    }

    fn is_normalized(&self) -> bool {
        self.0 <= self.1
    }

    fn is_loop(&self) -> bool {
        self.0 == self.1
    }

    fn reverse(&self) -> Self {
        Edge(self.1, self.0)
    }
}

impl EdgeOps for ColoredEdge {
    fn normalized(&self) -> Self {
        ColoredEdge(self.0.min(self.1), self.0.max(self.1), self.2)
    }

    fn is_normalized(&self) -> bool {
        self.0 <= self.1
    }

    fn is_loop(&self) -> bool {
        self.0 == self.1
    }

    fn reverse(&self) -> Self {
        ColoredEdge(self.1, self.0, self.2)
    }
}

impl From<(Node, Node)> for Edge {
    fn from(value: (Node, Node)) -> Self {
        Edge(value.0, value.1)
    }
}

impl From<&(Node, Node)> for Edge {
    fn from(value: &(Node, Node)) -> Self {
        Edge(value.0, value.1)
    }
}

impl From<&Edge> for Edge {
    fn from(value: &Edge) -> Self {
        *value
    }
}
