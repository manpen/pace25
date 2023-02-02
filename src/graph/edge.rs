use super::*;

pub trait EdgeOps {
    fn normalized(&self) -> Self;
    fn is_normalized(&self) -> bool;
    fn is_loop(&self) -> bool;
    fn reverse(&self) -> Self;
}

impl EdgeOps for Edge {
    fn normalized(&self) -> Self {
        (self.0.min(self.1), self.0.max(self.1))
    }

    fn is_normalized(&self) -> bool {
        self.0 <= self.1
    }

    fn is_loop(&self) -> bool {
        self.0 == self.1
    }

    fn reverse(&self) -> Self {
        (self.1, self.0)
    }
}

impl EdgeOps for ColoredEdge {
    fn normalized(&self) -> Self {
        (self.0.min(self.1), self.0.max(self.1), self.2)
    }

    fn is_normalized(&self) -> bool {
        self.0 <= self.1
    }

    fn is_loop(&self) -> bool {
        self.0 == self.1
    }

    fn reverse(&self) -> Self {
        (self.1, self.0, self.2)
    }
}
