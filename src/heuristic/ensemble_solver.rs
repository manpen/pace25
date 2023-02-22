pub trait Solver {
    fn solve(&mut self, graph: &G) -> (u32,ContractionSequence);
}

pub struct EnsembleSolver {
    pub solvers: Vec<Box<dyn Solver>>
}
