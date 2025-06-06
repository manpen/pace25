use std::{
    collections::HashSet,
    hash::Hasher,
    sync::{Arc, Mutex},
    time::Duration,
};

use ::highs::{ColProblem, HighsModelStatus, Model, Row};
use fxhash::FxHasher64;
use itertools::Itertools as _;
use stream_bitset::prelude::*;

use crate::prelude::*;

type ProblemDigest = u64;

#[derive(Default)]
pub struct HighsCache {
    failed_runs: Mutex<HashSet<u64>>,
}

impl HighsCache {
    fn insert(&self, digest: ProblemDigest) {
        info!("Register failed solve attempt with digest {digest:?}");
        self.failed_runs.lock().unwrap().insert(digest);
    }

    fn contains(&self, digest: &ProblemDigest) -> bool {
        let found = self.failed_runs.lock().unwrap().contains(digest);
        if found {
            info!("Found failed solve attempt with digest {digest:?}");
        }
        found
    }
}

pub fn unit_weight(_: Node) -> f64 {
    1.0
}

pub struct HighsDominatingSetSolver {
    row_of_node: Vec<Option<Row>>,
    int_to_ext: Vec<Node>,

    con_buffer: Vec<(Node, Row)>, // prevent allocation during build

    cache: Option<Arc<HighsCache>>,
}

pub struct HighsProblem<'a, G> {
    digest: Option<ProblemDigest>,
    context: &'a mut HighsDominatingSetSolver,

    graph: &'a G,
    covered: &'a BitSet,
    nodes: Option<&'a [Node]>,

    problem: ColProblem,
    num_terms: NumEdges,
}

impl HighsDominatingSetSolver {
    pub fn new(n: NumNodes) -> Self {
        Self {
            row_of_node: vec![Default::default(); n as usize],
            int_to_ext: Vec::with_capacity(n as usize),
            cache: None,
            con_buffer: Vec::with_capacity(128),
        }
    }

    pub fn register_cache(&mut self, cache: Arc<HighsCache>) {
        self.cache = Some(cache)
    }

    pub fn build_problem<'a, G, W>(
        &'a mut self,
        graph: &'a G,
        covered: &'a BitSet,
        redundant: &'a BitSet,
        mut node_weights: W,
    ) -> HighsProblem<'a, G>
    where
        G: AdjacencyList + AdjacencyTest,
        W: FnMut(Node) -> f64,
    {
        let mut problem = ColProblem::default();

        let skip_constraints_of = self.compute_skip_constraints(graph, covered, redundant, None);

        for u in skip_constraints_of.iter_cleared_bits() {
            self.row_of_node[u as usize] = Some(problem.add_row(1..));
        }

        let mut num_terms = 0;
        let mut hasher = self.cache.as_ref().map(|_| FxHasher64::default());

        for node in redundant.iter_cleared_bits() {
            num_terms += self.add_column_to_problem(
                graph,
                &mut problem,
                node,
                node_weights(node),
                hasher.as_mut(),
            );
        }

        HighsProblem {
            digest: hasher.map(|h| h.finish()),
            context: self,
            graph,
            covered,
            nodes: None,
            num_terms,
            problem,
        }
    }

    pub fn build_problem_of_subgraph<'a, G, W>(
        &'a mut self,
        graph: &'a G,
        covered: &'a BitSet,
        redundant: &'a BitSet,
        nodes: &'a [Node],
        mut node_weights: W,
    ) -> HighsProblem<'a, G>
    where
        G: AdjacencyList + AdjacencyTest,
        W: FnMut(Node) -> f64,
    {
        let mut problem = ColProblem::default();

        let skip_constraints_of =
            self.compute_skip_constraints(graph, covered, redundant, Some(nodes));

        for &u in nodes.iter() {
            self.row_of_node[u as usize] =
                (!skip_constraints_of.get_bit(u)).then(|| problem.add_row(1..));
        }

        debug_assert!(
            nodes
                .iter()
                .filter(|&&u| redundant.get_bit(u) && !covered.get_bit(u))
                .all(|&u| self.row_of_node[u as usize].is_some())
        );

        let mut hasher = self.cache.as_ref().map(|_| FxHasher64::default());
        let mut num_terms = 0;
        for &node in nodes {
            if redundant.get_bit(node) {
                continue;
            }

            num_terms += self.add_column_to_problem(
                graph,
                &mut problem,
                node,
                node_weights(node),
                hasher.as_mut(),
            );
        }

        HighsProblem {
            digest: hasher.map(|h| h.finish()),
            context: self,
            graph,
            covered,
            nodes: Some(nodes),
            num_terms,
            problem,
        }
    }

    fn compute_skip_constraints<G>(
        &self,
        graph: &G,
        covered: &BitSet,
        redundant: &BitSet,
        nodes: Option<&[u32]>,
    ) -> BitSet
    where
        G: AdjacencyList + AdjacencyTest,
    {
        let mut skip_constraints_of = covered.clone();

        let mut test_redundant_node = |u| {
            if let Some((a, b)) = graph
                .neighbors_of(u)
                .filter(|&v| v != u && !redundant.get_bit(v))
                .collect_tuple()
                && graph.has_edge(a, b)
            {
                skip_constraints_of.set_bit(a);
                skip_constraints_of.set_bit(b);
            }
        };

        if let Some(nodes) = &nodes {
            for &u in nodes.iter() {
                if redundant.get_bit(u) && !covered.get_bit(u) {
                    test_redundant_node(u);
                }
            }
        } else {
            for u in (redundant.bitmask_stream() - covered).iter_set_bits() {
                test_redundant_node(u);
            }
        }

        skip_constraints_of
    }

    fn add_column_to_problem<G>(
        &mut self,
        graph: &G,
        problem: &mut ColProblem,
        node: Node,
        weight: f64,
        hasher: Option<&mut FxHasher64>,
    ) -> NumEdges
    where
        G: AdjacencyList + AdjacencyTest,
    {
        debug_assert!(self.con_buffer.is_empty());

        self.con_buffer.extend(
            graph
                .closed_neighbors_of(node)
                .filter_map(|v| Some((v, self.row_of_node[v as usize]?))),
        );

        if self.con_buffer.is_empty() {
            return 0;
        }

        let num_terms = self.con_buffer.len() as NumEdges;
        if let Some(hasher) = hasher {
            hasher.write_u32(node);
            hasher.write_u64(weight.to_bits());
            self.con_buffer.sort_unstable_by_key(|(u, _)| *u);
            for (v, _) in self.con_buffer.iter() {
                hasher.write_u32(*v);
            }
            hasher.write_u32(Node::MAX);
        }

        self.int_to_ext.push(node);
        problem.add_integer_column(
            weight,
            0..=1,
            self.con_buffer.drain(..).map(|(_, r)| (r, 1.0)),
        );

        num_terms
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SolverResult {
    Optimal(Vec<Node>),
    Suboptimal(Vec<Node>),
    Timeout,
    Infeasible,
}

impl SolverResult {
    pub fn solution(&self) -> Option<&[Node]> {
        match self {
            SolverResult::Optimal(items) => Some(items.as_slice()),
            SolverResult::Suboptimal(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    pub fn take_solution(self) -> Option<Vec<Node>> {
        match self {
            SolverResult::Optimal(items) => Some(items),
            SolverResult::Suboptimal(items) => Some(items),
            _ => None,
        }
    }
}

impl<G: AdjacencyList> HighsProblem<'_, G> {
    pub fn number_of_variables(&self) -> NumNodes {
        self.problem.num_cols() as NumNodes
    }

    pub fn number_of_terms(&self) -> NumEdges {
        self.num_terms
    }

    pub fn solve_exact(mut self, timeout: Option<Duration>) -> SolverResult {
        self.solve_impl(false, timeout)
    }

    pub fn solve_allow_subopt(mut self, timeout: Option<Duration>) -> SolverResult {
        let result = self.solve_impl(true, timeout);

        // We only get a Suboptimal solution, if a timeout happend. In this case,
        // the solver may return an infeasible solution. So, we explicitly check
        // whether a solution is feasible and return a Timeout if this is not the
        // case.
        if let SolverResult::Suboptimal(x) = &result {
            let mut covered = self.covered.clone();
            covered.set_bits(x.iter().flat_map(|u| self.graph.closed_neighbors_of(*u)));

            let infeasible = if let Some(subgraph) = self.nodes {
                subgraph.iter().any(|&u| !covered.get_bit(u))
            } else {
                !covered.are_all_set()
            };

            if infeasible {
                return SolverResult::Timeout;
            }
        }
        result
    }

    fn solve_impl(&mut self, allow_subopt: bool, timeout: Option<Duration>) -> SolverResult {
        if self.num_terms == 0 {
            return SolverResult::Optimal(Vec::new());
        }

        if let Some(cache) = &self.context.cache {
            let digest = self
                .digest
                .expect("A digest is computed iff a cacher is registered");
            if cache.contains(&digest) {
                info!("Skip previously failed solve attempt");
                return SolverResult::Timeout;
            }
        }

        // Prepare the model based on the previously computed problem
        let mut model = Model::new(std::mem::take(&mut self.problem));
        model.make_quiet();
        if let Some(tme) = timeout {
            model.set_option("time_limit", tme.as_secs_f64());
        }

        #[cfg(not(feature = "par"))]
        {
            model.set_option("parallel", "off");
            model.set_option("threads", "1");
        }
        model.set_sense(::highs::Sense::Minimise);

        // Solve and check whether a solution could be obtained
        let solved = model.solve();
        let mut subopt = false;
        match solved.status() {
            HighsModelStatus::Optimal => {}
            HighsModelStatus::Infeasible => return SolverResult::Infeasible,
            HighsModelStatus::ReachedTimeLimit => {
                if !allow_subopt {
                    self.try_to_cache_timeout();
                    return SolverResult::Timeout;
                }
                subopt = true;
            }
            e => panic!("Unhandled HighsStatus: {e:?}"),
        };

        // Extract solution
        let solution = solved
            .get_solution()
            .columns()
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v > 0.5).then_some(self.context.int_to_ext[i]))
            .collect();

        if subopt {
            SolverResult::Suboptimal(solution)
        } else {
            SolverResult::Optimal(solution)
        }
    }

    fn try_to_cache_timeout(&self) {
        if let Some(cache) = &self.context.cache {
            let digest = self
                .digest
                .expect("A digest is computed iff a cacher is registered");
            cache.insert(digest);
        }
    }
}

impl<G> Drop for HighsProblem<'_, G> {
    fn drop(&mut self) {
        if let Some(nodes) = self.nodes
            && nodes.len() < self.context.row_of_node.len() / 128
        {
            for &u in nodes {
                self.context.row_of_node[u as usize] = None;
            }
        } else {
            self.context.row_of_node.fill(None);
        }

        self.context.int_to_ext.clear();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    use crate::{
        exact::naive::naive_solver,
        graph::{AdjArray, GnpGenerator, GraphNodeOrder},
    };

    fn generate_random_graphs(n: NumNodes) -> impl Iterator<Item = (AdjArray, BitSet, BitSet)> {
        let mut rng = Pcg64Mcg::seed_from_u64(0x1234567);

        (0..).filter_map(move |i| {
            let graph = AdjArray::random_gnp(&mut rng, n, 3. / n as f64);

            let mut covered = graph.vertex_bitset_unset();
            for _ in 0..i % 7 {
                covered.set_bit(rng.gen_range(graph.vertices_range()));
            }
            let mut redundant = graph.vertex_bitset_unset();
            for _ in 0..i % 5 {
                redundant.set_bit(rng.gen_range(graph.vertices_range()));
            }
            redundant -= &covered;

            {
                // reject if infeasible
                let mut tmp = DominatingSet::new(graph.number_of_nodes());
                tmp.add_nodes(redundant.iter_cleared_bits());
                if !tmp.is_valid_given_previous_cover(&graph, &covered) {
                    return None;
                }
            }

            Some((graph, covered, redundant))
        })
    }

    #[test]
    fn cross_with_naive() {
        const NODES: NumNodes = 20;

        let mut solver = HighsDominatingSetSolver::new(NODES);
        for (graph, covered, redundant) in generate_random_graphs(NODES).take(500) {
            let naive = naive_solver(&graph, &covered, &redundant, None, None).unwrap();
            let problem = solver.build_problem(&graph, &covered, &redundant, unit_weight);

            let mut highs = DominatingSet::new(NODES);
            highs.add_nodes(
                problem
                    .solve_exact(None)
                    .solution()
                    .unwrap()
                    .iter()
                    .cloned(),
            );

            assert!(highs.is_valid_given_previous_cover(&graph, &covered));
            assert_eq!(naive.len(), highs.len());
            assert!(highs.iter().all(|u| !redundant.get_bit(u)));
        }
    }

    #[test]
    fn cross_with_naive_subgraph() {
        const NODES: NumNodes = 20;

        let mut solver = HighsDominatingSetSolver::new(NODES);
        let nodes_list = (0..NODES).collect_vec();

        for (graph, covered, redundant) in generate_random_graphs(NODES).take(500) {
            let naive = naive_solver(&graph, &covered, &redundant, None, None).unwrap();
            let problem = solver.build_problem_of_subgraph(
                &graph,
                &covered,
                &redundant,
                &nodes_list,
                unit_weight,
            );
            let mut highs = DominatingSet::new(NODES);
            highs.add_nodes(
                problem
                    .solve_exact(None)
                    .solution()
                    .unwrap()
                    .iter()
                    .cloned(),
            );

            assert!(highs.is_valid_given_previous_cover(&graph, &covered));
            assert_eq!(naive.len(), highs.len());
            assert!(highs.iter().all(|u| !redundant.get_bit(u)));
        }
    }
}
