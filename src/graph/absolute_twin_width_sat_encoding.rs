use itertools::Itertools;

use crate::prelude::contraction_sequence::ContractionSequence;

use super::{AdjacencyList, GraphEdgeOrder, ColoredAdjacencyList, ColoredAdjacencyTest, GraphEdgeEditing, relative_twin_width_sat_encoding::RelativeTwinWidthSatEncoding};
use std::fmt::Debug;

pub enum SatEncodingError {
    Unsat,
    ResourcesDepleted
}



pub struct AbsoluteTwinWidthSatEncoding<G> {
    graph: G,
    edges: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>>,
    ord: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>>,
    merge: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>>,
    red: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>>>,

    variable_id: i32,

    d: u32,

    //Map node id to node index
    graph_mapping: fxhash::FxHashMap<u32, u32>,
    graph_reverse_mapping: fxhash::FxHashMap<u32, u32>,
}

impl<
        G: Clone
            + AdjacencyList
            + GraphEdgeOrder
            + ColoredAdjacencyList
            + ColoredAdjacencyTest
            + GraphEdgeEditing
            + Debug,
    > AbsoluteTwinWidthSatEncoding<G>
{
    #[inline]
    pub fn add_new_variable(&mut self) -> i32 {
        let variable = self.variable_id;
        self.variable_id += 1;
        variable
    }

    pub fn new(graph: &G, d: u32) -> AbsoluteTwinWidthSatEncoding<G> {
        let mut graph_mapping = fxhash::FxHashMap::default();
        let mut graph_reverse_mapping = fxhash::FxHashMap::default();
        let mut variable_id = 1;


        let mut ord: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>> =
            fxhash::FxHashMap::default();
        let mut merge: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>> =
            fxhash::FxHashMap::default();
        let mut edges: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>> =
            fxhash::FxHashMap::default();
        let mut red: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>>> =
            fxhash::FxHashMap::default();

        let mut index = 0;
        for node_id in graph.vertices() {
            // Only contract non simple vertices
            if graph.degree_of(node_id) > 0 {
                graph_mapping.insert(node_id, index);
                graph_reverse_mapping.insert(index, node_id);
                index += 1;
            }
        }

        for i in 0..graph_mapping.len() {
            ord.insert(i as u32,fxhash::FxHashMap::default());
            edges.insert(i as u32,fxhash::FxHashMap::default());
            merge.insert(i as u32,fxhash::FxHashMap::default());
            red.insert(i as u32,fxhash::FxHashMap::default());
            for j in 0..graph_mapping.len() {
                ord.get_mut(&(i as u32)).unwrap().insert(j as u32,variable_id);
                variable_id+=1;
            }

            for j in i + 1..graph_mapping.len() {
                edges.get_mut(&(i as u32)).unwrap().insert(j as u32,variable_id);
                variable_id+=1;

                if i <= graph_mapping.len() - d as usize {
                    red.get_mut(&(i as u32)).unwrap().insert(j as u32,fxhash::FxHashMap::default());
                    merge.get_mut(&(i as u32)).unwrap().insert(j as u32,variable_id);
                    variable_id+=1;

                    for k in j+1..graph_mapping.len() {
                        red.get_mut(&(i as u32)).unwrap().get_mut(&(j as u32)).unwrap().insert(k as u32,variable_id);
                        variable_id+=1;
                    }
                }
            }
        }

        AbsoluteTwinWidthSatEncoding { graph: graph.clone(), edges, ord, merge, red, variable_id,d, graph_mapping, graph_reverse_mapping }
    }

    fn get_ord(&self, i: usize, j: usize) -> i32 {
        *self.ord.get(&(i as u32)).unwrap().get(&(j as u32)).unwrap()
    }

    fn get_edge(&self, i: usize, j: usize) -> i32 {
        *self.edges.get(&(i as u32)).unwrap().get(&(j as u32)).unwrap()
    }

    fn get_merge(&self, i: usize, j: usize) -> i32 {
        *self.merge.get(&(i as u32)).unwrap().get(&(j as u32)).unwrap()
    }

    fn get_red(&self, i: usize, j: usize, k: usize) -> i32 {
        *self.red.get(&(i as u32)).unwrap().get(&(j as u32)).unwrap().get(&(k as u32)).unwrap()
    }

    fn encode_order(&mut self, formula: &mut Vec<Vec<i32>>) {
        formula.push(vec![self.get_ord(self.graph_mapping.len()-1,self.graph_mapping.len()-1)]);
        formula.push(vec![self.get_ord(self.graph_mapping.len()-2,self.graph_mapping.len()-2)]);
        for i in 0..(self.graph_mapping.len()-1) {
            let mut encoding = Vec::new();
            for k in 0..self.graph_mapping.len() {
                encoding.push(self.get_ord(i, k));
            }
            RelativeTwinWidthSatEncoding::<G>::cardinality_naive_at_most_1(&mut self.variable_id, &encoding, formula);
            RelativeTwinWidthSatEncoding::<G>::cardinality_at_least_1(encoding, formula);
        }

        for i in 0..(self.graph_mapping.len()-1) {
            let mut encoding = Vec::new();
            for k in 0..self.graph_mapping.len() {
                encoding.push(self.get_ord(k, i));
            }
            RelativeTwinWidthSatEncoding::<G>::cardinality_naive_at_most_1(&mut self.variable_id, &encoding, formula);
        }
    }

    #[allow(unused)]
    fn encode_edges(&self, formula: &mut Vec<Vec<i32>>) {
        for i in 0..self.graph_mapping.len() {
            for j in i+1..self.graph_mapping.len() {
                for k in 0..self.graph_mapping.len() {
                    for m in k+1..self.graph_mapping.len() {
                        if self.graph.has_black_edge(*self.graph_mapping.get(&(k as u32)).unwrap(), *self.graph_mapping.get(&(m as u32)).unwrap()) {
                            formula.push(vec![-self.get_ord(i, k),-self.get_ord(j, m),self.get_edge(i, j)]);
                            formula.push(vec![-self.get_ord(i, m),-self.get_ord(j, k),self.get_edge(i, j)]);
                        }
                        else {
                            formula.push(vec![-self.get_ord(i, k),-self.get_ord(j, m),-self.get_edge(i, j)]);
                            formula.push(vec![-self.get_ord(i, m),-self.get_ord(j, k),-self.get_edge(i, j)]);
                        }
                    }
                }
            }
        }
    }

    fn encode_edges_2(&mut self, formula: &mut Vec<Vec<i32>>) {
        let mut auxillarys_variables: Vec<Vec<i32>> = Vec::new();

        for _ in 0..(self.graph_mapping.len() as u32) {
            auxillarys_variables.push((0..self.graph_mapping.len()).map(|_| self.add_new_variable()).collect_vec());
        }

        // Unfinished!
        for i in 0..self.graph_mapping.len() {
            for j in 0..self.graph_mapping.len() {
                let node_id = *self.graph_mapping.get(&(j as u32)).unwrap();
                let neighbors = self.graph.neighbors_of_as_bitset(node_id);
                for k in 0..self.graph_mapping.len() {
                    if j == k {
                        continue;
                    }
                    let value = *self.graph_reverse_mapping.get(&(k as u32)).unwrap();
                    if neighbors.at(value) {
                        formula.push(vec![-self.get_ord(i, j),auxillarys_variables[i][k]]);
                    }
                    else {
                        formula.push(vec![-self.get_ord(i, j),-auxillarys_variables[i][k]]);
                    }
                }
                let mut vars: Vec<i32> = neighbors.iter().map(|x| self.get_ord(i, *self.graph_mapping.get(&x).unwrap() as usize)).collect();
                vars.push(-auxillarys_variables[i][j]);
                formula.push(vars);
            }
        }
    }

    #[allow(unused)]
    fn break_symmetry(&mut self, formula: &mut Vec<Vec<i32>>) {
        let mut auxillarys_variables: Vec<Vec<i32>> = Vec::new();
        for i in 0..self.graph_mapping.len()-self.d as usize {
            auxillarys_variables.push(Vec::new());
            for k in 1..self.graph_mapping.len() {
                let variable = self.add_new_variable();
                auxillarys_variables[i].push(variable);
                for j in i+1..self.graph_mapping.len() {
                    formula.push(vec![-self.get_merge(i,j),-self.get_ord(j, k),variable])
                }
            }
        }

        for i in 0..self.graph_mapping.len()-self.d as usize {
            for j in 0..self.graph_mapping.len() {
                for k in j+1..self.graph_mapping.len() {
                    formula.push(vec![-auxillarys_variables[i][j],self.get_ord(i, k)]);
                }
            }
        }
    }

    #[allow(unused)]
    fn skip_double_hops(&mut self, formula: &mut Vec<Vec<i32>>) {
        for i in 0..self.graph_mapping.len()-self.d as usize {
            for j in i+1..self.graph_mapping.len() {
                let mut vars = Vec::new();
                for k in i+1..self.graph_mapping.len() {
                    if j == k {
                        continue;
                    }

                    let caux = self.add_new_variable();
                    if i > 1 {
                        formula.push(vec![-caux,self.get_edge(i, k), self.get_red(i-1, i, k)]);
                        if j < k {
                            formula.push(vec![-caux,self.get_edge(j, k), self.get_red(i-1, j, k)]);
                        }
                        else {
                            formula.push(vec![-caux,self.get_edge(k, j), self.get_red(i-1, k, j)]);
                        }
                    }
                    else {
                        formula.push(vec![-caux,self.get_edge(i, k)]);
                        if j < k {
                            formula.push(vec![-caux,self.get_edge(j, k)]);

                        }
                        else {
                            formula.push(vec![-caux,self.get_edge(k, j)]);
                        }
                    }
                    vars.push(caux);
                }
                if i > 0 {
                    let mut total_vars = vec![-self.get_merge(i, j), self.get_edge(i, j), self.get_red(i-1, i, j)];
                    total_vars.extend(vars);
                    formula.push(total_vars);
                }
                else {
                    let mut total_vars = vec![-self.get_merge(i, j), self.get_edge(i, j)];
                    total_vars.extend(vars);
                    formula.push(total_vars);
                }
            }
        }
    }

    fn encode_merge(&mut self, formula: &mut Vec<Vec<i32>>) {
        for i in 0..self.graph_mapping.len()-self.d as usize {
            let vars : Vec<i32> = (i+1..self.graph_mapping.len()).map(|x| self.get_edge(i, x)).collect();
            RelativeTwinWidthSatEncoding::<G>::cardinality_naive_at_most_1(&mut self.variable_id, &vars, formula);
            RelativeTwinWidthSatEncoding::<G>::cardinality_at_least_1( vars, formula);
        }
    }

    fn encode_red(&mut self, formula: &mut Vec<Vec<i32>>) {
        for i in 0..self.graph_mapping.len()-self.d as usize {
            for j in i+1..self.graph_mapping.len() {
                for k in j+1..self.graph_mapping.len() {
                    if i > 0 {
                        formula.push(vec![-self.get_merge(i, j), -self.get_red(i-1, i, k), self.get_red(i, j, k)]);
                        formula.push(vec![-self.get_merge(i, k), -self.get_red(i-1, i, j), self.get_red(i, j, k)]);
                        formula.push(vec![-self.get_red(i-1, j, k), self.get_red(i, j, k)]);
                    }

                    formula.push(vec![-self.get_merge(i, j), -self.get_edge(i, k), self.get_edge(j, k), self.get_red(i, j, k)]);
                    formula.push(vec![-self.get_merge(i, j), -self.get_edge(j, k), self.get_edge(i, k), self.get_red(i, j, k)]);
                    formula.push(vec![-self.get_merge(i, k), -self.get_edge(i, j), self.get_edge(j, k), self.get_red(i, j, k)]);
                    formula.push(vec![-self.get_merge(i, k), -self.get_edge(j, k), self.get_edge(i, j), self.get_red(i, j, k)]);
                }
            }
        }
    }

    fn tred(&self, i: usize, j: usize, k: usize) -> i32 {
        if j < k {
            *self.red.get(&(i as u32)).unwrap().get(&(j as u32)).unwrap().get(&(k as u32)).unwrap()
        }
        else {
            *self.red.get(&(i as u32)).unwrap().get(&(k as u32)).unwrap().get(&(j as u32)).unwrap()
        }
    }

    fn encode_counters(&mut self, formula: &mut Vec<Vec<i32>>) {
        for i in 0..self.graph_mapping.len()-self.d as usize {
            for j in i+1..self.graph_mapping.len() {
                let vars : Vec<i32> = (i+1..self.graph_mapping.len()).filter_map(|k| if k != j { Some(self.tred(i, j, k))} else {None}).collect();
                RelativeTwinWidthSatEncoding::<G>::cardinality_at_most_k_sequential_encoding(&mut self.variable_id, &vars, self.d, formula);
            }
        }
    }

    pub fn encode(&mut self) -> Vec<Vec<i32>> {
        let mut formula = Vec::new();
        self.encode_edges_2(&mut formula);
        self.encode_order(&mut formula);
        self.encode_merge(&mut formula);
        self.encode_red(&mut formula);
        self.encode_counters(&mut formula);

        formula
    }

    // Please note that this will run the sat solver for every integer <= upper bound until the solution becomes unsatisfiable
    pub fn solve_kissat(&mut self) -> Result<(u32, ContractionSequence),SatEncodingError> {
        loop {
            let encoding = self.encode();
            println!("Solving for width {}",self.d);

            let mut kissat_solver = cat_solver::Solver::new();

            let mut mapping: fxhash::FxHashSet<i32> = fxhash::FxHashSet::default();
            for x in encoding.into_iter() {
                x.iter().for_each(|v| {
                    mapping.insert(v.abs());
                });
                kissat_solver.add_clause(x.into_iter());
            }

            // Max the limits
            kissat_solver.set_limit("conflicts", 10_000_000).unwrap();
            kissat_solver.set_limit("decisions", 10_000_000).unwrap();

            if let Some(solved) = kissat_solver.solve() {
                if solved {
                    return Ok((self.d, ContractionSequence::new(self.graph.number_of_nodes())));
                }
                else {
                    return Err(SatEncodingError::Unsat);
                }
            }
            else {
                println!("Ressource expired!");
                return Err(SatEncodingError::ResourcesDepleted);
            }
        }
    }
}