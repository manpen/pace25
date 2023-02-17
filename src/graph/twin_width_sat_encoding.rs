//use crate::graph::{AdjacencyList, GraphEdgeOrder, GraphEdgeEditing};
use log::info;
use splr::Certificate;
use varisat::{CnfFormula, ExtendFormula, Lit};

use crate::{
    exact::contraction_sequence::ContractionSequence,
    graph::{
        AdjacencyList, ColoredAdjacencyList, ColoredAdjacencyTest, GraphEdgeEditing, GraphEdgeOrder,
    },
};
use core::fmt::Debug;
use std::cmp::Ordering;

pub struct TwinWidthSatEncoding<G> {
    graph: G,
    edges: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>>>,
    ord: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>>,
    merge: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>>,
    variable_id: i32,
}

impl<
        G: Clone
            + AdjacencyList
            + GraphEdgeOrder
            + ColoredAdjacencyList
            + ColoredAdjacencyTest
            + GraphEdgeEditing
            + Debug,
    > TwinWidthSatEncoding<G>
{
    #[inline]
    pub fn add_new_variable(&mut self) -> i32 {
        let variable = self.variable_id;
        self.variable_id += 1;
        variable
    }

    pub fn new(graph: &G) -> Self {
        let mut variable_id = 1;

        let mut ord: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>> =
            fxhash::FxHashMap::default();
        let mut merge: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>> =
            fxhash::FxHashMap::default();
        let mut edges: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>>> =
            fxhash::FxHashMap::default();

        for i in 0..(graph.number_of_nodes() as usize) {
            for j in i + 1..(graph.number_of_nodes() as usize) {
                for k in 0..(graph.number_of_nodes() as usize) {
                    edges
                        .entry(k as u32)
                        .and_modify(|f| {
                            f.entry(i as u32)
                                .and_modify(|f| {
                                    f.insert(j as u32, variable_id);
                                    variable_id += 1;
                                })
                                .or_insert({
                                    let mut jmap = fxhash::FxHashMap::default();
                                    jmap.insert(j as u32, variable_id);
                                    variable_id += 1;
                                    jmap
                                });
                        })
                        .or_insert({
                            let mut imap = fxhash::FxHashMap::default();
                            let mut jmap = fxhash::FxHashMap::default();

                            jmap.insert(j as u32, variable_id);
                            variable_id += 1;

                            imap.insert(i as u32, jmap);
                            imap
                        });
                }
                ord.entry(i as u32)
                    .and_modify(|f: &mut fxhash::FxHashMap<u32, i32>| {
                        f.insert(j as u32, variable_id);
                        variable_id += 1;
                    })
                    .or_insert({
                        let mut jmap = fxhash::FxHashMap::default();
                        jmap.insert(j as u32, variable_id);
                        variable_id += 1;
                        jmap
                    });

                merge
                    .entry(i as u32)
                    .and_modify(|f: &mut fxhash::FxHashMap<u32, i32>| {
                        f.insert(j as u32, variable_id);
                        variable_id += 1;
                    })
                    .or_insert({
                        let mut jmap = fxhash::FxHashMap::default();
                        jmap.insert(j as u32, variable_id);
                        variable_id += 1;
                        jmap
                    });
            }
        }

        TwinWidthSatEncoding {
            graph: graph.clone(),
            edges,
            ord,
            merge,
            variable_id,
        }
    }

    #[inline]
    pub fn tord(&self, i: u32, j: u32) -> i32 {
        // Apply permutation which we used on ord
        if i < j {
            *self.ord.get(&i).unwrap().get(&j).unwrap()
        } else {
            -*self.ord.get(&j).unwrap().get(&i).unwrap()
        }
    }

    #[inline]
    pub fn tedge(&self, n: u32, i: u32, j: u32) -> i32 {
        // Apply permutation which we used on edges
        if i < j {
            *self
                .edges
                .get(&n)
                .unwrap()
                .get(&i)
                .unwrap()
                .get(&j)
                .unwrap()
        } else {
            *self
                .edges
                .get(&n)
                .unwrap()
                .get(&j)
                .unwrap()
                .get(&i)
                .unwrap()
        }
    }

    #[inline]
    pub fn get_merge(&self, i: u32, j: u32) -> i32 {
        *self.merge.get(&i).unwrap().get(&j).unwrap()
    }

    #[inline]
    pub fn cardinality_at_least_1(vars: Vec<i32>, formula: &mut Vec<Vec<i32>>) {
        // Encode at least one
        formula.push(vars);
    }

    #[inline]
    pub fn cardinality_naive_at_most_1(vars: &Vec<i32>, formula: &mut Vec<Vec<i32>>) {
        // Encode at least one
        for i in 0..vars.len() {
            for j in i + 1..vars.len() {
                formula.push(vec![-vars[i], -vars[j]]);
            }
        }
    }

    pub fn cardinality_at_most_k_sequential_encoding(
        id_counter: &mut i32,
        vars: &Vec<i32>,
        upper_bound: u32,
        formula: &mut Vec<Vec<i32>>,
    ) {
        // Based on the description of the sequential encoding from here https://www.carstensinz.de/papers/CP-2005.pdf
        if upper_bound == 0 {
            for x in vars.iter() {
                formula.push(vec![-*x]);
            }
            return;
        }

        let mut restrictions: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>> =
            fxhash::FxHashMap::default();

        for i in 1..vars.len() + 1 {
            let mut inner = fxhash::FxHashMap::default();
            for j in 1..upper_bound + 1 {
                inner.insert(j, *id_counter);
                *id_counter += 1;
            }
            restrictions.insert(i as u32, inner);
        }

        // not x_1 OR s_1_1
        formula.push(vec![
            -vars[0],
            *restrictions.get(&1).unwrap().get(&1).unwrap(),
        ]);

        for j in 1..upper_bound + 1 {
            // NOT
            formula.push(vec![-*restrictions.get(&1).unwrap().get(&j).unwrap()]);
        }

        for i in 2..vars.len() {
            formula.push(vec![
                -vars[i - 1],
                *restrictions.get(&(i as u32)).unwrap().get(&1).unwrap(),
            ]);
            formula.push(vec![
                -*restrictions.get(&(i as u32 - 1)).unwrap().get(&1).unwrap(),
                *restrictions.get(&(i as u32)).unwrap().get(&1).unwrap(),
            ]);
            for j in 2..upper_bound + 1 {
                formula.push(vec![
                    -vars[i - 1],
                    -*restrictions
                        .get(&((i - 1) as u32))
                        .unwrap()
                        .get(&(j - 1))
                        .unwrap(),
                    *restrictions.get(&(i as u32)).unwrap().get(&j).unwrap(),
                ]);
                formula.push(vec![
                    -*restrictions
                        .get(&((i - 1) as u32))
                        .unwrap()
                        .get(&j)
                        .unwrap(),
                    *restrictions.get(&(i as u32)).unwrap().get(&j).unwrap(),
                ]);
            }
            formula.push(vec![
                -vars[i - 1],
                -*restrictions
                    .get(&(i as u32 - 1))
                    .unwrap()
                    .get(&(upper_bound))
                    .unwrap(),
            ]);
        }

        formula.push(vec![
            -vars[vars.len() - 1],
            -*restrictions
                .get(&((vars.len() - 1) as u32))
                .unwrap()
                .get(&(upper_bound))
                .unwrap(),
        ]);
    }

    pub fn amo_commander(&mut self, vars: Vec<i32>, m: u32, formula: &mut Vec<Vec<i32>>) {
        let mut cnt = 0;
        let mut groups: Vec<Vec<i32>> = Vec::new();

        while cnt < vars.len() {
            let mut current_group = Vec::new();
            for i in 0..m.min(vars.len() as u32 - cnt as u32) {
                current_group.push(vars[cnt + i as usize])
            }
            groups.push(current_group);
            cnt += m as usize;
        }

        let mut commands = Vec::new();
        for mut current_group in groups.into_iter() {
            if current_group.len() > 1 {
                let new_command = self.add_new_variable();

                commands.push(new_command);
                current_group.push(-new_command);
                TwinWidthSatEncoding::<G>::cardinality_naive_at_most_1(&current_group, formula);
                TwinWidthSatEncoding::<G>::cardinality_at_least_1(current_group, formula);
            } else {
                commands.push(current_group[0]);
            }
        }
        if commands.len() < 2 * m as usize {
            TwinWidthSatEncoding::<G>::cardinality_naive_at_most_1(&commands, formula);
        } else {
            self.amo_commander(commands, m, formula);
        }
    }

    pub fn encode_reds(&mut self, formula: &mut Vec<Vec<i32>>) {
        let mut auxillarys_variables: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>> =
            fxhash::FxHashMap::default();

        for i in 0..self.graph.number_of_nodes() {
            auxillarys_variables.insert(i, fxhash::FxHashMap::default());
            for j in i + 1..self.graph.number_of_nodes() {
                if j == i {
                    continue;
                }

                let aux_var = self.add_new_variable();
                auxillarys_variables.get_mut(&i).unwrap().insert(j, aux_var);

                for k in 0..self.graph.number_of_nodes() {
                    if k == j || i == k {
                        continue;
                    }
                    formula.push(vec![-self.tedge(k, i, j), aux_var]);
                }
            }
        }

        for i in 0..self.graph.number_of_nodes() {
            for j in 0..self.graph.number_of_nodes() {
                if i == j {
                    continue;
                }

                for k in 0..self.graph.number_of_nodes() {
                    if j == k || i == k {
                        continue;
                    }

                    for m in k + 1..self.graph.number_of_nodes() {
                        if i == m || j == m {
                            continue;
                        }
                        formula.push(vec![
                            -self.tord(i, j),
                            -self.tord(j, k),
                            -self.tord(j, m),
                            -self.tedge(i, k, m),
                            self.tedge(j, k, m),
                        ]);
                    }
                }
            }
        }

        for i in 0..self.graph.number_of_nodes() {
            for j in i + 1..self.graph.number_of_nodes() {
                if i == j {
                    continue;
                }

                for k in 1..self.graph.number_of_nodes() {
                    if k == i || k == j {
                        continue;
                    }

                    if i < k {
                        formula.push(vec![
                            -self.get_merge(i, j),
                            -self.tord(i, k),
                            -*auxillarys_variables.get(&i).unwrap().get(&k).unwrap(),
                            self.tedge(i, j, k),
                        ]);
                    } else {
                        formula.push(vec![
                            -self.get_merge(i, j),
                            -self.tord(i, k),
                            -*auxillarys_variables.get(&k).unwrap().get(&i).unwrap(),
                            self.tedge(i, j, k),
                        ]);
                    }
                }
            }
        }
    }

    pub fn decode_sat_solution(&self, solution: Vec<i32>) -> ContractionSequence {
        let mut contraction_sequence = ContractionSequence::new(self.graph.number_of_nodes());

        let mut decoded = fxhash::FxHashMap::default();
        let mut merge = fxhash::FxHashMap::default();

        for x in solution.iter() {
            decoded.insert(x.abs(), *x > 0);
        }

        let mut ordering: Vec<u32> = (0..self.graph.len() as u32).collect();
        let find_order = |x: &u32, y: &u32| -> Ordering {
            if x < y {
                let id = self.ord.get(x).unwrap().get(y).unwrap();
                if *decoded.get(id).unwrap() {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            } else {
                let id = self.ord.get(y).unwrap().get(x).unwrap();
                if *decoded.get(id).unwrap() {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
        };

        for i in 0..self.graph.number_of_nodes() {
            for y in i + 1..self.graph.number_of_nodes() {
                let id = self.merge.get(&i).unwrap().get(&y).unwrap();
                if *decoded.get(id).unwrap() {
                    merge.insert(i, y);
                }
            }
        }

        ordering.sort_by(find_order);

        for i in ordering[0..ordering.len() - 1].iter() {
            contraction_sequence.merge_node_into(*i, *merge.get(i).unwrap());
        }

        contraction_sequence
    }

    // Please note that this will run the sat solver for every integer <= upper bound until the solution becomes unsatisfiable
    pub fn solve(&mut self, mut ub: u32) -> Option<(u32, ContractionSequence)> {
        let mut last_valid_solution = None;
        let mut last_valid_bound = ub;
        loop {
            println!("Encoding for twin width max d={ub}");
            let encoding = self.encode(ub);

            match Certificate::try_from(encoding).expect("panic!") {
                Certificate::UNSAT => {
                    println!(
                        "Finding twin width d={ub} is impossible returning previous solution!"
                    );
                    if let Some(solution) = last_valid_solution {
                        let seq = self.decode_sat_solution(solution);
                        return Some((last_valid_bound, seq));
                    } else {
                        return None;
                    }
                }
                Certificate::SAT(vec) => {
                    if ub == 0 {
                        let seq = self.decode_sat_solution(vec);
                        return Some((ub, seq));
                    }
                    last_valid_bound = ub;
                    last_valid_solution = Some(vec);
                    ub -= 1;
                    continue;
                }
            }
        }
    }

    // Please note that this will run the sat solver for every integer <= upper bound until the solution becomes unsatisfiable
    pub fn solve_varisat(&mut self, mut ub: u32) -> Option<(u32, ContractionSequence)> {
        let mut last_valid_solution = None;
        let mut last_valid_bound = ub;
        loop {
            info!("Encoding done for twin width max d={ub}");
            let encoding = self.encode(ub);

            let mut cnf = CnfFormula::new();

            let mut mapping: fxhash::FxHashMap<i32, Lit> = fxhash::FxHashMap::default();
            let mut reversed_mapping: fxhash::FxHashMap<usize, i32> = fxhash::FxHashMap::default();
            for clauses in encoding.into_iter() {
                let literals: Vec<Lit> = clauses
                    .into_iter()
                    .map(|x| {
                        let absolute = x.abs();
                        if let Some(exists) = mapping.get(&absolute) {
                            if x < 0 {
                                exists.var().negative()
                            } else {
                                exists.var().positive()
                            }
                        } else {
                            let literal = cnf.new_lit();
                            mapping.insert(absolute, literal);
                            reversed_mapping.insert(literal.index(), x.abs());
                            if x < 0 {
                                literal.var().negative()
                            } else {
                                literal.var().positive()
                            }
                        }
                    })
                    .collect();
                cnf.add_clause(&literals);
            }

            let solver_time = std::time::Instant::now();
            let mut solver = varisat::Solver::new();
            solver.add_formula(&cnf);
            if let Ok(solved) = solver.solve() {
                if solved {
                    info!("Found solution in {}ms", solver_time.elapsed().as_millis());
                    let solution = solver.model().unwrap();
                    let sat_solution: Vec<i32> = solution
                        .into_iter()
                        .map(|x| {
                            if x.is_negative() {
                                -*reversed_mapping.get(&x.index()).unwrap()
                            } else {
                                *reversed_mapping.get(&x.index()).unwrap()
                            }
                        })
                        .collect();
                    if ub == 0 {
                        let seq = self.decode_sat_solution(sat_solution);
                        return Some((ub, seq));
                    }
                    last_valid_bound = ub;
                    last_valid_solution = Some(sat_solution);
                    ub -= 1;
                    continue;
                } else if let Some(solution) = last_valid_solution {
                    let seq = self.decode_sat_solution(solution);
                    return Some((last_valid_bound, seq));
                } else {
                    return None;
                }
            }
        }
    }

    pub fn encode(&mut self, d: u32) -> Vec<Vec<i32>> {
        let mut formula = Vec::new();
        for i in 0..self.graph.number_of_nodes() {
            for j in 0..self.graph.number_of_nodes() {
                if i == j {
                    continue;
                }

                for k in 0..self.graph.number_of_nodes() {
                    if i == k || k == j {
                        continue;
                    }
                    formula.push(vec![-self.tord(i, j), -self.tord(j, k), self.tord(i, k)]);
                }
            }
        }
        for i in 0..self.graph.number_of_nodes() {
            for j in i + 1..self.graph.number_of_nodes() {
                formula.push(vec![-self.get_merge(i, j), self.tord(i, j)])
            }
        }

        for i in 0..(self.graph.number_of_nodes() - 1) {
            let mut atleast_encoded = Vec::new();
            let mut amocommander_encoded = Vec::new();
            for j in i + 1..self.graph.number_of_nodes() {
                let var = *self.merge.get(&i).unwrap().get(&j).unwrap();
                atleast_encoded.push(var);
                amocommander_encoded.push(var);
            }
            TwinWidthSatEncoding::<G>::cardinality_at_least_1(atleast_encoded, &mut formula);
            self.amo_commander(amocommander_encoded, 2, &mut formula);
        }

        for i in 0..self.graph.number_of_nodes() {
            let neighbors_i = self.graph.neighbors_of_as_bitset(i);
            for j in i + 1..self.graph.number_of_nodes() {
                let mut neighbors_j = self.graph.neighbors_of_as_bitset(j);
                neighbors_j.unset_bit(i);

                neighbors_j.xor(&neighbors_i);
                neighbors_j.unset_bit(j);

                for k in neighbors_j.iter() {
                    formula.push(vec![
                        -self.get_merge(i, j),
                        -self.tord(i, k),
                        self.tedge(i, j, k),
                    ]);
                }
            }
        }

        self.encode_reds(&mut formula);

        // Totalizer!
        for i in 0..self.graph.number_of_nodes() - 1 {
            for x in 0..self.graph.number_of_nodes() {
                let mut vars = Vec::new();
                for y in 0..self.graph.number_of_nodes() {
                    if x == y {
                        continue;
                    }
                    vars.push(self.tedge(i, x, y));
                }

                TwinWidthSatEncoding::<G>::cardinality_at_most_k_sequential_encoding(
                    &mut self.variable_id,
                    &vars,
                    d,
                    &mut formula,
                );
            }
        }

        //sb_ord
        for i in 0..self.graph.number_of_nodes() - 1 {
            formula.push(vec![*self
                .ord
                .get(&i)
                .unwrap()
                .get(&(self.graph.number_of_nodes() - 1))
                .unwrap()]);
        }

        //sb_red
        for i in 0..self.graph.number_of_nodes() {
            for j in i + 1..self.graph.number_of_nodes() {
                if i == j {
                    continue;
                }

                for k in 1..self.graph.number_of_nodes() {
                    if k == i || k == j {
                        continue;
                    }
                    formula.push(vec![-self.tord(j, i), -self.tedge(i, j, k)]);
                }
            }
        }

        formula
    }
}

#[cfg(test)]
pub mod tests {
    use std::{fs::File, io::BufReader};

    use splr::Certificate;

    use crate::{
        graph::{AdjArray, EdgeColor, GraphEdgeEditing, GraphNew},
        io::PaceReader,
    };

    use super::TwinWidthSatEncoding;

    #[test]
    fn simple_encoding() {
        let graph = AdjArray::test_only_from([(1, 2), (1, 0), (4, 3), (0, 5), (5, 4)]);
        let mut encoding = TwinWidthSatEncoding::new(&graph);

        encoding.encode(2);
    }

    #[test]
    fn at_most_k() {
        let vars = vec![1, 2, 3, 4, 5, 6];
        let mut id = 7;
        let mut formula = Vec::new();
        TwinWidthSatEncoding::<AdjArray>::cardinality_at_most_k_sequential_encoding(
            &mut id,
            &vars,
            2,
            &mut formula,
        );

        formula.push(vec![1, 2, 4, 6]);
        formula.push(vec![3]);
        formula.push(vec![5]);
        match Certificate::try_from(formula).expect("panic!") {
            Certificate::UNSAT => {
                assert_eq!(true, true);
            }
            Certificate::SAT(_) => {
                panic!("This formula is not satifiable");
            }
        }
    }

    #[test]
    fn tiny_set_verification() {
        for (i, tww) in [1, 2, 0, 0, 3, 0, 2, 4, 1, 2].into_iter().enumerate() {
            if i == 4 {
                continue; // too slow
            }

            let filename = format!("instances/tiny/tiny{:>03}.gr", i + 1);
            let reader = File::open(filename.clone())
                .unwrap_or_else(|_| panic!("Cannot open file {}", &filename));
            let buf_reader = BufReader::new(reader);

            let pace_reader =
                PaceReader::try_new(buf_reader).expect("Could not construct PaceReader");

            let mut graph = AdjArray::new(pace_reader.number_of_nodes());
            graph.add_edges(pace_reader, EdgeColor::Black);

            let mut solver = TwinWidthSatEncoding::new(&graph);
            if let Some((size, _)) = solver.solve_varisat(tww) {
                assert_eq!(size, tww as u32);
            } else {
                panic!("Could not verify solver")
            }

            println!("{} verified.", filename);
        }
    }
}
