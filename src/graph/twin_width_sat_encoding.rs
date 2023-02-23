//use crate::graph::{AdjacencyList, GraphEdgeOrder, GraphEdgeEditing};
use cat_solver::Solver;
use splr::Certificate;
use varisat::{CnfFormula, ExtendFormula, Lit};

use crate::prelude::*;
use core::fmt::Debug;
use std::cmp::Ordering;

pub struct TwinWidthSatEncoding<G> {
    graph: G,
    edges: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>>>,
    ord: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>>,
    merge: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>>,
    variable_id: i32,

    //Map node id to node index
    graph_mapping: fxhash::FxHashMap<u32, u32>,
    graph_reverse_mapping: fxhash::FxHashMap<u32, u32>,

    assumptions_enabled: bool,
    complement_graph: bool,
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

    pub fn enable_assumptions(&mut self) {
        self.assumptions_enabled = true;
    }

    pub fn new_complement_graph(graph: &G) -> Self {
        let mut encoding = TwinWidthSatEncoding::<G>::new(graph);
        encoding.complement_graph = true;
        encoding
    }

    pub fn new(graph: &G) -> Self {
        let mut variable_id = 1;

        let mut ord: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>> =
            fxhash::FxHashMap::default();
        let mut merge: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>> =
            fxhash::FxHashMap::default();
        let mut edges: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>>> =
            fxhash::FxHashMap::default();

        let mut graph_mapping = fxhash::FxHashMap::default();
        let mut graph_reverse_mapping = fxhash::FxHashMap::default();

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
            for j in i + 1..graph_mapping.len() {
                for k in 0..graph_mapping.len() {
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
            graph_mapping,
            graph_reverse_mapping,
            assumptions_enabled: false,
            complement_graph: false,
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
    pub fn cardinality_naive_at_most_1(
        id_counter: &mut i32,
        vars: &Vec<i32>,
        formula: &mut Vec<Vec<i32>>,
    ) {
        // Encode at least one
        if vars.len() <= 1 {
            return;
        }

        let restrictions: Vec<i32> = (0..vars.len())
            .map(|_| {
                let id = *id_counter;
                *id_counter += 1;
                id
            })
            .collect();

        formula.push(vec![-vars[0], restrictions[0]]);
        formula.push(vec![
            -vars[vars.len() - 1],
            -restrictions[restrictions.len() - 2],
        ]);
        for i in 1..vars.len() - 1 {
            formula.push(vec![-vars[i], restrictions[i]]);
            formula.push(vec![-restrictions[i - 1], restrictions[i]]);
            formula.push(vec![-vars[i], -restrictions[i - 1]]);
        }
    }

    pub fn cardinality_at_most_k_sequential_encoding(
        id_counter: &mut i32,
        vars: &Vec<i32>,
        upper_bound: u32,
        formula: &mut Vec<Vec<i32>>,
    ) {
        // Based on the description of the sequential encoding from here https://www.carstensinz.de/papers/CP-2005.pdf
        if vars.len() <= upper_bound as usize {
            // If we have fewer variables than our upper bound then this is trivially true
            return;
        } else if upper_bound == 0 {
            for x in vars.iter() {
                formula.push(vec![-*x]);
            }
            return;
        } else if upper_bound == 1 {
            return TwinWidthSatEncoding::<G>::cardinality_naive_at_most_1(
                id_counter, vars, formula,
            );
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

        for j in 2..(upper_bound + 1) {
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

    #[inline]
    pub fn cardinality_naive_at_most_1_encode_assumption(
        id_counter: &mut i32,
        vars: &Vec<i32>,
        assumption: i32,
        formula: &mut Vec<Vec<i32>>,
    ) {
        // Encode at least one
        if vars.len() <= 1 {
            return;
        }

        let restrictions: Vec<i32> = (0..vars.len())
            .map(|_| {
                let id = *id_counter;
                *id_counter += 1;
                id
            })
            .collect();

        formula.push(vec![-vars[0], restrictions[0], assumption]);
        formula.push(vec![
            -vars[vars.len() - 1],
            -restrictions[restrictions.len() - 2],
            assumption,
        ]);
        for i in 1..vars.len() - 1 {
            formula.push(vec![-vars[i], restrictions[i], assumption]);
            formula.push(vec![-restrictions[i - 1], restrictions[i], assumption]);
            formula.push(vec![-vars[i], -restrictions[i - 1], assumption]);
        }
    }

    pub fn cardinality_at_most_k_sequential_encoding_encode_assumption(
        id_counter: &mut i32,
        vars: &Vec<i32>,
        upper_bound: u32,
        assumption: i32,
        formula: &mut Vec<Vec<i32>>,
    ) {
        // Based on the description of the sequential encoding from here https://www.carstensinz.de/papers/CP-2005.pdf
        if upper_bound == 0 {
            for x in vars.iter() {
                formula.push(vec![-*x, assumption]);
            }
            return;
        } else if upper_bound == 1 {
            return TwinWidthSatEncoding::<G>::cardinality_naive_at_most_1_encode_assumption(
                id_counter, vars, assumption, formula,
            );
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
            assumption,
        ]);

        for j in 2..(upper_bound + 1) {
            // NOT
            formula.push(vec![
                -*restrictions.get(&1).unwrap().get(&j).unwrap(),
                assumption,
            ]);
        }

        for i in 2..vars.len() {
            formula.push(vec![
                -vars[i - 1],
                *restrictions.get(&(i as u32)).unwrap().get(&1).unwrap(),
                assumption,
            ]);
            formula.push(vec![
                -*restrictions.get(&(i as u32 - 1)).unwrap().get(&1).unwrap(),
                *restrictions.get(&(i as u32)).unwrap().get(&1).unwrap(),
                assumption,
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
                    assumption,
                ]);
                formula.push(vec![
                    -*restrictions
                        .get(&((i - 1) as u32))
                        .unwrap()
                        .get(&j)
                        .unwrap(),
                    *restrictions.get(&(i as u32)).unwrap().get(&j).unwrap(),
                    assumption,
                ]);
            }
            formula.push(vec![
                -vars[i - 1],
                -*restrictions
                    .get(&(i as u32 - 1))
                    .unwrap()
                    .get(&(upper_bound))
                    .unwrap(),
                assumption,
            ]);
        }

        formula.push(vec![
            -vars[vars.len() - 1],
            -*restrictions
                .get(&((vars.len() - 1) as u32))
                .unwrap()
                .get(&(upper_bound))
                .unwrap(),
            assumption,
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
                TwinWidthSatEncoding::<G>::cardinality_naive_at_most_1(
                    &mut self.variable_id,
                    &current_group,
                    formula,
                );
                TwinWidthSatEncoding::<G>::cardinality_at_least_1(current_group, formula);
            } else {
                commands.push(current_group[0]);
            }
        }
        if commands.len() < 2 * m as usize {
            TwinWidthSatEncoding::<G>::cardinality_naive_at_most_1(
                &mut self.variable_id,
                &commands,
                formula,
            );
        } else {
            self.amo_commander(commands, m, formula);
        }
    }

    pub fn encode_reds(&mut self, formula: &mut Vec<Vec<i32>>) {
        let mut auxillarys_variables: fxhash::FxHashMap<u32, fxhash::FxHashMap<u32, i32>> =
            fxhash::FxHashMap::default();

        for i in 0..(self.graph_mapping.len() as u32) {
            auxillarys_variables.insert(i, fxhash::FxHashMap::default());
            for j in (i + 1)..(self.graph_mapping.len() as u32) {
                if j == i {
                    continue;
                }

                let aux_var = self.add_new_variable();
                auxillarys_variables.get_mut(&i).unwrap().insert(j, aux_var);

                for k in 0..(self.graph_mapping.len() as u32) {
                    if k == j || i == k {
                        continue;
                    }
                    formula.push(vec![-self.tedge(k, i, j), aux_var]);
                }
            }
        }

        for i in 0..(self.graph_mapping.len() as u32) {
            for j in 0..(self.graph_mapping.len() as u32) {
                if i == j {
                    continue;
                }

                for k in 0..(self.graph_mapping.len() as u32) {
                    if j == k || i == k {
                        continue;
                    }

                    for m in k + 1..(self.graph_mapping.len() as u32) {
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

        for i in 0..(self.graph_mapping.len() as u32) {
            for j in i + 1..(self.graph_mapping.len() as u32) {
                if i == j {
                    continue;
                }

                for k in 0..(self.graph_mapping.len() as u32) {
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

        let mut ordering: Vec<u32> = (0..self.graph_mapping.len() as u32).collect();
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

        for i in 0..(self.graph_mapping.len() as u32) {
            for y in i + 1..(self.graph_mapping.len() as u32) {
                let id = self.merge.get(&i).unwrap().get(&y).unwrap();
                if *decoded.get(id).unwrap() {
                    merge.insert(i, y);
                }
            }
        }

        ordering.sort_by(find_order);

        for i in ordering[0..ordering.len() - 1].iter() {
            contraction_sequence.merge_node_into(
                *self.graph_reverse_mapping.get(i).unwrap(),
                *self
                    .graph_reverse_mapping
                    .get(merge.get(i).unwrap())
                    .unwrap(),
            );
        }

        contraction_sequence
    }

    // Please note that this will run the sat solver for every integer <= upper bound until the solution becomes unsatisfiable
    pub fn solve(&mut self, mut ub: u32) -> Option<(u32, ContractionSequence)> {
        let mut last_valid_solution = None;
        let mut last_valid_bound = ub;

        loop {
            let (encoding, _) = self.encode(ub);

            match Certificate::try_from(encoding).expect("panic!") {
                Certificate::UNSAT => {
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
    pub fn solve_cadical(&mut self, mut ub: u32) -> Option<(u32, ContractionSequence)> {
        let mut last_valid_solution = None;
        let mut last_valid_bound = ub;

        // Enable iterative solving
        self.enable_assumptions();
        let (encoding, assumptions) = self.encode(ub);

        let mut cadical: cadical::Solver<cadical::Timeout> = cadical::Solver::default();

        let mut mapping: fxhash::FxHashSet<i32> = fxhash::FxHashSet::default();
        for x in encoding.into_iter() {
            x.iter().for_each(|v| {
                mapping.insert(v.abs());
            });
            cadical.add_clause(x);
        }

        loop {
            cadical.set_limit("preprocessing", 3).unwrap();
            cadical.set_limit("localsearch", 3).unwrap();

            // Make the upper bound which is currently active sharp by assuming only this upper bound
            if let Some(solved) =
                cadical.solve_with(assumptions.iter().enumerate().map(|(i, x)| {
                    if i as u32 == ub {
                        -*x
                    } else {
                        *x
                    }
                }))
            {
                if solved {
                    let mut solution = Vec::new();
                    for x in mapping.iter() {
                        match cadical.value(*x) {
                            Some(true) => {
                                solution.push(*x);
                            }
                            Some(false) => {
                                solution.push(-*x);
                            }
                            None => {
                                solution.push(*x);
                            }
                        }
                    }
                    if ub == 0 {
                        let seq = self.decode_sat_solution(solution);
                        return Some((ub, seq));
                    }
                    last_valid_bound = ub;
                    last_valid_solution = Some(solution);
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

    // Please note that this will run the sat solver for every integer <= upper bound until the solution becomes unsatisfiable
    pub fn solve_varisat(&mut self, mut ub: u32) -> Option<(u32, ContractionSequence)> {
        let mut last_valid_solution = None;
        let mut last_valid_bound = ub;
        self.enable_assumptions();
        let (encoding, assumptions) = self.encode(ub);
        let mut cnf = CnfFormula::new();

        let mut mapping: fxhash::FxHashMap<i32, Lit> = fxhash::FxHashMap::default();
        let mut reversed_mapping: fxhash::FxHashMap<usize, i32> = fxhash::FxHashMap::default();
        for clauses in encoding.iter() {
            let literals: Vec<Lit> = clauses
                .iter()
                .map(|x| {
                    let absolute = x.abs();
                    if let Some(exists) = mapping.get(&absolute) {
                        if *x < 0 {
                            exists.var().negative()
                        } else {
                            exists.var().positive()
                        }
                    } else {
                        let literal = cnf.new_lit();
                        mapping.insert(absolute, literal);
                        reversed_mapping.insert(literal.index(), x.abs());
                        if *x < 0 {
                            literal.var().negative()
                        } else {
                            literal.var().positive()
                        }
                    }
                })
                .collect();
            cnf.add_clause(&literals);
        }

        let mut solver = varisat::Solver::new();
        solver.add_formula(&cnf);

        loop {
            // Make the upper bound which is currently active sharp by assuming only this upper bound
            let assumptions: Vec<Lit> = assumptions
                .iter()
                .enumerate()
                .map(|(x, i)| {
                    if x as u32 == ub {
                        mapping.get(i).unwrap().var().negative()
                    } else {
                        mapping.get(i).unwrap().var().positive()
                    }
                })
                .collect();
            solver.assume(&assumptions);
            if let Ok(solved) = solver.solve() {
                if solved {
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

    // Please note that this will run the sat solver for every integer <= upper bound until the solution becomes unsatisfiable
    pub fn solve_kissat(&mut self, mut ub: u32) -> Option<(u32, ContractionSequence)> {
        let mut last_valid_solution = None;
        let mut last_valid_bound = ub;
        loop {
            let (encoding, _) = self.encode(ub);
            println!("Solving for width {ub}");

            let mut kissat_solver = Solver::new();

            let mut mapping: fxhash::FxHashSet<i32> = fxhash::FxHashSet::default();
            for x in encoding.into_iter() {
                x.iter().for_each(|v| {
                    mapping.insert(v.abs());
                });
                kissat_solver.add_clause(x.into_iter());
            }

            // Max the limits
            kissat_solver.set_limit("conflicts", 0x40000000).unwrap();
            kissat_solver.set_limit("decisions", 0x40000000).unwrap();

            if let Some(solved) = kissat_solver.solve() {
                if solved {
                    //println!("Found solution in {}ms width {}", solver_time.elapsed().as_millis(),ub);
                    let mut solution = Vec::new();
                    for x in mapping.into_iter() {
                        match kissat_solver.value(x) {
                            Some(true) => {
                                solution.push(x);
                            }
                            Some(false) => {
                                solution.push(-x);
                            }
                            None => {
                                solution.push(x);
                            }
                        }
                    }
                    if ub == 0 {
                        let seq = self.decode_sat_solution(solution);
                        return Some((ub, seq));
                    }
                    last_valid_bound = ub;
                    last_valid_solution = Some(solution);
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

    pub fn encode(&mut self, at_most_d: u32) -> (Vec<Vec<i32>>, Vec<i32>) {
        let mut formula = Vec::new();
        for i in 0..(self.graph_mapping.len() as u32) {
            for j in 0..(self.graph_mapping.len() as u32) {
                if i == j {
                    continue;
                }

                for k in 0..(self.graph_mapping.len() as u32) {
                    if i == k || k == j {
                        continue;
                    }
                    formula.push(vec![-self.tord(i, j), -self.tord(j, k), self.tord(i, k)]);
                }
            }
        }
        for i in 0..(self.graph_mapping.len() as u32) {
            for j in (i + 1)..(self.graph_mapping.len() as u32) {
                formula.push(vec![-self.get_merge(i, j), self.tord(i, j)])
            }
        }

        for i in 0..((self.graph_mapping.len() as u32) - 1) {
            let mut atleast_encoded = Vec::new();
            let mut amocommander_encoded = Vec::new();
            for j in (i + 1)..(self.graph_mapping.len() as u32) {
                let var = *self.merge.get(&i).unwrap().get(&j).unwrap();
                atleast_encoded.push(var);
                amocommander_encoded.push(var);
            }
            TwinWidthSatEncoding::<G>::cardinality_at_least_1(atleast_encoded, &mut formula);
            self.amo_commander(amocommander_encoded, 2, &mut formula);
        }

        for (skip_len, node_id) in self
            .graph
            .vertices()
            .filter(|x| self.graph.degree_of(*x) > 0)
            .enumerate()
        {
            let mut neighbors_i = self.graph.neighbors_of_as_bitset(node_id);
            if self.complement_graph {
                neighbors_i.not();
                for x in neighbors_i.clone().iter() {
                    if self.graph.degree_of(x) < 1 {
                        neighbors_i.unset_bit(x);
                    }
                }
                neighbors_i.unset_bit(node_id);
            }
            for node_j_id in self
                .graph
                .vertices()
                .filter(|x| self.graph.degree_of(*x) > 0)
                .skip(skip_len + 1)
            {
                let mut neighbors_j = self.graph.neighbors_of_as_bitset(node_j_id);
                if self.complement_graph {
                    neighbors_j.not();
                    for x in neighbors_j.clone().iter() {
                        if self.graph.degree_of(x) < 1 {
                            neighbors_j.unset_bit(x);
                        }
                    }
                }
                neighbors_j.unset_bit(node_id);

                neighbors_j.xor(&neighbors_i);
                neighbors_j.unset_bit(node_j_id);

                for k_id in neighbors_j.iter() {
                    formula.push(vec![
                        -self.get_merge(
                            *self.graph_mapping.get(&node_id).unwrap(),
                            *self.graph_mapping.get(&node_j_id).unwrap(),
                        ),
                        -self.tord(
                            *self.graph_mapping.get(&node_id).unwrap(),
                            *self.graph_mapping.get(&k_id).unwrap(),
                        ),
                        self.tedge(
                            *self.graph_mapping.get(&node_id).unwrap(),
                            *self.graph_mapping.get(&node_j_id).unwrap(),
                            *self.graph_mapping.get(&k_id).unwrap(),
                        ),
                    ]);
                }
            }
        }

        self.encode_reds(&mut formula);

        let mut assumptions: Vec<i32> = (0..at_most_d + 1)
            .map(|_| self.add_new_variable())
            .collect();
        // Totalizer!
        for i in 0..(self.graph_mapping.len() as u32) {
            for x in 0..(self.graph_mapping.len() as u32) {
                let mut vars = Vec::new();
                for y in 0..(self.graph_mapping.len() as u32) {
                    if x == y {
                        continue;
                    }
                    vars.push(self.tedge(i, x, y));
                }

                if self.assumptions_enabled {
                    for d in 0..(at_most_d + 1) {
                        TwinWidthSatEncoding::<G>::cardinality_at_most_k_sequential_encoding_encode_assumption(
                            &mut self.variable_id,
                            &vars,
                            d,
                            assumptions[d as usize],
                            &mut formula,
                        );
                    }
                } else {
                    TwinWidthSatEncoding::<G>::cardinality_at_most_k_sequential_encoding(
                        &mut self.variable_id,
                        &vars,
                        at_most_d,
                        &mut formula,
                    );
                }
            }
        }

        if !self.assumptions_enabled {
            assumptions.clear();
        }

        //sb_ord
        for i in 0..(self.graph_mapping.len() as u32) - 1 {
            formula.push(vec![*self
                .ord
                .get(&i)
                .unwrap()
                .get(&((self.graph_mapping.len() as u32) - 1))
                .unwrap()]);
        }

        //sb_red
        for i in 0..(self.graph_mapping.len() as u32) {
            for j in (i + 1)..(self.graph_mapping.len() as u32) {
                if i == j {
                    continue;
                }

                for k in 0..(self.graph_mapping.len() as u32) {
                    if k == i || k == j {
                        continue;
                    }
                    formula.push(vec![-self.tord(j, i), -self.tedge(i, j, k)]);
                }
            }
        }

        (formula, assumptions)
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
        let vars = vec![1, -2, 3, 4, 5, 6];
        let mut id = 7;
        let mut formula = Vec::new();
        TwinWidthSatEncoding::<AdjArray>::cardinality_at_most_k_sequential_encoding(
            &mut id,
            &vars,
            2,
            &mut formula,
        );

        formula.push(vec![1, -2, 4, 6]);
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
                assert_eq!(size, tww);
            } else {
                panic!("Could not verify solver")
            }

            println!("{filename} verified.",);
        }
    }
}
