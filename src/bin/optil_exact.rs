use std::io::{stdin, stdout};
use tww::prelude::*;
type Graph = AdjMatrix;

fn main() -> std::io::Result<()> {
    let graph = {
        let stdin = stdin();
        Graph::try_read_pace(stdin.lock())?
    };

    let (ub, heur_seq) = heuristic_solve(&graph);

    let (_sol_size, mut sol) =
        branch_and_bound::BranchAndBound::new_with_bounds(graph.clone(), 0, ub.saturating_sub(1))
            .solve()
            .unwrap_or((ub, heur_seq));

    sol.add_unmerged_singletons(&graph).unwrap();

    sol.pace_writer(stdout().lock())?;

    Ok(())
}
