# PACE25: PaceYourself Solvers

## Evaluation
We adopted the [official Docker Containers](https://github.com/MarioGrobler/PACE2025-docker) and hope that evaluating the
solver boils down to running `docker-compose build` and `docker-compose up`. Observe that there's only one Dominating Set
image containing both the heuristic and exact solver.

## Building manually
Most of the Solver is implemented in Rust (nightly) and relies on Cargo:

```bash
# Install Rust
curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly --profile minimal
. ~/.cargo/env
cargo b --profile optil -F optil

# The binaries are now stored in target/optil/{heuristic,exact}
```

The `-F optil` feature disables logging and disables command line parsing. If you are not building for PACE we recommend to
use `cargo b -r` instead (this places the binaries into `target/release`). This will give you a lot of command line arguments. 

## External Solvers
The exact solver depends on two external MaxSAT solvers:
 - https://maxsat-evaluations.github.io/2024/mse24-solver-src/exact/unweighted/UWrMaxSat-SCIP-MaxPre.zip
 - https://maxsat-evaluations.github.io/2024/mse24-solver-src/exact/unweighted/EvalMaxSAT_2024.zip

The linked archives contain Linux binaries that seem to work on most x64 system directly; but compiling from scratch is also
an option. Please put them into either the current working director or the same directory as our solvers. Use the names
`EvalMaxSAT_bin` and `uwrmaxsat`.

## Team
- Lukas Geis (Goethe University Frankfurt)
- Alexander Leonhardt (Goethe University Frankfurt)
- Johannes Meintrup (Technische Hochschule Mittelhessen)
- Ulrich Meyer (Goethe University Frankfurt)
- Manuel Penschuck (Goethe University Frankfurt)
