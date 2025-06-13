#!/bin/bash
set -e 
set -x
. /root/.cargo/env

cd /build

# pull maxsat solver
curl -o solver.zip https://maxsat-evaluations.github.io/2024/mse24-solver-src/exact/unweighted/UWrMaxSat-SCIP-MaxPre.zip
unzip solver.zip UWrMaxSat-SCIP-MaxPre/bin/uwrmaxsat
mv UWrMaxSat-SCIP-MaxPre/bin/uwrmaxsat /solver
rm -rf solver.zip UWrMaxSat-SCIP-MaxPre

curl -o solver.zip https://maxsat-evaluations.github.io/2024/mse24-solver-src/exact/unweighted/EvalMaxSAT_2024.zip
unzip solver.zip EvalMaxSAT_2024/bin/EvalMaxSAT
mv EvalMaxSAT_2024/bin/EvalMaxSAT /solver/EvalMaxSAT_bin
rm -rf solver.zip EvalMaxSAT_2024

# build our solver
cd /build
cargo b --profile optil -F optil
cp /build/target/optil/{heuristic,highs_*,exact} /solver 
rm -rf /build

