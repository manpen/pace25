#!/usr/bin/env bash 
set -e 
export RUSTFLAGS='-C target-feature=+crt-static -C target-cpu=haswell -C target-feature=+bmi2,+bmi1 -Zlocation-detail=none -Zfmt-debug=none' 
BUILD_CMD="cargo build --target=x86_64-unknown-linux-musl --profile optil -F optil" 

$BUILD_CMD --bin heuristic

ls -ahl target/x86_64-unknown-linux-musl/optil/heuristic