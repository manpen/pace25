#!/usr/bin/env bash 
set -e 
export RUSTFLAGS='-C target-feature=+crt-static -C target-cpu=haswell -C target-feature=+bmi2,+bmi1' 
BUILD_CMD="cargo build --target x86_64-unknown-linux-gnu --profile optil" 

$BUILD_CMD --bin optil_exact

ls -ahl target/x86_64-unknown-linux-gnu/optil/optil_exact