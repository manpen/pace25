#!/usr/bin/env bash 
set -e 
DOCKER_IMAGE=rust_on_ubuntu
CARGO_PROFILE=optil

docker build \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -t $DOCKER_IMAGE \
    rust_on_ubuntu

docker run --rm -it \
  -v "$(pwd)":/crate \
  -w /crate \
  -e RUSTFLAGS='-C target-cpu=haswell -C target-feature=+bmi2,+bmi1 -Zlocation-detail=none -Zfmt-debug=none' \
  $DOCKER_IMAGE \
  cargo build --profile $CARGO_PROFILE -F optil -Z unstable-options --artifact-dir target_docker --bin heuristic

ls -ahl target/target_docker/heuristic