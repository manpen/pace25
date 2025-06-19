FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Basic Setup
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    time \
    util-linux \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install solver related software
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    g++ build-essential pkg-config cmake clang-14 git\
    curl ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly --profile minimal

RUN mkdir /solver

# Fetch external solvers
RUN mkdir /build && cd /build && \
    curl -o solver.zip https://maxsat-evaluations.github.io/2024/mse24-solver-src/exact/unweighted/UWrMaxSat-SCIP-MaxPre.zip && \
    unzip solver.zip UWrMaxSat-SCIP-MaxPre/bin/uwrmaxsat && \
    mv UWrMaxSat-SCIP-MaxPre/bin/uwrmaxsat /solver && \
    rm -rf solver.zip UWrMaxSat-SCIP-MaxPre && \
    curl -o solver.zip https://maxsat-evaluations.github.io/2024/mse24-solver-src/exact/unweighted/EvalMaxSAT_2024.zip && \
    unzip solver.zip EvalMaxSAT_2024/bin/EvalMaxSAT && \
    mv EvalMaxSAT_2024/bin/EvalMaxSAT /solver/EvalMaxSAT_bin && \
    cd / && rm -rf /build

# Build our solvers
COPY Cargo.toml /build/
COPY src /build/src
COPY stream-bitset /build/stream-bitset
RUN . /root/.cargo/env && \
    mkdir -p /build && cd /build && \
    RUSTFLAGS="-Ctarget-cpu=native" cargo b --profile optil -F optil && \
    ls -ahl /build/target/optil/ && \
    cp /build/target/optil/heuristic   /solver && \
    cp /build/target/optil/highs_child /solver && \
    cp /build/target/optil/exact       /solver && \
    cd / && rm -rf /build

# working directory
WORKDIR /pace
COPY pace-eval-ds/src/eval.py eval.py
COPY pace-eval-ds/src/ds_verifier.py verifier.py

RUN mkdir -p /output && chmod 777 /output
CMD ["python3", "eval.py"]