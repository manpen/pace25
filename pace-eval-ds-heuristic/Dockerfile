FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Basic Setup
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    time \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

#################################################
# Install deps of solver & build it
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    g++ build-essential pkg-config cmake clang-14 git\
    curl ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly --profile minimal

RUN mkdir /build
COPY Cargo.toml /build/
COPY src /build/src
COPY stream-bitset /build/stream-bitset
RUN . /root/.cargo/env && \
    cd /build && ls -ahl && \
    cargo b --profile optil -F optil --bin heuristic && \
    mkdir /solver && cp /build/target/optil/heuristic /solver && \
    cd / && rm -rf /build
####################################################

# working directory
WORKDIR /pace
COPY pace-eval-ds-heuristic/src/eval.py eval.py
COPY pace-eval-ds-heuristic/src/ds_verifier.py verifier.py

RUN mkdir -p /output && chmod 777 /output
CMD ["python3", "eval.py"]