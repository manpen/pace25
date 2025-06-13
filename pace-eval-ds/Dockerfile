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

RUN mkdir /build /solver
COPY Cargo.toml pace-eval-ds/build.sh /build/
COPY src /build/src
COPY stream-bitset /build/stream-bitset
RUN bash /build/build.sh


# working directory
WORKDIR /pace
COPY pace-eval-ds/src/eval.py eval.py
COPY pace-eval-ds/src/ds_verifier.py verifier.py

RUN mkdir -p /output && chmod 777 /output
CMD ["python3", "eval.py"]