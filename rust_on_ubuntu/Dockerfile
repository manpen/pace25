FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    g++ \
    curl \
    ca-certificates \
    build-essential \
    libssl-dev \
    pkg-config \
    cmake \
    clang-19 \
    && rm -rf /var/lib/apt/lists/*

ARG USER_ID=1000
ARG GROUP_ID=1000    
RUN userdel ubuntu && \
    groupadd -g ${GROUP_ID} rustuser && \
    useradd -m -u ${USER_ID} -g rustuser -s /bin/bash rustuser && \
    echo "rustuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER rustuser
WORKDIR /home/rustuser
ENV PATH="/home/rustuser/.cargo/bin:${PATH}"

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly --profile minimal

WORKDIR /crate
CMD [ "bash" ]
