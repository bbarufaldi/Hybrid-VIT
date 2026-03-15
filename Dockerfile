FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies including python 3.10
RUN apt-get update && apt-get install -y \
    libgdcm-dev \
    python3 \
    python3-pip \
    git \
    wget \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN git clone https://github.com/NVIDIA/cuda-samples.git

ENV PATH="${PATH}:app/build:app/cuda-samples"

RUN pip3 install --no-cache-dir -r requirements.txt
