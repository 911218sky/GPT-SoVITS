ARG CUDA_VERSION=12.6

# Build Stage
FROM nvidia/cuda:${CUDA_VERSION}.0-runtime-ubuntu24.04 AS builder

LABEL maintainer="sky1218"
LABEL version="V4"
LABEL description="Docker image for GPT-SoVITS"

ARG CUDA_VERSION=12.6
ENV CUDA_VERSION=${CUDA_VERSION}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    wget \
    ca-certificates \
    curl \
    unzip \
    doxygen \
    build-essential \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

SHELL ["/bin/bash", "-c"]
WORKDIR /workspace/GPT-SoVITS

COPY Docker /workspace/GPT-SoVITS/Docker/
RUN bash Docker/uv_install.sh

ENV PATH="/root/uv/env/bin:/root/uv/bin:/root/uv:$PATH"
ENV UV_PYTHON="/root/uv/env/bin/python"
ENV VIRTUAL_ENV="/root/uv/env"

COPY extra-req.txt requirements.txt install.sh /workspace/GPT-SoVITS/
RUN bash Docker/install_wrapper.sh

RUN uv cache clean && \
    rm -rf /root/.cache/pip /root/.cache/huggingface /root/.cache/uv && \
    find /root/uv/env -name "*.pyc" -delete && \
    find /root/uv/env -name "*.pyo" -delete && \
    find /root/uv/env -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /root/uv/env -name "*.a" -delete 2>/dev/null || true

# Final Stage
FROM nvidia/cuda:${CUDA_VERSION}.0-runtime-ubuntu24.04

ARG CUDA_VERSION=12.6
ENV CUDA_VERSION=${CUDA_VERSION}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=builder /root/uv /root/uv

ENV PATH="/root/uv/env/bin:/root/uv/bin:/root/uv:$PATH"
ENV UV_PYTHON="/root/uv/env/bin/python"
ENV VIRTUAL_ENV="/root/uv/env"
ENV PYTHONPATH="/workspace/GPT-SoVITS"

# Setup shell environment
RUN echo 'source /root/uv/etc/profile.d/uv.sh' >> ~/.bashrc

WORKDIR /workspace/GPT-SoVITS

COPY . /workspace/GPT-SoVITS

RUN chmod +x /workspace/GPT-SoVITS/Docker/entrypoint.sh

EXPOSE 9871 9872 9873 9874 9880

ENTRYPOINT ["/workspace/GPT-SoVITS/Docker/entrypoint.sh"]
