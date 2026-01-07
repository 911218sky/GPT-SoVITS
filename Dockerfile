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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

SHELL ["/bin/bash", "-c"]
WORKDIR /workspace/GPT-SoVITS

COPY Docker /workspace/GPT-SoVITS/Docker/
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true
RUN bash Docker/miniconda_install.sh

ENV PATH="/root/miniconda3/bin:$PATH"

COPY extra-req.txt requirements.txt install.sh /workspace/GPT-SoVITS/
RUN bash Docker/install_wrapper.sh

RUN conda clean -afy && \
    pip cache purge && \
    rm -rf /root/.cache/pip /root/.cache/huggingface && \
    find /root/miniconda3 -name "*.pyc" -delete && \
    find /root/miniconda3 -name "*.pyo" -delete && \
    find /root/miniconda3 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /root/miniconda3 -name "*.a" -delete 2>/dev/null || true && \
    rm -rf /root/miniconda3/pkgs/*

# Final Stage
FROM nvidia/cuda:${CUDA_VERSION}.0-runtime-ubuntu24.04

ARG CUDA_VERSION=12.6
ENV CUDA_VERSION=${CUDA_VERSION}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=builder /root/miniconda3 /root/miniconda3

ENV PATH="/root/miniconda3/bin:$PATH"
ENV PYTHONPATH="/workspace/GPT-SoVITS"

RUN /root/miniconda3/bin/conda init bash && \
    echo "conda activate base" >> ~/.bashrc

WORKDIR /workspace/GPT-SoVITS

COPY . /workspace/GPT-SoVITS

RUN chmod +x /workspace/GPT-SoVITS/Docker/entrypoint.sh

EXPOSE 9871 9872 9873 9874 9880

ENTRYPOINT ["/workspace/GPT-SoVITS/Docker/entrypoint.sh"]
