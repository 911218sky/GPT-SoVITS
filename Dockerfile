ARG CUDA_VERSION=12.6

FROM nvidia/cuda:${CUDA_VERSION}.0-runtime-ubuntu22.04

LABEL maintainer="sky1218"
LABEL version="V4"
LABEL description="Docker image for GPT-SoVITS"

ARG CUDA_VERSION=12.6

ENV CUDA_VERSION=${CUDA_VERSION}
ENV DEBIAN_FRONTEND=noninteractive

# Install bash and essential tools (nvidia/cuda runtime image is minimal)
# doxygen is needed for opencc to build from source
# software-properties-common is needed to add PPA for updated libstdc++
RUN apt-get update && apt-get install -y --no-install-recommends \
  bash \
  wget \
  ca-certificates \
  curl \
  unzip \
  doxygen \
  build-essential \
  software-properties-common \
  && rm -rf /var/lib/apt/lists/*

# Update libstdc++ to get GLIBCXX_3.4.32 (required by opencc)
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
  apt-get update && \
  apt-get install -y libstdc++6 && \
  rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace/GPT-SoVITS

COPY Docker /workspace/GPT-SoVITS/Docker/

ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

RUN bash Docker/miniconda_install.sh

ENV PATH="/root/miniconda3/bin:$PATH"

COPY extra-req.txt /workspace/GPT-SoVITS/

COPY requirements.txt /workspace/GPT-SoVITS/

COPY install.sh /workspace/GPT-SoVITS/

RUN bash Docker/install_wrapper.sh

EXPOSE 9871 9872 9873 9874 9880

ENV PYTHONPATH="/workspace/GPT-SoVITS"

RUN /root/miniconda3/bin/conda init bash && echo "conda activate base" >> ~/.bashrc

WORKDIR /workspace

RUN rm -rf /workspace/GPT-SoVITS

WORKDIR /workspace/GPT-SoVITS

COPY . /workspace/GPT-SoVITS

RUN chmod +x /workspace/GPT-SoVITS/Docker/entrypoint.sh

ENTRYPOINT ["/workspace/GPT-SoVITS/Docker/entrypoint.sh"]