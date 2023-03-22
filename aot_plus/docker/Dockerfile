ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
RUN rm -rf /etc/apt/sources.list.d/cuda.list
RUN rm -rf /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV PROJECT=aot
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ARG python=3.7
ENV PYTHON_VERSION=${python}

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Core tools
RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    docker.io \
    ffmpeg \
    git \
    htop \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    ninja-build \
    unzip \
    vim \
    wget \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install OpenSSH for MPI to communicate between containers
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install Pydata and other deps
RUN pip install easydict scipy numpy pyquaternion matplotlib jupyter h5py \
    awscli tqdm progress path.py pyyaml opencv-python pandas \
    numba cython scikit-learn moviepy imageio yacs Pillow scikit-image

# Settings for S3
RUN aws configure set default.s3.max_concurrent_requests 100 && \
    aws configure set default.s3.max_queue_size 10000

#pip install spatial-correlation-sampler

# Expose Port for jupyter (8888)
EXPOSE 8888

# create project workspace dir
WORKDIR /workspace/${PROJECT}

ENV PYTHONPATH="/workspace/${PROJECT}:$PYTHONPATH"
RUN git config --global --add safe.directory /workspace/${PROJECT}
