ARG CUDA="10.2"
ARG CUDNN="8"
ARG PYTHON="3.8.5"
ARG PYTORCH="1.8"

# TODO: conda installs its own version of cuda
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

ENV CUDA_PATH /usr/local/cuda
ENV CUDA_ROOT /usr/local/cuda/bin
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64

RUN apt-get -qq update
# libsm6 and libxext6 are needed for cv2
RUN apt-get update && apt-get install -y libxext6 libsm6 libxrender1 build-essential sudo \
    libgl1-mesa-glx git wget rsync tmux nano dcmtk fftw3-dev liblapacke-dev libpng-dev \
    libglib2.0-0 libopenblas-dev jq && \
  rm -rf /var/lib/apt/lists/*
RUN ldconfig

# Make a user


RUN adduser --disabled-password --gecos '' runner
RUN adduser runner sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER runner


WORKDIR /tmp
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH "/home/runner/miniconda3/bin:$PATH:$CUDA_ROOT"

# Setup python packages
RUN conda update -n base conda -yq
RUN conda install python=${PYTHON}
RUN conda install pip
RUN conda install pytorch=${PYTORCH} torchvision cudatoolkit=${CUDA} -c pytorch

RUN pip install loguru
RUN pip install monai
RUN pip install scikit-image
RUN pip install opencv-python
RUN pip install memcnn
RUN pip install omegaconf
RUN pip install tensorboard
RUN pip install wandb
RUN pip install pandas


# Create runner for input and output
USER root
RUN mkdir /runner
RUN mkdir /runner/ganslate && chmod 777 /runner/ganslate
RUN mkdir /runner/data
RUN mkdir /runner/output && chmod 777 /runner/output
USER runner

WORKDIR /runner

ENTRYPOINT $0 $@
