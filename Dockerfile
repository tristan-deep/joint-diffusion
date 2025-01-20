## Dockerfile for GPU environment with Tensorflow and Pytorch
## Takes a good 10 minutes to install this, be patient
## Author: Tristan Stevens

# cuda image
# FROM ubuntu:22.04
FROM tensorflow/tensorflow:2.9.1-gpu

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Do not prompt for input when installing packages
ARG DEBIAN_FRONTEND=noninteractive

# Set pip cache directory
ENV PIP_CACHE_DIR=/tmp/pip_cache

# Install sudo
RUN apt-get update && apt-get install -y sudo

# Add non-root users
ARG BASE_UID=1000
ARG NUM_USERS=51

# Create users in a loop
RUN for i in $(seq 0 $NUM_USERS); do \
        USER_UID=$((BASE_UID + i)); \
        USERNAME="devcontainer$i"; \
        groupadd --gid $USER_UID $USERNAME && \
        useradd --uid $USER_UID --gid $USER_UID -m --shell /bin/bash $USERNAME && \
        echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
        chmod 0440 /etc/sudoers.d/$USERNAME; \
        echo "export PATH=\$PATH:/home/$USERNAME/.local/bin" >> /home/$USERNAME/.bashrc; \
    done

# Install python, pip, git, opencv dependencies, ffmpeg, imagemagick, and ssh keyscan github
RUN apt-get install -y python3 python3-pip git python3-tk python3-venv \
                       libsm6 libxext6 libxrender-dev libqt5gui5 \
                       ffmpeg imagemagick openssh-client \
                       texlive-latex-extra texlive-fonts-recommended dvipng cm-super && \
    python3 -m pip install pip -U && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /joint-diffusion
COPY . /joint-diffusion/

RUN pip install tensorflow-addons tensorflow-gan tensorflow-datasets tensorflow-estimator
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu113 torch==1.12.1+cu113 torchvision torchmetrics ax-platform
RUN pip install -r requirements/requirements.txt
RUN pip install -U urllib3 requests
