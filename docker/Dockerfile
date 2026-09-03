FROM ubuntu:22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

# These get overwritten by values in compose.yaml
ARG PYTHON_VERSION=3.12

COPY ./docker/setup .

RUN apt-get update && apt-get install -y \
    gcc g++ gfortran git patch wget pkg-config liblapack-dev libmetis-dev cmake curl \
    apt-utils gnupg2 vim libsm6 libxext6 libxrender-dev nano screen tmux htop \
    openssh-client unzip software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    coinor-libipopt-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python version as the default 'python' and 'python3'
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Install and upgrade pip for python version
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

RUN apt-get update

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && ln -fs /usr/share/zoneinfo/US/Pacific /etc/localtime && apt-get install -y tzdata && dpkg-reconfigure --frontend noninteractive tzdata

# Install python packages (casadi, jax, etc.)
RUN python3 -m pip install -U pip && python3 -m pip install -U opencv-python numpy Pillow matplotlib scipy sympy matplotlib natsort tqdm h5py seaborn ipython jupyter jupyterlab marimo jax casadi cvxpy pyscipopt tqdm astroquery mujoco mujoco-mjx warp-lang equinox nbconvert[webpdf]

# Required for exporting marimo notebook as pdf
RUN playwright install --with-deps chromium

# Install moreau
ARG MOREAU_LICENSE_KEY
ENV MOREAU_LICENSE_KEY=${MOREAU_LICENSE_KEY}
RUN python3.12 -m pip install moreau --extra-index-url https://kkMeu-DhcjKJQMlfKzSqkxXqjEQndqjgE:@pypi.fury.io/optimalintellect/

# Install interactive viewer + virtual display
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libgl1-mesa-dri libegl1-mesa libosmesa6-dev \
    libglfw3 libglew-dev libxrandr2 libxinerama1 libxcursor1 libxi6 \
    xvfb x11vnc novnc websockify

# Install ipopt
RUN apt-get install -y coinor-libipopt-dev -y

# Install acados
RUN /install_acados.sh

ENV ACADOS_SOURCE_DIR="/acados"
ENV ACADOS_PATH="/acados"
ENV LD_LIBRARY_PATH="/acados/lib:${LD_LIBRARY_PATH}"

# Install tera renderer
RUN /install_tera_renderer.sh

WORKDIR /workspace

CMD ["/bin/bash"]
