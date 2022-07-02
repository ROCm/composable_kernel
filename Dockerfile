FROM ubuntu:18.04

ARG ROCMVERSION=5.1
ARG OSDB_BKC_VERSION

RUN set -xe

ARG BUILD_THREADS=8
ARG DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/.apt_$ROCMVERSION/
# Add rocm repository
RUN apt-get update
RUN apt-get install -y wget gnupg
RUN wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN sh -c "echo deb [arch=amd64] $DEB_ROCM_REPO ubuntu main > /etc/apt/sources.list.d/rocm.list"
RUN wget --no-check-certificate -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN sh -c "echo deb https://apt.kitware.com/ubuntu/ bionic main | tee -a /etc/apt/sources.list"

# ADD requirements.txt requirements.txt
# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    build-essential \
    cmake-data=3.15.1-0kitware1 \
    cmake=3.15.1-0kitware1 \
    curl \
    g++ \
    gdb \
    git \
    hip-rocclr \
    jq \
    libelf-dev \
    libncurses5-dev \
    libnuma-dev \
    libpthread-stubs0-dev \
    llvm-amdgpu \
    pkg-config \
    python \
    python3.8 \
    python-dev \
    python3-dev \
    python-pip \
    python3-pip \
    software-properties-common \
    wget \
    rocm-dev \
    rocm-device-libs \
    rocm-cmake \
    vim \
    zlib1g-dev \
    openssh-server \
    clang-format-10 \
    kmod && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup ubsan environment to printstacktrace
RUN ln -s /usr/bin/llvm-symbolizer-3.8 /usr/local/bin/llvm-symbolizer
ENV UBSAN_OPTIONS=print_stacktrace=1

# Install an init system
RUN wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb
RUN dpkg -i dumb-init_*.deb && rm dumb-init_*.deb

# Install cget
RUN pip install cget

# Install rclone
RUN pip install https://github.com/pfultz2/rclone/archive/master.tar.gz

ARG PREFIX=/opt/rocm
# Install dependencies
RUN cget install pfultz2/rocm-recipes
# Install rbuild
RUN pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/6d78a0553babdaea8d2da5de15cbda7e869594b8.tar.gz
# Install packages for processing the performance results
RUN pip3 install --upgrade pip
RUN pip3 install sqlalchemy
RUN pip3 install pymysql
RUN pip3 install pandas
RUN pip3 install setuptools-rust
RUN pip3 install sshtunnel
# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ADD rbuild.ini /rbuild.ini
ADD dev-requirements.txt dev-requirements.txt
RUN rbuild prepare -s develop -d $PREFIX
RUN groupadd -f render

# Install the new rocm-cmake version
RUN git clone -b master https://github.com/RadeonOpenCompute/rocm-cmake.git  && \
  cd rocm-cmake && mkdir build && cd build && \
  cmake  .. && cmake --build . && cmake --build . --target install
