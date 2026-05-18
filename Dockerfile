
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG ROCMVERSION=7.1.1
ARG DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/.apt_$ROCMVERSION/

# TheRock nightly tarball configuration.
# By default, discovers the latest tarball from the nightlies index.
# Manual overrides:
#   Pin a specific tarball:
#     --build-arg TARBALL_URL=https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-linux-multiarch-7.13.0a20260430.tar.gz
#   Change the arch variant (default: multiarch):
#     --build-arg TARBALL_PATTERN=therock-dist-linux-gfx90a
#     --build-arg TARBALL_PATTERN=therock-dist-linux-gfx94X-dcgpu
ARG TARBALL_URL=""
ARG TARBALL_BASE=https://rocm.nightlies.amd.com/tarball-multi-arch
ARG TARBALL_PATTERN=therock-dist-linux-multiarch

ARG compiler_version=""
ARG compiler_commit=""
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=$PATH:/opt/rocm/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
ENV HIP_PLATFORM=amd

# Add rocm repository
RUN set -xe && \
    apt-get update && apt-get install -y --allow-unauthenticated apt-utils wget gnupg2 curl

RUN if [ "$compiler_version" = "therock" ]; then \
        rm -rf /opt/rocm && mkdir /opt/rocm && \
        if [ -n "$TARBALL_URL" ]; then \
            echo "Using provided TARBALL_URL: $TARBALL_URL" ; \
        else \
            echo "Discovering latest tarball from $TARBALL_BASE..." && \
            TARBALL_URL="${TARBALL_BASE}/$(curl -sL "${TARBALL_BASE}/" \
                | grep -oP '"name":\s*"\K'"${TARBALL_PATTERN}"'-[^"]+\.tar\.gz' \
                | sort -V | tail -1)" && \
            echo "Found: $TARBALL_URL" ; \
        fi && \
        wget -q -O /tmp/rocm.tar.gz "$TARBALL_URL" && \
        tar -xzf /tmp/rocm.tar.gz -C /opt/rocm --strip-components=1 && \
        rm /tmp/rocm.tar.gz ; \
    else echo "using the release compiler" && \
        wget https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/noble/amdgpu-install_7.1.1.70101-1_all.deb && \
        apt install ./amdgpu-install_7.1.1.70101-1_all.deb -y && \
        apt update && \
        apt install python3-setuptools python3-wheel -y && \
        apt install rocm-dev -y; \
    fi
    
# Install SCCACHE
ENV SCCACHE_VERSION="0.14.0"
ENV SCCACHE_INSTALL_LOCATION=/usr/local/.cargo/bin
ENV PATH=$PATH:${SCCACHE_INSTALL_LOCATION}
RUN set -x && \
    mkdir -p ${SCCACHE_INSTALL_LOCATION} && \
    wget -qO sccache.tar.gz https://github.com/mozilla/sccache/releases/download/v$SCCACHE_VERSION/sccache-v$SCCACHE_VERSION-x86_64-unknown-linux-musl.tar.gz && \
    tar -xzf sccache.tar.gz --strip-components=1 -C ${SCCACHE_INSTALL_LOCATION} && \
    chmod +x ${SCCACHE_INSTALL_LOCATION}/sccache

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    build-essential \
    cmake \
    git \
    iputils-ping \
    jq \
    libelf-dev \
    libnuma-dev \
    libpthread-stubs0-dev \
    mpich \
    net-tools \
    pkg-config \
    python3-full \
    python3-pip \
    redis \
    sshpass \
    stunnel \
    software-properties-common \
    vim \
    nano \
    zlib1g-dev \
    zip \
    libzstd-dev \
    openssh-server \
    clang-format-18 \
    kmod && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf amdgpu-install* && \
#Install latest ccache
    git clone https://github.com/ccache/ccache.git && \
    cd ccache && mkdir build && cd build && cmake .. && make install && \
#Install ninja build tracing tools
    cd / && \
    wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip && \
    gunzip /usr/local/bin/ninja.gz && \
    chmod a+x /usr/local/bin/ninja && \
#Install ClangBuildAnalyzer
    git clone https://github.com/aras-p/ClangBuildAnalyzer.git && \
    cd ClangBuildAnalyzer/ && \
    make -f projects/make/Makefile && \
    cd / && \
#Install latest cppcheck
    git clone https://github.com/danmar/cppcheck.git && \
    cd cppcheck && mkdir build && cd build && cmake .. && cmake --build . && \
    cd / && \
# Install an init system
    wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && rm dumb-init_*.deb && \
# Install packages for processing the performance results
    pip3 install --break-system-packages --upgrade pytest pymysql pandas==2.2.3 sqlalchemy==2.0.3 setuptools-rust setuptools sshtunnel==0.4.0 && \
# Add render group
    groupadd -f render && \
# Install the new rocm-cmake version
    git clone -b master https://github.com/ROCm/rocm-cmake.git  && \
    cd rocm-cmake && mkdir build && cd build && \
    cmake  .. && cmake --build . && cmake --build . --target install
