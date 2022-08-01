## Docker script
```bash
docker run                                     \
-it                                            \
--privileged                                   \
--group-add sudo                               \
-w /root/workspace                             \
-v ${PATH_TO_LOCAL_WORKSPACE}:/root/workspace  \
rocm/tensorflow:rocm5.1-tf2.6-dev              \
/bin/bash
```

# Install newer version of rocm-cmake
https://github.com/RadeonOpenCompute/rocm-cmake

## Build
```bash
mkdir build && cd build
```

```bash
# Need to specify target ID, example below is gfx908 and gfx90a
cmake                                                                 \
-D BUILD_DEV=OFF                                                      \
-D CMAKE_BUILD_TYPE=Release                                           \
-D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 --offload-arch=gfx90a -O3" \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                        \
-D CMAKE_INSTALL_PREFIX=${PATH_TO_CK_INSTALL_DIRECTORY}               \
..
```

### Build and Run Examples
```bash
 make -j examples
```
Instructions for running each individual examples are under ```example/```

## Tests
```bash
 make -j examples tests
 make test
```

## Build ckProfiler
```bash
 make -j ckProfiler
```
Instructions for running ckProfiler are under ```profiler/```

## Install CK
```bash
make install
```

## Using CK as pre-built kernel library
Instructions for using CK as a pre-built kernel library are under ```client_example/```

## Caveat
### Kernel Timing and Verification
CK's own kernel timer will warn up kernel once, and then run it multiple times
to get average kernel time. For some kernels that use atomic add, this will cause
output buffer to be accumulated multiple times, causing verfication failure.
To work around it, do not use CK's own timer and do verification at the same time.
CK's own timer and verification in each example and ckProfiler can be enabled or
disabled from command line.
