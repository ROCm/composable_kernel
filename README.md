## Docker script
```bash
docker run                                     \
-it                                            \
--privileged                                   \
--group-add sudo                               \
-w /root/workspace                             \
-v ${PATH_TO_LOCAL_WORKSPACE}:/root/workspace  \
rocm/tensorflow:rocm4.3.1-tf2.6-dev            \
/bin/bash
```

## Build
```bash
mkdir build && cd build
```

```bash
# Need to specify target ID, example below is gfx908 and gfx90a
cmake                                                                 \
-D BUILD_DEV=OFF                                                      \
-D CMAKE_BUILD_TYPE=Release                                           \
-D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 --offload-arch=gfx90a -O3  \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                        \
..
```

### Build and Run Examples
```bash
 make -j examples
```
Instructions for running each individual examples are under ```example/```

## Tests
```bash
 make -j tests
 make test
```

## Build ckProfiler
```bash
 make -j ckProfiler
```
Instructions for running ckProfiler are under ```profiler/```
