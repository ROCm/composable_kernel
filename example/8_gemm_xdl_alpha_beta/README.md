# Instructions for ```gemm_xdl_alpha_beta``` Example

## Docker script
```bash
docker run                                                                   \
-it                                                                          \
--rm                                                                         \
--privileged                                                                 \
--group-add sudo                                                             \
-w /root/workspace                                                           \
-v ${PATH_TO_LOCAL_WORKSPACE}:/root/workspace                                \
rocm/tensorflow:rocm4.3.1-tf2.6-dev                                          \
/bin/bash
```

## Build ```gemm_xdl_alpha_beta```
```bash
mkdir build && cd build
```

```bash
# Need to specify target ID, example below is gfx908
cmake                                                                  \
-D BUILD_DEV=OFF                                                       \
-D CMAKE_BUILD_TYPE=Release                                            \
-D CMAKE_CXX_FLAGS="-DCK_AMD_GPU_GFX908 --amdgpu-target=gfx908 -O3 "   \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                              \
-D CMAKE_PREFIX_PATH=/opt/rocm                                         \
..
```

```bash
 make -j gemm_xdl_alpha_beta
```

## Run ```gemm_xdl_alpha_beta```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
./example/gemm_xdl_alpha_beta 0 1 5
```

