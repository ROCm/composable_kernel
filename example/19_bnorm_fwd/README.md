# Instructions for ```batchnorm_fwd_nhwc_blockwise``` Example

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

## Build ```batchnorm_fwd_nhwc_blockwise```
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
 make -j batchnorm_fwd_nhwc_blockwise
```

## Run ```batchnorm_fwd_nhwc_blockwise```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: save result Mean/invVariance (0=no, 1=yes)
#arg2: update running Mean/Variance (0=no, 1=yes) 
#arg3: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg4: run kernel # of times (>1)
./bin/example_batchnorm_fwd_nhwc_blockwise -D 128,16,16,1024 -v 1  1 0 2 10
```

Result 
```
Perf: 1.32944 ms, 50.485 GB/s, DeviceBatchNorm_NHWC_C_Blockwise<256,M_C8_S1,K_C32_S1,InOutVectorSize_1_ScaleBiasMeanVarVectorSize_1>
```
