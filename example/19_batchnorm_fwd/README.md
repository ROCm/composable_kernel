# Instructions for ```batchnorm_fwd_nhwc``` Example

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

## Build ```batchnorm_fwd_nhwc```
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
 make -j batchnorm_fwd_nhwc
```

## Run ```batchnorm_fwd_nhwc```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: save result Mean/invVariance (0=no, 1=yes)
#arg2: update running Mean/Variance (0=no, 1=yes) 
#arg3: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg4: run kernel # of times (>1)
./bin/example_batchnorm_fwd_nhwc -D 128,16,16,1024 -v 1  0 0 3 10
```

Result 
```
in_n_c_hi_wi: dim 4, lengths {128, 192, 71, 71}, strides {967872, 1, 13632, 192}
out_n_c_ho_wo: dim 4, lengths {128, 192, 36, 36}, strides {248832, 1, 6912, 192}
launch_and_time_kernel: grid_dim {124416, 1, 1}, block_dim {64, 1, 1} 
Warm up
Start running 10 times...
Perf: 0.415453 ms, 1.37996 TFlops, 749.726 GB/s
error: 0
max_diff: 0, 1, 1
```
