# Instructions for ```conv2d_bwd_wgt_xdl``` Example

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

## Build ```conv2d_bwd_wgt_xdl```
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
 make -j conv2d_bwd_wgt_xdl
```

## Run ```conv2d_bwd_wgt_xdl```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4: is show log (0=no, 1=yes)
#arg5 to 19: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, RightPx, split-k
./example/conv2d_bwd_wgt_xdl 0 1 5 0 4
```

Result 
```
in_n_c_hi_wi: dim 4, lengths {128, 1024, 14, 14}, strides {200704, 1, 14336, 1024}
wei_k_c_y_x: dim 4, lengths {256, 1024, 3, 3}, strides {9216, 1, 3072, 1024}
out_n_k_ho_wo: dim 4, lengths {128, 256, 6, 6}, strides {9216, 1, 1536, 256}
arg.a_grid_desc_kbatch_k0_m_k1_{4, 144, 256, 8}
arg.b_grid_desc_kbatch_k0_n_k1_{4, 144, 9216, 8}
arg.c_grid_desc_m_n_{ 256, 9216}
launch_and_time_kernel: grid_dim {576, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 5 times...
Perf: 0.401084 ms, 54.2112 TFlops, 145.75 GB/s
```
