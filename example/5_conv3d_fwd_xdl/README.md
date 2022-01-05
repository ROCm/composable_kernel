# Instructions for ```conv3d_fwd_xdl``` Example

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

## Build ```conv3d_fwd_xdl```
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
 make -j conv3d_fwd_xdl
```

## Run ```conv3d_fwd_xdl```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4 to 24: N, K, C, Z, Y, X, Di, Hi, Wi, Sz, Sy, Sx, Dz, Dy, Dx, leftPz, LeftPy, LeftPx, RightPz, RightPy, RightPx
./example/conv3d_fwd_xdl 0 1 5
```

Result (MI200)
```
in_n_c_hi_wi: dim 4, lengths {128, 192, 71, 71}, strides {967872, 1, 13632, 192}
wei_k_c_y_x: dim 4, lengths {256, 192, 3, 3}, strides {1728, 1, 576, 192}
out_n_k_ho_wo: dim 4, lengths {128, 256, 36, 36}, strides {331776, 1, 9216, 256}
arg.a_grid_desc_k0_m_k1_{216, 165888, 8}
arg.b_grid_desc_k0_n_k1_{216, 256, 8}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {1296, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 1.43206 ms, 102.486 TFlops, 232.947 GB/s
```

