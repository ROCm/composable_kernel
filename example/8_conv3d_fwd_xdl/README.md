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
in_n_di_hi_wi_c: dim 5, lengths {4, 71, 71, 71, 192}, strides {68718912, 967872, 13632, 192, 1}
wei_k_z_y_x_c: dim 5, lengths {128, 3, 3, 3, 192}, strides {5184, 1728, 576, 192, 1}
out_n_do_ho_wo_k: dim 5, lengths {4, 36, 36, 36, 128}, strides {5971968, 165888, 4608, 128, 1}
a_grid_desc_b_k0_m_k1{1, 648, 186624, 8}
b_grid_desc_k0_n_k1{648, 128, 8}
c_grid_desc_b_m_n{ 1, 186624, 128}
launch_and_time_kernel: grid_dim {729, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 3.06193 ms, 80.8868 TFlops, 195.581 GB/s
```

