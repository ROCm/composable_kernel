# Instructions for ```conv2d_bwd_data_xdl``` Example

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

## Build ```conv2d_bwd_data_xdl```
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
 make -j conv2d_bwd_data_xdl
```

## Run ```conv2d_bwd_data_xdl```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4 to 18: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, RightPx
./bin/conv2d_bwd_data_xdl 0 1 5
```

Result (MI100 @ 1087Mhz, 79.5597TFlops peak FP16)
```
in_n_c_hi_wi: dim 4, lengths {128, 256, 71, 71}, strides {1290496, 1, 18176, 256}
wei_k_c_y_x: dim 4, lengths {256, 256, 3, 3}, strides {2304, 1, 768, 256}
out_n_k_ho_wo: dim 4, lengths {128, 256, 36, 36}, strides {331776, 1, 9216, 256}
arg.a_grid_desc_k0_m_k1_container_{128, 175232, 8}
arg.b_grid_desc_k0_n_k1_container_{128, 256, 8}
arg.c_grid_desc_m_n_container_{ 175232, 256}
arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_( 2738, 4, 2, 2, 4, 2 ) 
launch_and_time_kernel: grid_dim {2738, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 1 times...
arg.a_grid_desc_k0_m_k1_container_{64, 175232, 8}
arg.b_grid_desc_k0_n_k1_container_{64, 256, 8}
arg.c_grid_desc_m_n_container_{ 175232, 256}
arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_( 2738, 4, 2, 2, 4, 2 ) 
launch_and_time_kernel: grid_dim {2738, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 1 times...
arg.a_grid_desc_k0_m_k1_container_{64, 175232, 8}
arg.b_grid_desc_k0_n_k1_container_{64, 256, 8}
arg.c_grid_desc_m_n_container_{ 175232, 256}
arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_( 2738, 4, 2, 2, 4, 2 ) 
launch_and_time_kernel: grid_dim {2738, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 1 times...
arg.a_grid_desc_k0_m_k1_container_{32, 175232, 8}
arg.b_grid_desc_k0_n_k1_container_{32, 256, 8}
arg.c_grid_desc_m_n_container_{ 175232, 256}
arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_( 2738, 4, 2, 2, 4, 2 ) 
launch_and_time_kernel: grid_dim {2738, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 1 times...
Perf: 2.45966 ms, 79.5597 TFlops, 169.325 GB/s
```
