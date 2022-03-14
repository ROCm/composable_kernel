<<<<<<< HEAD:example/13_pool2d_fwd/README.md
# Instructions for ```pool2d_fwd``` Example
=======
# Instructions for ```grouped_gemm_xdl``` Example
>>>>>>> 17f80fcf4bb6e6e17f26ec1550aa194b962c50d7:example/14_grouped_gemm/README.md

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

<<<<<<< HEAD:example/13_pool2d_fwd/README.md
## Build ```pool2d_fwd```
=======
## Build ```grouped_gemm_xdl```
>>>>>>> 17f80fcf4bb6e6e17f26ec1550aa194b962c50d7:example/14_grouped_gemm/README.md
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
<<<<<<< HEAD:example/13_pool2d_fwd/README.md
 make -j pool2d_fwd
```

## Run ```pool2d_fwd```
=======
 make -j example_grouped_gemm_xdl_fp16
```

## Run ```grouped_gemm_xdl```
>>>>>>> 17f80fcf4bb6e6e17f26ec1550aa194b962c50d7:example/14_grouped_gemm/README.md
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
<<<<<<< HEAD:example/13_pool2d_fwd/README.md
#arg4 to 15: N, C, Y, X, Hi, Wi, Sy, Sx, LeftPy, LeftPx, RightPy, RightPx
./example/pool2d_fwd 1 1 10
=======
./bin/example_grouped_gemm_xdl_fp16 0 1 5
>>>>>>> 17f80fcf4bb6e6e17f26ec1550aa194b962c50d7:example/14_grouped_gemm/README.md
```

Result 
```
<<<<<<< HEAD:example/13_pool2d_fwd/README.md
in_n_c_hi_wi: dim 4, lengths {128, 192, 71, 71}, strides {967872, 1, 13632, 192}
out_n_c_ho_wo: dim 4, lengths {128, 192, 36, 36}, strides {248832, 1, 6912, 192}
launch_and_time_kernel: grid_dim {124416, 1, 1}, block_dim {64, 1, 1} 
Warm up
Start running 10 times...
Perf: 0.415453 ms, 1.37996 TFlops, 749.726 GB/s
error: 0
max_diff: 0, 1, 1
=======
gemm[0] a_m_k: dim 2, lengths {256, 64}, strides {64, 1} b_k_n: dim 2, lengths {64, 128}, strides {1, 64} c_m_n: dim 2, lengths {256, 128}, strides {128, 1}
gemm[1] a_m_k: dim 2, lengths {512, 128}, strides {128, 1} b_k_n: dim 2, lengths {128, 256}, strides {1, 128} c_m_n: dim 2, lengths {512, 256}, strides {256, 1}
gemm[2] a_m_k: dim 2, lengths {768, 192}, strides {192, 1} b_k_n: dim 2, lengths {192, 384}, strides {1, 192} c_m_n: dim 2, lengths {768, 384}, strides {384, 1}
gemm[3] a_m_k: dim 2, lengths {1024, 256}, strides {256, 1} b_k_n: dim 2, lengths {256, 512}, strides {1, 256} c_m_n: dim 2, lengths {1024, 512}, strides {512, 1}
group: 0 arg.a_grid_desc_k0_m_k1_{8, 256, 8}, arg.b_grid_desc_k0_n_k1_{8, 128, 8}, arg.c_grid_desc_m_n_{ 256, 128}
group: 1 arg.a_grid_desc_k0_m_k1_{16, 512, 8}, arg.b_grid_desc_k0_n_k1_{16, 256, 8}, arg.c_grid_desc_m_n_{ 512, 256}
group: 2 arg.a_grid_desc_k0_m_k1_{24, 768, 8}, arg.b_grid_desc_k0_n_k1_{24, 384, 8}, arg.c_grid_desc_m_n_{ 768, 384}
group: 3 arg.a_grid_desc_k0_m_k1_{32, 1024, 8}, arg.b_grid_desc_k0_n_k1_{32, 512, 8}, arg.c_grid_desc_m_n_{ 1024, 512}
launch_and_time_kernel: grid_dim {30, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 5 times...
Perf: 0.037887 ms, 11.0706 TFlops, 90.8132 GB/s, DeviceGroupedGemmXdl<256, 256, 128, 4, 8, 32, 32, 4, 2>
>>>>>>> 17f80fcf4bb6e6e17f26ec1550aa194b962c50d7:example/14_grouped_gemm/README.md
```
