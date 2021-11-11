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

## Build ```ckProfiler```
```bash
mkdir build && cd build
```

```bash
# Need to Specify target ID, example below is gfx908
cmake                                                                  \
-D BUILD_DEV=OFF                                                       \
-D CMAKE_BUILD_TYPE=Release                                            \
-D CMAKE_CXX_FLAGS="-DCK_AMD_GPU_GFX908 --amdgpu-target=gfx908 -O3 "   \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                              \
-D CMAKE_PREFIX_PATH=/opt/rocm                                         \
..
```

```bash
 make -j ckProfiler
```

## Profile GEMM kernels
```bash
#arg1: tensor operation (gemm=GEMM)
#arg2: data type (0=fp32, 1=fp16)
#arg3: matrix layout (0=NN, 1=NT, 2=TN, 3=TT)
#arg4: verification (0=no, 1=yes)
#arg5: initialization (0=no init, 1=integer value, 2=decimal value)
#arg6: print matrix value (0=no, 1=yes)
#arg7: run kernel # of times (>1)
#arg8 to 13: M, N, K, StrideA, StrideB, StrideC

#####################   op  datatype  layout  verify  init  log  repeat  M___ N___ K___  StrideA StrideB StrideC
./profiler/ckProfiler gemm         1       1       1     1    0       5  3840 4096 4096     4096    4096    4096
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
```bash
a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
b_k_n: dim 2, lengths {4096, 4096}, strides {1, 4096}
c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {480, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 1.1933 ms, 107.977 TFlops, 79.0848 GB/s
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {480, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 1.26163 ms, 102.129 TFlops, 74.8017 GB/s
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {960, 1, 1}, block_dim {128, 1, 1}
Warm up
Start running 5 times...
Perf: 1.9314 ms, 66.7127 TFlops, 48.8618 GB/s
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {960, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 1.71033 ms, 75.3357 TFlops, 55.1776 GB/s
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {1920, 1, 1}, block_dim {128, 1, 1}
Warm up
Start running 5 times...
Perf: 3.66508 ms, 35.1558 TFlops, 25.7489 GB/s
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {1920, 1, 1}, block_dim {128, 1, 1}
Warm up
Start running 5 times...
Perf: 3.4591 ms, 37.2493 TFlops, 27.2822 GB/s
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {3840, 1, 1}, block_dim {64, 1, 1}
Warm up
Start running 5 times...
Perf: 5.91757 ms, 21.774 TFlops, 15.9477 GB/s
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {1920, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 3.98802 ms, 32.309 TFlops, 23.6638 GB/s
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {1920, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 3.08822 ms, 41.7228 TFlops, 30.5587 GB/s
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {3840, 1, 1}, block_dim {128, 1, 1}
Warm up
Start running 5 times...
Perf: 6.3231 ms, 20.3775 TFlops, 14.9249 GB/s
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {3840, 1, 1}, block_dim {128, 1, 1}
Warm up
Start running 5 times...
Perf: 4.86525 ms, 26.4835 TFlops, 19.3971 GB/s
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {7680, 1, 1}, block_dim {64, 1, 1}
Warm up
Start running 5 times...
Perf: 14.0612 ms, 9.16343 TFlops, 6.7115 GB/s
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {7680, 1, 1}, block_dim {64, 1, 1}
Warm up
Start running 5 times...
Perf: 10.1509 ms, 12.6934 TFlops, 9.29694 GB/s
Best Perf: 1.1933 ms, 107.977 TFlops, 79.0848 GB/s
```

## Profile forward convolution kernels
```bash
#arg1: tensor operation (conv=Convolution)
#arg2: data type (0=fp32, 1=fp16)
#arg3: input tensor layout (0=NCHW, 1=NHWC)
#arg4: weight tensor layout (0=KCYX, 1=KYXC)
#arg5: output tensor layout (0=NKHW, 1=NHWK)
#arg6: verification (0=no, 1=yes)
#arg7: initialization (0=no init, 1=integer value, 2=decimal value)
#arg8: print matrix value (0=no, 1=yes)
#arg9: run kernel # of times (>1)
#arg10 to 24: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, RightPx
 #####################   op  datatype  in_layout   wei_layout  out_layout  verify  init  log  repeat  N__ K___ C___ Y X Hi__ Wi__ Strides Dilations LeftPads RightPads
 ./profiler/ckProfiler conv         1          1            1           1       1     1    0       5  128  256  192 3 3   71   71     2 2       1 1      1 1       1 1
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
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
Perf: 1.46118 ms, 100.445 TFlops, 228.306 GB/s
arg.a_grid_desc_k0_m_k1_{216, 165888, 8}
arg.b_grid_desc_k0_n_k1_{216, 256, 8}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {1296, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 1.42509 ms, 102.988 TFlops, 234.086 GB/s
arg.a_grid_desc_k0_m_k1_{216, 165888, 8}
arg.b_grid_desc_k0_n_k1_{216, 256, 8}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {2592, 1, 1}, block_dim {128, 1, 1}
Warm up
Start running 5 times...
Perf: 1.92757 ms, 76.141 TFlops, 173.065 GB/s
arg.a_grid_desc_k0_m_k1_{216, 165888, 8}
arg.b_grid_desc_k0_n_k1_{216, 256, 8}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {2592, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 1.44397 ms, 101.642 TFlops, 231.027 GB/s
arg.a_grid_desc_k0_m_k1_{216, 165888, 8}
arg.b_grid_desc_k0_n_k1_{216, 256, 8}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {5184, 1, 1}, block_dim {128, 1, 1}
Warm up
Start running 5 times...
Perf: 2.12356 ms, 69.1136 TFlops, 157.092 GB/s
arg.a_grid_desc_k0_m_k1_{216, 165888, 8}
arg.b_grid_desc_k0_n_k1_{216, 256, 8}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {5184, 1, 1}, block_dim {128, 1, 1}
Warm up
Start running 5 times...
Perf: 1.87149 ms, 78.4225 TFlops, 178.251 GB/s
arg.a_grid_desc_k0_m_k1_{216, 165888, 8}
arg.b_grid_desc_k0_n_k1_{216, 256, 8}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {10368, 1, 1}, block_dim {64, 1, 1}
Warm up
Start running 5 times...
Perf: 2.54501 ms, 57.6687 TFlops, 131.078 GB/s
arg.a_grid_desc_k0_m_k1_{216, 165888, 8}
arg.b_grid_desc_k0_n_k1_{216, 256, 8}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {5184, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 1.92089 ms, 76.4059 TFlops, 173.667 GB/s
arg.a_grid_desc_k0_m_k1_{216, 165888, 8}
arg.b_grid_desc_k0_n_k1_{216, 256, 8}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {5184, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 1.84594 ms, 79.5082 TFlops, 180.718 GB/s
arg.a_grid_desc_k0_m_k1_{216, 165888, 8}
arg.b_grid_desc_k0_n_k1_{216, 256, 8}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {10368, 1, 1}, block_dim {128, 1, 1}
Warm up
Start running 5 times...
Perf: 3.39502 ms, 43.2301 TFlops, 98.26 GB/s
arg.a_grid_desc_k0_m_k1_{216, 165888, 8}
arg.b_grid_desc_k0_n_k1_{216, 256, 8}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {10368, 1, 1}, block_dim {128, 1, 1}
Warm up
Start running 5 times...
Perf: 2.93298 ms, 50.0403 TFlops, 113.739 GB/s
arg.a_grid_desc_k0_m_k1_{216, 165888, 8}
arg.b_grid_desc_k0_n_k1_{216, 256, 8}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {20736, 1, 1}, block_dim {64, 1, 1}
Warm up
Start running 5 times...
Perf: 3.65925 ms, 40.1085 TFlops, 91.1647 GB/s
arg.a_grid_desc_k0_m_k1_{216, 165888, 8}
arg.b_grid_desc_k0_n_k1_{216, 256, 8}
arg.c_grid_desc_m_n_{ 165888, 256}
launch_and_time_kernel: grid_dim {20736, 1, 1}, block_dim {64, 1, 1}
Warm up
Start running 5 times...
Perf: 3.39573 ms, 43.2211 TFlops, 98.2395 GB/s
Best Perf: 1.42509 ms, 102.988 TFlops, 234.086 GB/s
```
