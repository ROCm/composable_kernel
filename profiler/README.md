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

################        op  datatype  layout  verify  init  log  repeat  M___ N___ K___  StrideA StrideB StrideC
./bin/ckProfiler      gemm         1       1       1     1    0       5  3840 4096 4096     4096    4096    4096
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
```bash
a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
b_k_n: dim 2, lengths {4096, 4096}, strides {1, 4096}
c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
....
Best Perf: 1.1933 ms, 107.977 TFlops, 79.0848 GB/s
```

## Profile 2d forward convolution kernels
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
 ################          op datatype  in_layout   wei_layout  out_layout  verify  init  log  repeat  N__ K___ C___ Y X Hi__ Wi__ Strides Dilations LeftPads RightPads
 ./bin/ckProfiler  conv2d_fwd        1          1            1           1       1     1    0       5  128  256  192 3 3   71   71     2 2       1 1      1 1       1 1
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
```
in_n_c_hi_wi: dim 4, lengths {128, 192, 71, 71}, strides {967872, 1, 13632, 192}
wei_k_c_y_x: dim 4, lengths {256, 192, 3, 3}, strides {1728, 1, 576, 192}
out_n_k_ho_wo: dim 4, lengths {128, 256, 36, 36}, strides {331776, 1, 9216, 256}
....
Best Perf: 1.42509 ms, 102.988 TFlops, 234.086 GB/s
```

## Profile contraction kernels
```bash
#arg1: tensor operation (contraction_bilinear=CONTRACTION+Bilinear)
#arg2: data type (0: fp32; 1: f64)\n"
#arg3: matrix layout (0: A[m0, m1, k0, k1] * B[k0, k1, n0, n1] + D[m0, m1, n0, n1] = E[m0, m1, n0, n1];
#                     1: A[m0, m1, k0, k1] * B[n0, n1, k0, k1] + D[m0, m1, n0, n1] = E[m0, m1, n0, n1];
#                     2: A[k0, k1, m0, m1] * B[k0, k1, n0, n1] + D[m0, m1, n0, n1] = E[m0, m1, n0, n1];
#                     3: A[k0, k1, m0, m1] * B[n0, n1, k0, k1] + D[m0, m1, n0, n1] = E[m0, m1, n0, n1])
#arg4: verification (0: no; 1: yes)
#arg5: initialization (0: no init; 1: integer value; 2: decimal value)
#arg6: print tensor value (0: no; 1: yes)
#arg7: time kernel (0: no, 1: yes)
#arg8 and arg9: alpha and beta
#arg10 to 15: M0, M1, N0, N1, K0, K1
#arg16 to 31: Strides for A, B, D and E (skip for default)

################                   op  datatype  layout  verify  init  log  time  alpha  beta  M0  M1  N0  N1  K0  K1
./bin/ckProfiler contraction_bilinear         0       1       0     0    0     1    1.0   1.0 128 128 128 128 128 128
```

Result (MI100)
```bash
a_m_k: dim 4, lengths {128, 128, 128, 128}, strides {2097152, 16384, 128, 1}
b_k_n: dim 4, lengths {128, 128, 128, 128}, strides {128, 1, 2097152, 16384}
d_m_n: dim 4, lengths {128, 128, 128, 128}, strides {2097152, 16384, 128, 1}
e_m_n: dim 4, lengths {128, 128, 128, 128}, strides {2097152, 16384, 128, 1}
....
Best Perf: 211.405 ms, 41.6077 TFlops, 15.2372 GB/s
```

## Profile batched gemm multiple D kernels
```bash
#arg1: tensor operation (batched_gemm_multi_d=Batched GEMM multi D);
#arg2: data type (0: fp16; 1: int8)
#arg3: matrix layout (0: A[g, m, k] * B[g, k, n] = C[g, m, n];
#                     1: A[g, m, k] * B[g, n, k] = C[g, m, n];
#                     2: A[g, k, m] * B[g, k, n] = C[g, m, n];
#                     3: A[g, k, m] * B[g, n, k] = C[g, m, n])
#arg4: verification (0: no; 1: yes)
#arg5: initialization (0: no init; 1: integer value; 2: decimal value)
#arg6: print tensor value (0: no; 1: yes)
#arg7: time kernel (0=n0, 1=yes)
#arg8 to 17: M, N, K, StrideA, StrideB, StrideC, BatchStrideA, BatchStrideB, BatchStrideC, BatchCount

################                   op  datatype  layout  verify  init  log  time    M    N    K StrideA StrideB StrideC BatchStrideA BatchStrideB BatchStrideC BatchCount
./bin/ckProfiler batched_gemm_multi_d         0       1       0     0    0     1 4096 4096 4096    4096    4096    4096     16777216     16777216     16777216         16
```

Result (Radeon RX 6800 XT)
```bash
arg.a_grid_desc_k0_m0_m1_k1_{2048, 4096, 2}
arg.b_grid_desc_k0_n0_n1_k1_{2048, 4096, 2}
arg.e_grid_desc_m_n_{ 4096, 4096}
....
Best Perf: 58.0306 ms, 37.8942 TFlops, 27.7545 GB/s
## Profile grouped convolution backward data kernels
```bash
# arg1: tensor operation (grouped_conv_bwd_data: Grouped Convolution Backward Data)
# arg2: data type (0: Output fp32, Weight fp32, Input fp32
#                  1: Output fp16, Weight fp16, Input fp16
#                  2: Output bf16, Weight bf16, Input bf16
# arg3: tensor layout (0: Output[G, N, Hi, Wi, C], Weight[G, K, Y, X, C], Input[G, N, Ho, Wo, K]
#                      1: Output[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Input[N, Ho, Wo, G, K])
# arg4: verification (0: no, 1: yes)
# arg5: initialization (0: no init, 1: integer value, 2: decimal value)
# arg6: print tensor value (0: no; 1: yes)
# arg7: time kernel (0: no, 1: yes)
# Following arguments (depending on number of spatial dims):
#  Number of spatial dimensions (1=Conv1d, 2=Conv2d, 3=Conv3d)
#  G, N, K, C, 
#  <filter spatial dimensions>, (ie Y, X for 2D)
#  <input image spatial dimensions>, (ie Hi, Wi for 2D)
#  <strides>, (ie Sy, Sx for 2D)
#  <dilations>, (ie Dy, Dx for 2D)
#  <left padding>, (ie LeftPy, LeftPx for 2D)
#  <right padding>, (ie RightPy, RightPx for 2D)

 ################                   op   datatype  layout  verify  init  log  time  Ndims  G  N   K   C  Y  X  Hi  Wi  Sy  Sx  Dy  Dx  LeftPy  LeftPx  RightPy  RightPx
./bin/ckProfiler grouped_conv_bwd_data          1       0       1     1    0     1      2 32  4 192 192  3  3  28  28   1   1   1   1       1       1        1        1

 ```

Result (MI100, FP16, GNHWC_GKYXC_GNHWK)
```
out: dim 5, lengths {32, 4, 192, 28, 28}, strides {602112, 150528, 1, 5376, 192}
wei: dim 5, lengths {32, 192, 192, 3, 3}, strides {331776, 1728, 1, 576, 192}
in: dim 5, lengths {32, 4, 192, 28, 28}, strides {602112, 150528, 1, 5376, 192}
....
Best configuration parameters:
name: DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 256, 32, 8, 2, Default, 32, 32, 2, 4, 8, 4, 1, 1>
avg_time: 0.768321
tflops: 86.6679
GB/s: 127.947
```

## Profile grouped convolution backward weight kernels
```bash
# arg1: tensor operation (grouped_conv_bwd_weight: Grouped Convolution Backward Weight)
# arg2: data type (0: Input fp32, Weight fp32, Output fp32
#                  1: Input fp16, Weight fp16, Output fp16
#                  2: Input bf16, Weight fp32, Output bf16)
# arg3: tensor layout (0: Input[G, N, C, Hi, Wi], Weight[G, K, C, Y, X], Output[G, N, K, Ho, Wo]
#                      1: Input[G, N, Hi, Wi, C], Weight[G, K, Y, X, C], Output[G, N, Ho, Wo, K]
#                      2: Input[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Output[N, Ho, Wo, G, K]
# arg4: verification (0: no, 1: yes)
# arg5: initialization (0: no init, 1: integer value, 2: decimal value)
# arg6: print tensor value (0: no; 1: yes)
# arg7: time kernel (0: no, 1: yes)
# Following arguments (depending on number of spatial dims):
#  Number of spatial dimensions (1=Conv1d, 2=Conv2d, 3=Conv3d)
#  G, N, K, C, 
#  <filter spatial dimensions>, (ie Y, X for 2D)
#  <input image spatial dimensions>, (ie Hi, Wi for 2D)
#  <strides>, (ie Sy, Sx for 2D)
#  <dilations>, (ie Dy, Dx for 2D)
#  <left padding>, (ie LeftPy, LeftPx for 2D)
#  <right padding>, (ie RightPy, RightPx for 2D)
# SplitK

 ################                   op   datatype  layout  verify  init  log  time  Ndims  G   N   K   C  Y  X  Hi  Wi  Sy  Sx  Dy  Dx  LeftPy  LeftPx  RightPy  RightPx  SplitK
./bin/ckProfiler grouped_conv_bwd_weight          1       0       1     1    0     1      2 32 256 256 512  3  3  28  28   1   1   1   1       1       0        0        0       1

 ```

Result (MI100, FP16, GNHWC_GKYXC_GNHWK)
```
input: dim 5, lengths {32, 512, 1024, 28, 28}, strides {411041792, 802816, 1, 28672, 1024}
weight: dim 5, lengths {32, 512, 1024, 3, 3}, strides {4718592, 9216, 1, 3072, 1024}
output: dim 5, lengths {32, 512, 512, 26, 26}, strides {177209344, 346112, 1, 13312, 512}
....
Best configuration parameters:
name: DeviceGroupedConvBwdWeight_Xdl_CShuffle<256, 256, 128, 4, Default, 8, 4, 2, 8, 4, 8, 2, 1, 1, 8>
avg_time: 68.5216
tflops: 95.337
GB/s: 69.2301
```
Note: This kernel use atomic add, this will cause output buffer to be accumulated multiple times, causing verification failure. To work around it, do not use CK's own timer and do verification at the same time.
