# Instructions for ```example_gemm_add_multiply_dl_fp16```

## Run ```example_gemm_add_multiply_dl_fp16```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: time kernel (0=no, 1=yes)
#arg4 to 11: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD0, StrideD1, StrideE"
./bin/example_gemm_add_multiply_dl_fp16 1 1 1
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
```
a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
b_k_n: dim 2, lengths {4096, 4096}, strides {4096, 1}
d0_m_n: dim 2, lengths {3840, 4096}, strides {0, 1}
d1_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
e_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
arg.a_grid_desc_k0_m0_m1_k1_{2048, 3840, 2}
arg.b_grid_desc_k0_n0_n1_k1_{2048, 4096, 2}
arg.e_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {960, 1, 1}, block_dim {256, 1, 1}
Warm up 1 time
Start running 10 times...
Perf: 3.99904 ms, 32.22 TFlops, 31.9913 GB/s, DeviceGemmMultipleD_Dl<256, 128, 128, 16, 2, 4, 4, 1>
```
