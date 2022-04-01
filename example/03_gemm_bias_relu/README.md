# Instructions for ```example_gemm_xdl_bias_relu_add```

## Run ```example_gemm_xdl_bias_relu_add```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC
./bin/example_gemm_xdl_bias_relu_add 0 1 5 3840 4096 4096 4096 4096 4096
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
```
a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
b_k_n: dim 2, lengths {4096, 4096}, strides {1, 4096}
c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
c0_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
c1_m_n: dim 2, lengths {3840, 4096}, strides {1, 0}
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
arg.c0_grid_desc_m_n_{ 3840, 4096}
arg.c1_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {480, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 1.27583 ms, 100.992 TFlops, 73.9688 GB/s
```
