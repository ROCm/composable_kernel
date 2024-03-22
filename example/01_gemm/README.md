# Instructions for ```example_gemm_xdl```

## Run ```example_gemm_xdl```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
./bin/example_gemm_xdl 0 1 5
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
```
a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
b_k_n: dim 2, lengths {4096, 4096}, strides {1, 4096}
c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {480, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 1.19685 ms, 107.657 TFlops, 78.8501 GB/s
```

# Instructions for ```example_gemm_xdl_streamk```

## Run ```example_gemm_xdl_streamk```
```bash
# arg1: verification (0=no, 1=yes)
# arg2: initialization (0=no init, 1=integer value, 2=decimal value)
# arg3: time kernel (0=no, 1=yes)
# arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC
# arg10: NumSKBlocks(optional, defaults to DP GEMM)
bin/example_gemm_xdl_streamk 1 2  1 3840 4096 4096 4096 4096  4096  312
```

Result (MI250 @ 1700Mhz, 181TFlops peak FP16 on 1 dye)
```
a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
b_k_n: dim 2, lengths {4096, 4096}, strides {4096, 1}
c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
Recommended grid size :312
Perf: 1.21689 ms, 105.884 TFlops, 79.2748 GB/s, GemmXdlStreamK_RRR_B256_Vec8x2x8_128x128x4x8

```
