# Instructions for ```example_conv3d_fwd_xdl```

## Run ```example_conv3d_fwd_xdl```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4 to 24: N, K, C, Z, Y, X, Di, Hi, Wi, Sz, Sy, Sx, Dz, Dy, Dx, leftPz, LeftPy, LeftPx, RightPz, RightPy, RightPx
./bin/example_conv3d_fwd_xdl 0 1 5
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
```
wei: dim 5, lengths {256, 3, 3, 3, 192}, strides {5184, 1728, 576, 192, 1}
out: dim 5, lengths {4, 36, 36, 36, 256}, strides {11943936, 331776, 9216, 256, 1}
num_batches_of_GEMM = 1
a_grid_desc_k0_m_k1{648, 186624, 8}
b_grid_desc_k0_n_k1{648, 256, 8}
c_grid_desc_m_n{ 186624, 256}
launch_and_time_kernel: grid_dim {1458, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 4.58795 ms, 107.965 TFlops, 141.23 GB/s
```
