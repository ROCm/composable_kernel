# Instructions for ```example_conv_xdl_bias_relu```

## Run ```example_conv_xdl_bias_relu```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4 to 18: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, RightPx
./bin/example_conv_xdl_bias_relu 0 1 5
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
```
in_n_c_hi_wi: dim 4, lengths {128, 192, 71, 71}, strides {967872, 1, 13632, 192}
wei_k_c_y_x: dim 4, lengths {256, 192, 3, 3}, strides {1728, 1, 576, 192}
out_n_k_ho_wo: dim 4, lengths {128, 256, 36, 36}, strides {331776, 1, 9216, 256}
bias_k: dim 1, lengths {256}, strides {1}
launch_and_time_kernel: grid_dim {1296, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 1.39009 ms, 105.581 TFlops, 239.981 GB/s
```
