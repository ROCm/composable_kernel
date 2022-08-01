```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: time kernel (0=no, 1=yes)
#Following arguments (depending on number of spatial dims):
# N spatial dimensions
# G, N, K, C,
# <filter spatial dimensions>, (ie Y, X for 2D)
# <input image spatial dimensions>, (ie Hi, Wi for 2D)
# <strides>, (ie Sy, Sx for 2D)
# <dilations>, (ie Dy, Dx for 2D)
# <left padding>, (ie LeftPy, LeftPx for 2D)
# <right padding>, (ie RightPy, RightPx for 2D)

bin/example_grouped_convnd_fwd_bias_relu_xdl_fp16 1 1 1
```

Result (MI100)
```
in: dim 5, lengths {1, 128, 192, 71, 71}, strides {6912, 967872, 1, 13632, 192}
wei: dim 5, lengths {1, 256, 192, 3, 3}, strides {192, 1728, 1, 576, 192}
bias: dim 5, lengths {1, 128, 256, 36, 36}, strides {256, 0, 1, 0, 0}
out: dim 5, lengths {1, 128, 256, 36, 36}, strides {256, 331776, 1, 9216, 256}
launch_and_time_kernel: grid_dim {1296, 1, 1}, block_dim {256, 1, 1}
Warm up 1 time
Start running 10 times...
Perf: 1.19215 ms, 123.112 TFlops, 279.827 GB/s, DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<256, 128, 256, 32, Default>
```
