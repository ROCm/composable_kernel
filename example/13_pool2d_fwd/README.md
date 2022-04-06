# Instructions for ```example_pool2d_fwd``` Example

## Run ```example_pool2d_fwd```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3: run kernel # of times (>1)
#arg4 to 15: N, C, Y, X, Hi, Wi, Sy, Sx, LeftPy, LeftPx, RightPy, RightPx
./bin/example_pool2d_fwd 1 1 10
```

Result 
```
in_n_c_hi_wi: dim 4, lengths {128, 192, 71, 71}, strides {967872, 1, 13632, 192}
out_n_c_ho_wo: dim 4, lengths {128, 192, 36, 36}, strides {248832, 1, 6912, 192}
launch_and_time_kernel: grid_dim {124416, 1, 1}, block_dim {64, 1, 1} 
Warm up
Start running 10 times...
Perf: 0.415453 ms, 1.37996 TFlops, 749.726 GB/s
error: 0
max_diff: 0, 1, 1
```
