# Instructions for ```example_gemv_splitk```

## Run ```example_gemv_splitk```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4: number of splitk batches
./bin/example_gemv_splitk 0 1 5 151

```

Result (MI250 @ 800Mhz, 181.05TFlops peak FP16)
```
a_m_k: dim 2, lengths {1, 4608}, strides {4608, 1}
b_k_n: dim 2, lengths {4608, 1104}, strides {1, 4608}
c_m_n: dim 2, lengths {1, 1104}, strides {1104, 1}
arg.a_grid_desc_kbatch_k0_m_k1_{1,4, 1, 8}
arg.b_grid_desc_kbatch_k0_n_k1_{1,4, 1104, 8}
arg.c_grid_desc_m_n_{ 1, 1104}
launch_and_time_kernel: grid_dim {1359, 1, 1}, block_dim {64, 1, 1}
Warm up
Start running 10 times...
Perf: 0.0191358 ms, 0.531698 TFlops,532.295 GB/s
```
