# Instructions for ```example_gemv_splitk```

## Run ```example_gemv_splitk```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4: number of splitk batches
bin/example_gemv_splitk_fp16 1 2 1 231

```

Result (MI250 @ 800Mhz, 181.05TFlops peak FP16)
```
a_m_k: dim 2, lengths {1, 4608}, strides {4608, 1}
b_k_n: dim 2, lengths {4608, 1104}, strides {1104, 1}
c_m_n: dim 2, lengths {1, 1104}, strides {1104, 1}
Perf: 0.0111038 ms, 0.916305 TFlops, 917.334 GB/s, deviceTsmmDl<64, 1, 128, 3, 4, 1, 2, 1>
```
