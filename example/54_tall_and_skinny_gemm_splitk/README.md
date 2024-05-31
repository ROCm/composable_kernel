# Instructions for ```example_gemv_splitk```

## Run ```example_gemv_splitk```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4: number of splitk batches
bin/example_tall_and_skinny_gemm_splitk_fp16 1 2 1 231

```

Result (MI250 @ 800Mhz, 181.05TFlops peak FP16)
```
a_m_k: dim 2, lengths {16, 1024}, strides {1024, 1}
b_k_n: dim 2, lengths {1024, 16}, strides {16, 1}
c_m_n: dim 2, lengths {16, 16}, strides {16, 1}
Perf: 0.0065438 ms, 0.0801198 TFlops, 10.0932 GB/s, deviceTsmmDl<64, 16, 128, 4, 2, 16, 2, 1>
```
