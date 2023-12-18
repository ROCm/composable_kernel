# Instructions for ```example_complex_contraction_bilinear_xdl_fp32```

## Run
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: time kernel (0=no, 1=yes)
./bin/example_contraction_bilinear_xdl_fp32 1 1 1
```

Result (MI210 )
```
a_ms_ks_re: dim 4, lengths {30, 128, 32, 64}, strides {524288, 4096, 128, 1}
b_ns_ks_re: dim 4, lengths {32, 64, 32, 64}, strides {524288, 4096, 128, 1}
d_ms_ns_re: dim 4, lengths {30, 128, 32, 64}, strides {524288, 4096, 128, 1}
e_ms_ns_re: dim 4, lengths {30, 128, 32, 64}, strides {524288, 4096, 128, 1}
a_ms_ks_img: dim 4, lengths {30, 128, 32, 64}, strides {524288, 4096, 128, 1}
b_ns_ks_img: dim 4, lengths {32, 64, 32, 64}, strides {524288, 4096, 128, 1}
d_ms_ns_img: dim 4, lengths {30, 128, 32, 64}, strides {524288, 4096, 128, 1}
e_ms_ns_img: dim 4, lengths {30, 128, 32, 64}, strides {524288, 4096, 128, 1}
Perf: 4.51253 ms, 14.2768 TFlops, 31.6023 GB/s, 
DeviceContractionMultipleD_Xdl_CShuffle<2, 2, 2, 256, 256, 128, 16, 4, 4, 2, 2>
```
