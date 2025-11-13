# Quant GEMM Matrix Multiplication

This folder contains examples of quant GEMMs using the ck_tile tile-programming implementation.

- BQuant kernel with blocks of B matrix sharing scales: custom GEMM pipeline

## Quantization Mode Comparison

| Quant Mode | A Matrix Organization | A Scale Shape | B Matrix Organization | B Scale Shape |
|------------|----------------------|---------------|----------------------|---------------|
| **BQuant** | Not quantized | N/A | Blocks along K dimension<br/>Each GroupSize×N block shares one scale | `[K/GroupSize, N]` |

---

## Features

- **Preshuffled GEMM**: Shuffle the GEMM of B (weight) matrix in the warp layout and bypass the shared memory to do the GEMM calculation. Best performance solution for GEMM.(not supported)
- **TransposeC**: Transpose the C Matrix Output layout to have the best coalesced scale reading
- **Preshuffled Quant**: Preshuffle the input matrix to load multiple Quant warp blocks along the selected dimension.(not supported)
- **Precision**: Supports uint8, split into two fp4 in the pipeline (for B Matrix).
- **Validation**: CPU validation and error tolerance options.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
# Compile the quant kernels
make tile_example_mxfp4_gemm -j
```
This will result in an executable `build/bin/tile_example_mxfp4_gemm`

## example
```
args:
          -b    batch size (default:1)
          -m    m dimension (default:1024)
          -n    n dimension (default:2048)
          -k    k dimension (default:64)
   -a_layout    Tensor A data layout (default: R)
   -b_layout    Tensor B data layout (default: C)
   -c_layout    Tensor C data layout (default: R)
   -stride_a    Tensor A stride (default:0)
   -stride_b    Tensor B stride (default:0)
   -stride_c    Tensor C stride (default:0)
          -v    0. No validation, 1. Validation on CPU
          -e    Absolute error tolerance (default:1e-5)
       -prec    data type. bf16f4 (default:bf16f4)
     -warmup    number of iterations before benchmark the kernel (default:10)
     -repeat    number of iterations to benchmark the kernel (default:100)
      -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
 -quant_mode    Which quant method to use bquant
```

User need to select correct mapping of config for each quant mode:

|  | quant_mode as runtime argument | Config in cpp file |
|:--------|:-----:|-------|

| For selecting BQuant  | bquant  | GemmConfigBQuantPrefill    |


