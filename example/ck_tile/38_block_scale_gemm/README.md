# Quant GEMM Matrix Multiplication

This folder contains examples of quant GEMMs using the ck_tile tile-programming implementation.

- AQuant kernel with blocks of A matrix sharing scales: custom GEMM pipeline
- BQuant kernel with blocks of B matrix sharing scales: custom GEMM pipeline
- Row and Column-wise scaled: scaling implemented in Epilogue
- Tensor-wise scaled: scaling implemented in Epilogue

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
# Compile the quant kernels
make tile_example_gemm_quant_basic -j
```
This will result in an executable `build/bin/tile_example_gemm_quant_basic`

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
          -v    0. No validation, 1. Validation on CPU, 2. Validation on GPU (default:1)
          -e    Absolute error tolerance (default:1e-5)
       -prec    data type. fp8/bf8/i4fp8/i4bf8/i4f32fp8/i4f32bf8 (default:fp8)
     -warmup    number of iterations before benchmark the kernel (default:10)
     -repeat    number of iterations to benchmark the kernel (default:100)
      -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
 -quant_mode    Which quant method to use (aquant, bquant, tensor, rowcol)
```

User need to select correct mapping of config for each quant mode:

|  | quant_mode as runtime argument | Config in cpp file |
|:--------|:-----:|-------|
| For selecting AQuant  | aquant  | GemmConfigQuant    |
| For selecting Aquant with Preshuffle   | aquant  | GemmConfigPreshuffleQuant    |
| For selecting BQuant  | bquant  | GemmConfigQuant    |
| For selecting RowCol quant  | rowcolquant  | GemmConfigRowColQuant    |

