# GEMM Matrix Multiplication

This folder contains the Block-Scale GEMM example using the ck_tile tile-programming implementation.

## Overview
- **Operation**: Computes C = (A × B) scaled by AQ.
  - A shape: M × K
  - B shape: K × N
  - AQ shape: M × (K / QuantGroupSize)
  - Scaling is applied per M row and per quant-group along K.
- **Layouts (current example)**: A = Row-major, AQ = Row-major, B = Column-major, C = Row-major.
- **Constraint**: K must be divisible by `QuantGroupSize` (AQK = K / QuantGroupSize).

## Pipelines
- **Compute v3 (Intrawave)**: compute-optimized tiling
- **Memory (Interwave)**: memory-optimized tiling, better cache/DRAM behavior for bandwidth-bound cases.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
# The aquant pipeline method on the gemm calculation
make tile_example_gemm_aquant_basic -j
```
This will result in an executable `build/bin/tile_example_gemm_aquant_basic`

## example
```
args:
          -m    m dimension (default:16)
          -n    n dimension (default:64)
          -k    k dimension (default:256)
   -a_layout    Tensor A data layout (default: R)
  -aq_layout    Tensor AQ data layout (default: R)
   -b_layout    Tensor B data layout (default: C)
   -c_layout    Tensor C data layout (default: R)
   -stride_a    Tensor A stride (default:0)
   -stride_q    Tensor AQ stride (default:0)
   -stride_b    Tensor B stride (default:0)
   -stride_c    Tensor C stride (default:0)
          -v    0. No validation, 1. Validation on CPU, 2. Validation on GPU (default:1)
       -prec    data type. fp8/bf8/i4fp8/i4bf8/i4f32fp8/i4f32bf8 (default:fp8)
     -warmup    number of iterations before benchmark the kernel (default:50)
     -repeat    number of iterations to benchmark the kernel (default:1000)
      -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
    -split_k    splitK value (default:1)
       -init    0:random, 1:linear, 2:constant(1) (default:2)
  -persistent   0:non-persistent, 1:persistent (default:0)
  -as_br_cr      Choose between as_br_cr and as_bs_cr (default:false)
```
