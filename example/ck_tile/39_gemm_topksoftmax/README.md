# GEMM + Topksoftmax Implementation

This folder contains example for GEMM + Topksoftmax using ck_tile tile-programming implementation.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
make tile_example_gemm_topksoftmax -j
```
This will result in an executable `build/bin/tile_example_gemm_topksoftmax`

## example
```
args:
          -b    batch size (default:1)
          -m    m dimension (number of input tokens, default:3840)
          -n    n dimension (number of experts, default:4096)
          -k    k dimension (default:2048)
          # (group)topksoftmax args
          -topk    topk (default:8)

   -a_layout    Tensor A data layout (default: R)
   -b_layout    Tensor B data layout (default: C)
   -c_layout    Tensor C data layout (default: R)
   -stride_a    Tensor A stride (default:0)
   -stride_b    Tensor B stride (default:0)
   -stride_c    Tensor C stride (default:0)
          -v    0. No validation, 1. Validation on CPU, 2. Validation on GPU (default:2)
          -e    Absolute error tolerance (default:1e-5)
       -prec    data type. fp16/bf16 (default:fp16)
     -warmup    number of iterations before benchmark the kernel (default:10)
     -repeat    number of iterations to benchmark the kernel (default:100)
      -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
```
