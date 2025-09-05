# Batched GEMM

This folder contains example for batched GEMM using ck_tile tile-programming implementation.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
make tile_example_batched_gemm -j
```
This will result in an executable `build/bin/tile_example_batched_gemm`

## example
```
args:
               -m    m dimension (default:512)
               -n    n dimension (default:1024)
               -k    k dimension (default:2048)
         -stride_a    Tensor A stride (default:0)
         -stride_b    Tensor B stride (default:0)
         -stride_c    Tensor C stride (default:0)
         -a_layout    A tensor data layout - Row by default (default:R)
         -b_layout    B tensor data layout - Row by default (default:C)
         -c_layout    C tensor data layout - Row by default (default:R)
   -batch_stride_a    Batch A stride (default:1048576)
   -batch_stride_b    Batch B stride (default:2097152)
   -batch_stride_c    Batch C stride (default:524288)
      -batch_count    Batch count (default:8)
               -v    0. No validation, 1. Validation on CPU, 2. Validation on GPU (default:2)
            -prec    data type. fp16/bf16/fp8/bf8 (default:fp16)
         -warmup    number of iterations before benchmark the kernel (default:50)
         -repeat    number of iterations to benchmark the kernel (default:100)
            -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
         -split_k    splitK value (default:1)
            -json    0: No Json, 1: Dump Results in Json format (default:0)
         -jsonfile    json file name to dump results (default:cktile_batched_gemm.json)
```