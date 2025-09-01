# FLATMM Matrix Multiplication

This folder contains example for FLATMM using ck_tile tile-programming implementation. Currently, it only supports the basic feature of the CK Tile FLATMM, but creates the placeholders for the future support on different FLATMM pipeline and different FLATMM modules. In the near future, we will gradually migrate all the FLATMM features from old CK to CK Tile.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
# The basic pipeline method on the flatmm calculation
make tile_example_flatmm_basic -j
```
This will result in an executable `build/bin/tile_example_flatmm_basic`

## example
```
args:
          -m    m dimension (default:256)
          -n    n dimension (default:256)
          -k    k dimension (default:128)
   -a_layout    A tensor data layout - Row by default (default:R)
   -b_layout    B tensor data layout - Row by default (default:C)
   -c_layout    C tensor data layout - Row by default (default:R)
   -stride_a    Tensor A stride (default:0)
   -stride_b    Tensor B stride (default:0)
   -stride_c    Tensor C stride (default:0)
          -v    0. No validation, 1. Validation on CPU, 2. Validation on GPU (default:1)
       -prec    data type. fp16/bf16/fp8/bf8 (default:fp16)
     -warmup    number of iterations before benchmark the kernel (default:50)
     -repeat    number of iterations to benchmark the kernel (default:100)
      -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
    -split_k    splitK value (default:1)
       -init    0:random, 1:linear, 2:constant(1) (default:0)
  -warp_tile    0: 16x16, 1: 32x32, 2: 16x16x128 (950 only), 3: 32x32x64 (950 only) (default:0)
       -json    0: No Json, 1: Dump Results in Json format (default:0)
   -jsonfile    json file name to dump results (default:flatmm_basic.json)
```
