#Multiple D GEMM

This folder contains example for Multiple D GEMM using ck_tile tile-programming implementation.

## build
```
#in the root of ck_tile
mkdir build && cd build
#you can replace < arch> with the appropriate architecture(for example gfx90a or gfx942) or \
    leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
#The basic pipeline method on the gemm calculation
make tile_example_gemm_multi_d_fp16 -j
```
This will result in an executable `build/bin/tile_example_gemm_multi_d_fp16`

## example
```
args:
          -m    m dimension (default:3840)
          -n    n dimension (default:4096)
          -k    k dimension (default:4096)
   -a_layout    A tensor data layout - Row by default (default:R)
   -b_layout    B tensor data layout - Col by default (default:C)
  -ds_layout    Ds tensor data layout - Row by default (default:R)
   -e_layout    E tensor data layout - Row by default (default:R)
   -stride_a    Tensor A stride (default:0)
   -stride_b    Tensor B stride (default:0)
  -stride_ds    Tensor Ds stride (default:0)
   -stride_e    Tensor E stride (default:0)
          -v    0. No validation, 1. Validation on GPU (default:1)
     -warmup    number of iterations before benchmark the kernel (default:50)
     -repeat    number of iterations to benchmark the kernel (default:100)
     -kbatch    kbatch for SplitK (default:1)
       -json    0: No Json, 1: Dump Results in Json format (default:0)
   -jsonfile    json file name to dump results (default:cktile_gemm_multi_d_fp16.json)
```
