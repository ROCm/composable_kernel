#Multiple D GEMM

This folder contains example for Multiple D GEMM using ck_tile tile-programming implementation.

## build
```
#in the root of ck_tile
mkdir build && cd build
#you can replace < arch> with the appropriate architecture(for example gfx90a or gfx942) or \
    leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
#The basic pipeline method on the gemm calculation
make tile_example_gemm_multi_d_fp16 -j
```
This will result in an executable `build/bin/tile_example_gemm_multi_d_fp16`

## example
```
args:
       -m  M dimensions - (Default: 3840)
       -n  N dimensions - (Default: 4096)
       -k  K dimensions - (Default: 4096)
-a_layout  Tensor A layout (default:R)
-b_layout  Tensor B layout (default:C)
-ds_layout Tensor D layout (default:R)
-e_layout  Tensor E layout (default:R)
-stride_a  Tensor A strides - (Default: 0)
-stride_b  Tensor B strides - (Default: 0)
-stride_e  Tensor C strides - (Default: 0)
-stride_ds Tensor D strides - (Default: 0)
-validate  0. No validation, 1. Validation on GPU. (Default: 1)
  -warmup  Number of iterations before benchmark the kernel. (Default: 10)
  -repeat  Number of iterations to benchmark the kernel. (Default: 100)
  -kbatch  kbatch for SplitK. (Default 1)
```
