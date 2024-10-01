# GEMM Matrix Multiplication

This folder contains example for GEMM using ck_tile tile-programming implementation. Currently, it only supports the basic feature of the CK Tile GEMM, but creates the placeholders for the future support on different GEMM pipeline and different GEMM modules. In the near future, we will gradually migrate all the GEMM features from old CK to CK Tile.

## build
```
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_example_gemm_basic -j
```
This will result in an executable `build/bin/tile_example_gemm_basic`

## example
```
args:
          -m    m dimension (default:3328)
          -n    m dimension (default:4096)
          -k    k dimension (default:64)
          -e    epsilon (default:1e-5)
          -v    cpu validation or not (default:1)
       -prec    precision (default:fp16)
```
