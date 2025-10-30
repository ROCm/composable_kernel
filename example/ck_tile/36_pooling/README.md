# Pooling Operator

This folder contains example for the pooling operator using ck_tile tile-programming implementation. Currently the pooling kernel only supports 2D and 3D pooling.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
# The 3D pooling example
make tile_example_pool3d -j`nproc`
```
This will result in an executable `build/bin/tile_example_pool3d`

## example
```
args:
          -N    batch size (default:2)
          -D    depth dimension (default:30)
          -H    height dimension (default:30)
          -W    width dimension (default:30)
          -C    channel dimension (default:32)
          -Z    pooling window depth (default:2)
          -Y    pooling window height (default:2)
          -X    pooling window width (default:2)
         -Sz    window stride depth (default:2)
         -Sy    window stride height (default:2)
         -Sx    window stride width (default:2)
         -Dz    window dilation depth (default:1)
         -Dy    window dilation height (default:1)
         -Dx    window dilation width (default:1)
     -LeftPz    left padding depth (default:1)
     -LeftPy    left padding height (default:1)
     -LeftPx    left padding width (default:1)
    -RightPz    right padding depth (default:1)
    -RightPy    right padding height (default:1)
    -RightPx    right padding width (default:1)
          -v    0: No validation, 1: CPU validation (default:1)
     -warmup    number of iterations before benchmark (default:0)
     -repeat    number of iterations to benchmark (default:1)
```
