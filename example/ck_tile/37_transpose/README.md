# Batched Transpose
This folder contains example for transpose load for architecture gfx950. This transpose load has some constraints in input tile distribution.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
# Make the transpose executable
make tile_example_transpose -j
```
This will result in an executable `build/bin/tile_example_transpose`

## example
```
args:
          -N    input batch size (default:2)
          -C    input channel size. (default:64)
          -H    input height size. (default:1)
          -W    input width size. (default:64)
          -v    whether do CPU validation or not (default: 1)
  -layout_in    input tensor data layout - NCHW by default
 -layout_out    output tensor data layout - NHWC by default
       -seed    seed to be used, -1 means random every time (default:-1)
     -k_name    t to 1 will print kernel name (default:0)
```