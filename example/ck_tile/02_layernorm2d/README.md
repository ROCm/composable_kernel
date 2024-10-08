# Layernorm2D forward

This folder contains example for Layernorm2D forward using ck_tile tile-programming implementation.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
make tile_example_layernorm2d_fwd -j
```
This will result in an executable `build/bin/tile_example_layernorm2d_fwd`

## example
```
args:
          -m    m dimension (default:3328)
          -n    m dimension (default:4096)
          -e    epsilon (default:1e-5)
          -v    cpu validation or not (default:1)
       -prec    precision (default:fp16)
```