# Image to Column

This folder contains example for Image to Column using ck_tile tile-programming implementation.

## build
```
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_example_img2col -j
```
This will result in an executable `build/bin/tile_example_img2col`
