# smoothquant

This folder contains example for smoothquant using ck_tile tile-programming implementation.

## build
```
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_smoothquant -j`nproc`
```
This will result in an executable `build/bin/tile_smoothquant`

## cmdline
```
args:
          -m    m dimension (default:3328)
          -n    n dimension (default:4096)
   -x_stride    input stride per row, if -1 then equal to n (default:-1)
   -y_stride    output stride per row, if -1 then equal to n (default:-1)
          -v    cpu validation or not (default:1)
      -kname    print kernel name or not (default:1)
       -prec    precision (default:fp16)
     -warmup    cold iter (default:5)
     -repeat    hot iter (default:20)
       -json    0: No Json, 1: Dump Results in Json format (default:0)
   -jsonfile    json file name to dump results (default:smoothquant.json)
```
