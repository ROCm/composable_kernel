# Add + Rmsnorm2D + rowwise dynamic quantization forward

This folder contains example for add + Rmsnorm2D + rowwise dynamic quantization forward using ck_tile tile-programming implementation. Rdquant is short for rowwise dynamic quantization here.

## build
```
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_add_rmsnorm2d_rdquant_fwd -j`nproc`
```
This will result in an executable `build/bin/tile_add_rmsnorm2d_rdquant_fwd`

## cmdline
```
args:
          -m    m dimension (default:3328)
          -n    n dimension (default:4096)
     -stride    stride per row, if -1 then equal to n (default:-1)
          -e    epsilon (default:1e-5)
     -save_x    save rms(invrms) or not. set to 1 in training case (default:1)
          -v    cpu validation or not (default:1)
      -kname    print kernel name or not (default:1)
       -prec    precision (default:fp16)
      -quant    precision (default:int8)
     -warmup    cold iter (default:5)
     -repeat    hot iter (default:20)
       -json    0: No Json, 1: Dump Results in Json format (default:0)
   -jsonfile    json file name to dump results (default:add_rmsnorm2d_rdquant_fwd.json)
```
