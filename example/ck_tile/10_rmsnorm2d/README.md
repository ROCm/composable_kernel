# Rmsnorm2D forward

This folder contains example for Rmsnorm2D forward using ck_tile tile-programming implementation.

## build
```
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_rmsnorm2d_fwd -j`nproc`
```
This will result in an executable `build/bin/tile_rmsnorm2d_fwd`

## cmdline
```
args:
           -m    m dimension (default:3328)
           -n    n dimension (default:4096)
    -x_stride    x row_stride, if -1 then equal to n (default:-1)
   -xr_stride    x residule row_stride, if -1 then equal to n (default:-1)
    -y_stride    y row_stride, if -1 then equal to n (default:-1)
   -yr_stride    y residule row_stride, if -1 then equal to n (default:-1)
           -e    epsilon (default:1e-5)
    -save_rms    save rms(invrms) or not. set to 1 in training case (default:0)
-save_unquant    save result before quant (default:0)
           -v    cpu validation or not (default:1)
       -kname    print kernel name or not (default:1)
      -prec_i    input precision (default:fp16)
      -prec_o    output precision, set auto will be the same as input (default:auto)
     -prec_sm    output quant scale type, set auto will use fp32. used when fquant=1 (default:auto)
     -prec_sy    output quant scale type, set auto will use fp32. used when fquant=1 or 2 (default:auto)
        -fadd    fused-add, 0:no fused add, 1:preadd+store, 2:preadd only (default:0)
      -fquant    fused-quant, 0:no, 1:smooth-dynamic-quant, 2:dynamic-quant (default:0)
      -warmup    cold iter (default:5)
      -repeat    hot iter (default:20)
           -s    sensitive model mode, 0: for no specific model, 1: for T5-like model (default:0)
        -json    0: No Json, 1: Dump Results in Json format (default:0)
    -jsonfile    json file name to dump results (default:rmsnorm2d_fwd.json)
```
