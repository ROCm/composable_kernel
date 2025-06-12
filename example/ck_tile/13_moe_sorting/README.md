# moe-sorting

This folder contains example for moe-sorting kernel using ck_tile tile-programming implementation. This kernel is often used in Moe model, before launching the fused-moe-gemm block. The input&weight is a `token*topk` 2d matrix. The op rearange the input weight ids into different experts and feed into fuse moe gemm kernel.

## build
```
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_example_moe_sorting -j
```
This will result in an executable `build/bin/tile_example_moe_sorting`

## example
```
args:
           -v    weather do CPU validation or not (default:1)
        -pr_i    index data type. (currently only int32 supported now) (default:int32)
        -pr_w    output weight data type(currently only fp32 supported now) (default:fp32)
           -t    number of input tokens (default:128)
     -local_t    number of local input tokens, dynamic token feature,  (default:-1)
                 will be used for EP case where each rank have different tokens.
                 This value will be stored in GPU buffer for cuda graph usage. if -1, then no this value/buffer
           -e    number of num_experts (default:8)
           -k    topk (default:4)
        -unit    unit_size (default:32)
-moe_buf_size    moe_buf_size (default:0)
   -local_eid    a list of experts enabled as local expert. e.g. "0,1,4,5" (default:-1)
                 please make sure eid is in ascending order!
        -seed    seed to be used, -1 means random every time (default:-1)
       -kname    when set to 1 it will print kernel name (default:0)
      -warmup    number of iterations before benchmark the kernel (default:5)
      -repeat    number of iterations to benchmark the kernel (default:20)
```
