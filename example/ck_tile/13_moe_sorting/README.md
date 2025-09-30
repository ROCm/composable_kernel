# moe-sorting

This folder contains example for moe-sorting kernel using ck_tile tile-programming implementation. This kernel is often used in Moe model, before launching the fused-moe-gemm block. The input&weight is a `token*topk` 2d matrix. The op rearange the input weight ids into different experts and feed into fuse moe gemm kernel.

## build
```
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_example_moe_sorting -j`nproc`
```
This will result in an executable `build/bin/tile_example_moe_sorting`

## example
```
args:
                 -v    turn CPU validation on (1) or off (0). (default:1)
              -pr_i    index data type.  Only int32 is currently supported. (default:int32)
              -pr_w    output weight data type. Only fp32 is currently supported. (default:fp32)
                 -t    number of input tokens. (default:128)
                       If "local_t" presents, this value indicates global concurrency of all ranks.
           -local_t    Number of local input tokens for curent rank. (default:-1)
                       This value must be within range "[0, t)", or "-1"(no such feature)
                       This feature is to simulate EP case where where each rank has different tokens.
                       Besides, this value will be stored in a GPU buffer, which is friendly for CUDA graph.
                 -e    number of num_experts (default:8)
                 -k    topk (default:4)
              -unit    unit_size (default:32)
-moe_buf_interm_dim    interm_dim(col) of the following fmoe buf (default:0)
-moe_buf_elem_bytes    fmoe buf element byte size, 1:8bit, 2:16bit, 4:32bit... (default:2)
                -ci    clear workspace inside API or not(if "0", require manually clear outside) (default:1)
          -dispatch    dispatch policy. 0:automatically pick up kernel, 1:use single kernel, 2:use mp kernel (default:0)
         -local_eid    a list of experts enabled as local expert. e.g. "0,1,4,5" (default:-1)
                       please make sure eid is in ascending order!
              -seed    seed to be used. When set to -1, a random seed will be generated each time invoking this example (default:-1)
             -kname    prints the kernel name when set to 1 (default:0)
            -warmup    number of iterations before benchmark the kernel (default:5)
            -repeat    number of iterations to benchmark the kernel (default:20)
              -json    0: No Json, 1: Dump Results in Json format (default:0)
          -jsonfile    json file name to dump results (default:moe_sorting.json)
```
