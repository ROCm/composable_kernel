# GEMM Matrix Multiplication

Use the files in this folder to generate and build applications that run Matrix multiplications using ck_tile programming based on the kernel parameters mentioned in the config file `./configs/instance_combination.json`.

# Kernel Configurations

User needs to provide kernel configuration such as datatype, layout, tile size, warp size, padding, pipeline, scheduler and epilogue in the config file. For reference please see `./configs/instance_combination.json`

## Build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
# To generate the executable
make tile_engine_gemm -j
```
`tile_engine_gemm` will be located in the `./bin/` directory.

## tile_engine_gemm inputs
```

          -m    m dimension (default:3840)
          -n    n dimension (default:4096)
          -k    k dimension (default:2048)
   -stride_a    Tensor A stride (default:0)
   -stride_b    Tensor B stride (default:0)
   -stride_c    Tensor C stride (default:0)
    -split_k    SplitK value (default:1)
          -v    No validation: 0, Validation on CPU: 1, Validation on GPU: 2 (default:2)
     -warmup    Number of iterations before benchmark the kernel (default:50)
     -repeat    Number of iterations to benchmark the kernel (default:100)
      -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
       -init    Value for initializing tensor- random: 0, linear: 1, constant(1): 2 (default:0)
   -pipeline    possible values are: compv3, compv4, mem (default:compv3)
  -scheduler    possible values are: intrawave, interwave (default:intrawave)
   -epilogue    possible values are: cshuffle, default (default:cshuffle)
      -pad_m    Pad in m direction - true/false (default:false)
      -pad_n    Pad in n direction - true/false (default:false)
      -pad_k    Pad in k direction - true/false (default:false)

Note: pipeline, scheduler, epilogue, pad_m, pad_n, pad_k should be one of the options specified in instance_combination.json 
```

## Example

Below example will run gemm kernel with default dimensions of matrices, for compv3 pipeline, intrawave scheduler and default epilogue with all possible tile sizes mentioned in Config file.

```
./bin/tile_engine_gemm -pipeline=compv3 -scheduler=intrawave -epilogue=default 
```
