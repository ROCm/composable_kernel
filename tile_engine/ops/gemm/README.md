# GEMM Matrix Multiplication

CK Tile Engine GEMM is used to generate and run GEMM kernels with different combinations of BlockTile sizes, WarpTile sizes, WarpTile mapping for all valid pipelines, schedulers and epilogues. 

# Kernel Configurations

Kernel parameters are specified in the `instance_combination.json` file, including matrix layouts, data types, padding settings, pipelines, schedulers, epilogues, and numerical values for tile and warp sizes.

Given a valid set of values, tile_engine_gemm will automatically iterate over all possible combinations of BlockTile and WarpTile sizes, as well as the specified pipelines, schedulers, and epilogues from `./configs/instance_combination.json`, and build the corresponding kernels.


## Build Instructions
``` bash
# in the root of composable kernel create build directory
mkdir build && cd build
# build composable kernel
sh ../script/cmake-ck-dev.sh  ../ <arch> # replace <arch> with the appropriate architecture (example gfx942) or leave blank
# generate the executable
make tile_engine_gemm -j
```
`tile_engine_gemm` will be located in the `./bin/` directory.

_`tile_engine_gemm` must be rebuilt everytime `instance_combination.json` is modified._
``` bash
rm -rf tile_engine/ && make tile_engine_gemm -j  # rebuild
```

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
-structured_sparsity    Sparsity for tensor - 0:false, 1:true (default: 0)
           -pipeline    possible values are: compv3, compv4, mem (default:compv3)
          -scheduler    possible values are: intrawave, interwave (default:intrawave)
           -epilogue    possible values are: cshuffle, default (default:cshuffle)
              -pad_m    Pad in m direction - true/false (default:false)
              -pad_n    Pad in n direction - true/false (default:false)
              -pad_k    Pad in k direction - true/false (default:false)

Note: pipeline, scheduler, epilogue, pad_m, pad_n, pad_k should be one of the options specified in instance_combination.json 
```
Note: In `./configs/instance_combination.json` pipeline, scheduler, epilogue, pad_m, pad_n, pad_k should be from one of the values specified above. 

## Example

The following JSON file specifies parameters used to generate and build GEMM kernels across all possible combinations of pipelines, schedulers, epilogues with different tile and warp sizes.

```json
{     
    /// other parameters ///
    
    "tile_m": {
      "values": [256]
    },
    "tile_n": {
      "values": [256]
    },
    "tile_k": {
      "values": [64, 32]
    },

    /// other parameters ///

    "pipeline": {
      "values": ["compv3", "compv4", "mem"]
    },
    "scheduler": {
      "values": ["intrawave", "interwave"]
    },
    "epilogue": {
      "values": ["default", "cshuffle"]
    }
}
```

At runtime, a specific subset of the generated kernels can be selected using command-line arguments.
``` bash
./bin/tile_engine_gemm -pipeline=compv3 -scheduler=intrawave -epilogue=default 
```
The above command runs kernels configured with the compv3 pipeline, intrawave scheduler, and default epilogue, while sweeping over different BlockTile sizes, WarpTile sizes, and WarpTile mappings.

