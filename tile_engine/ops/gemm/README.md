# GEMM Matrix Multiplication

CK Tile Engine GEMM is used to generate and run GEMM kernels with different combinations of BlockTile sizes, WarpTile sizes, WarpTile mapping for all valid pipelines, schedulers and epilogues. 

# Kernel Configurations

ser can provide kernel configuration such as tile size, warp size, padding, pipeline, scheduler and epilogue in the config file. For reference please see `./configs/user_provided_config.json`. The Tile engine also has default kernel configuration to expand the range of kernel configuration which is saved in `./configs/default_config.json`.

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
                      -m    The value for m dimension. Default is 3840.
                      -n    The value for n dimension. Default is 4096.
                      -k    The value for k dimension. Default is 2048.
              -stride_a    The stride value for tensor A. Default is 0.
              -stride_b    The stride value for tensor B. Default is 0.
              -stride_c    The stride value for tensor C  Default is 0.
                -split_k    The split value for k dimension. Default is 1.
                      -v    The type of validation. Set to 0 for no validation, 1 for validation on CPU, or 2 for validation on GPU. Default is 2, validation on GPU.
                -warmup    The number of iterations before benchmark the kernel. Default is 50.
                -repeat    The number of iterations to benchmark the kernel. Default is 100.
                  -timer    The type of timer. Possible values are gpu timer or cpu timer. Default is gpu timer.
                  -init    The method of tensor initialization. Set to 0 for random, to 1 for linear, or 2 for constant(1). Default is 0, random.
                -metric    Metric with which to measure kernel performance. Set to 0 for latency, 1 for tflops, or 2 for bandwidth. Default is 0, latency.
   -structured_sparsity    whether use sparsity kernel or not. Possible values are true or false. Default is false.
              -pipeline    The type of pipeline. Possible values are compv3, compv4 or mem. Default is compv3.     
              -epilogue    The type of epilogue. Possible values are cshuffle or default. Default is cshuffle.
                 -pad_m    Whether pad or not in m direction. Possible values are true or false. Default is false. 
                 -pad_n    Whether pad or not in n direction. Possible values are true or false. Default is false. 
                 -pad_k    Whether pad or not in k direction. Possible values are true or false. Default is false. 

Note: pipeline, scheduler, epilogue, pad_m, pad_n, pad_k should be one of the options specified in user_provided_config.json 
```
Note: In `./configs/user_provided_config.json` pipeline, scheduler, epilogue, pad_m, pad_n, pad_k should be from one of the values specified above. 

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

