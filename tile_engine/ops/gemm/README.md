# GEMM Matrix Multiplication

CK Tile Engine GEMM is used to generate and run GEMM kernels with different combinations of BlockTile sizes, WarpTile sizes, WarpTile mapping for all valid pipelines, schedulers and epilogues. 

# Kernel Configurations

Users can specify custom kernel configurations such as tile size, warp size, padding, pipeline, scheduler, and epilogue in the config file. This allows building only for selected configurations, significantly reducing build time.
For reference please see `./configs/user_provided_config.json`.


The Tile engine also has a default kernel configuration for providing range of configuration parameter values, which helps users who lack kernel development experience to benchmark. For reference please see in `./configs/default_config.json`

If user does not provide kernel configuration, the tile engine uses default kernel configuration to generate kernel instances and benchmark. 

## Build Instructions
``` bash
# in the root of composable kernel create build directory
mkdir build && cd build
# build composable kernel
# replace [Arch] with the appropriate architecture or leave blank and 
# replace [Datatype1;Datatype2;...] in comma separated datatypes string (possible datatypes are [fp8, bf8, int8, fp16, bf16])
# replace [Layout1;Layout2;...] in comma separated datatypes string (possible layouts are [rcr, rrr, crr, ccr])
sh ../script/cmake-ck-dev.sh  ../ [Arch] -DGEMM_DATATYPE="[Datatype1;Datatype2]" -DGEMM_LAYOUT="[Layout1;Layout2]"
# generate different executable for each passed datatype
make benchmark_gemm_[Datatype1]_[Layout1] -j
make benchmark_gemm_[Datatype1]_[Layout2] -j
make benchmark_gemm_[Datatype2]_[Layout1] -j
make benchmark_gemm_[Datatype2]_[Layout2] -j
```
`benchmark_gemm_[Datatype]_[Layout]` will be located in the `./bin/` directory.

`benchmark_gemm_[Datatype]_[Layout]` must be rebuilt everytime if configuration file is modified.

``` bash
rm -rf tile_engine/ && make benchmark_gemm_[Datatypes]_[Layout] -j  # rebuild
```

## For eaxmple build for gfx942 for fp8 and fp16 datatypes with rcr layout
``` bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ gfx942 -DGEMM_DATATYPE="fp8;fp16" -DGEMM_LAYOUT="rcr" 
make benchmark_gemm_fp8_rcr -j
make benchmark_gemm_fp16_rcr -j
```

## benchmark_gemm inputs
```
                      -m    The value for m dimension. Default is 3840.
                      -n    The value for n dimension. Default is 4096.
                      -k    The value for k dimension. Default is 2048.
               -stride_a    The stride value for tensor A. Default is 0.
               -stride_b    The stride value for tensor B. Default is 0.
               -stride_c    The stride value for tensor C  Default is 0.
                -split_k    The split value for k dimension. Default is 1.
                      -v    The type of validation. Set to 0 for no validation, 1 for validation on CPU, or 2 for validation on GPU. Default is 2, validation on GPU.
                    -log    Wether output kernel instance information or not. Possible values are true or false. Default is false.
                 -warmup    The number of iterations before benchmark the kernel. Default is 50.
                 -repeat    The number of iterations to benchmark the kernel. Default is 100.
                  -timer    Whether if the timer is gpu timer or not. Possible values are true or false. Default is true.  
                   -init    The method of tensor initialization. Set to 0 for random, to 1 for linear, or 2 for constant(1). Default is 0, random.
            -flush_cache    To flush cache in between different runs.Possible values are true or false. Default is false.
         -rotating_count    count to flush cache. Default is 5.     
                 -metric    Metric with which to measure kernel performance. Set to 0 for latency, 1 for tflops, or 2 for bandwidth. Default is 0, latency.
           -csv_filename    The filename of benchmark result. Default is gemm_kernel.
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
./bin/benchmark_gemm_[Datatype]_[Layout] -pipeline=compv3 -scheduler=intrawave -epilogue=default 
```
The above command runs kernels configured with the compv3 pipeline, intrawave scheduler, and default epilogue, while sweeping over different BlockTile sizes, WarpTile sizes, and WarpTile mappings.

