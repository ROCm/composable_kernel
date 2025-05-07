# GEMM Matrix Multiplication

Use the files in this folder to generate and build applications that run Matrix multiplications using ck_tile programming based on the kernel parameters mentioned in the config file.

# Gemm Problem

User needs to provide gemm problem such as datatype, layout in the config file. For reference please see `./configs/gemm_problem.json`.


# Kernel Configurations

User can provide kernel configuration such as tile size, warp size, padding, pipeline, scheduler and epilogue in the config file. For reference please see `./configs/user_provide_config.json`. The Tile engine also has default kernel configuration to expand the range of kernel configuration which is saved in `./configs/default.json`.


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
   -pipeline    The type of pipeline. Possible values are compv3, compv4 or mem. Default is compv3.     
   -epilogue    The type of epilogue. Possible values are cshuffle or default. Default is csshuffle.
      -pad_m    Whether pad or not in m direction. Possible values are true or false. Default is false. 
      -pad_n    Whether pad or not in n direction. Possible values are true or false. Default is false. 
      -pad_k    Whether pad or not in k direction. Possible values are true or false. Default is false. 

Note: pipeline, scheduler, epilogue, pad_m, pad_n, pad_k should be one of the options specified in instance_combination.json 
```

## Example

Below example will run gemm kernel with default dimensions of matrices, for compv3 pipeline, intrawave scheduler and default epilogue with all possible tile sizes mentioned in Config file.

```
./bin/tile_engine_gemm -pipeline=compv3 -scheduler=intrawave -epilogue=default 
```
