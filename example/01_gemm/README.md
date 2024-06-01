# Instructions for ```example_gemm_xdl```

## Build

After following instructions to build the CK library:

cd build
make example_gemm_xdl_streamk

## Run ```example_gemm_xdl```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: time kernel # (0=no, 1=yes)
#arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC"
#arg10 (optional): Number of CU's (NOTE: some examples do not function correctly with non-default arg10)
./bin/example_gemm_xdl 1 2 1
```
## Structure of example program (StreamK example)

1. /example/01_gemm/gemm_xdl_streamk.cpp is the executable, and basically just takes a bunch of parameters and starts everything else.
2. /example/01_gemm/run_gemm_example.ic is the test file that builds all the test data and outputs measurements.
3. /include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_streamk.hpp is the logic to invoke the kernels, but not the actual kernel logic.
4. /include/ck/tensor_operation/grid/gridwise_gemm_xdlops_streamk.hpp is the actual kernel logic.
5. /include/ck/host_utility/kernel_launch.hpp is the logic that launches the kernels.
6. /include/ck/stream_config.hpp is a file where you can change the # of warmups/number of times it runs.

## Changes in PR:

Set padding = 0 in lines 459-461 of /include/ck/tensor_operations/grid_gridwise_gemm_xdlops_streamk.hpp