# Copy Kernel
This folder contains basic setup code designed to provide a platform for novice 
CK_Tile kernel developers to test basic functionality with minimal additional 
code compared to the functional code. Sample functional code for a simple
tile distribution for DRAM window and LDS window are provided and data is moved
from DRAM to registers, registers to LDS, LDS to registers and finally data
is moved to output DRAM window for a simple copy operation.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture 
# (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
# Make the copy kernel executable
make test_copy -j
```
This will result in an executable `build/bin/test_copy_kernel`

## example
```
args:
          -m        input matrix rows. (default 64)
          -n        input matrix cols. (default 8)
          -id       warp to use for computation. (default 0)
          -v        validation flag to check device results. (default 1)
          -prec     datatype precision to use. (default fp16)
          -warmup   no. of warmup iterations. (default 50)
          -repeat   no. of iterations for kernel execution time. (default 100)
```