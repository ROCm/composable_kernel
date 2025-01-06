# Cross GPU Reduce Communication
This folder contains example for different GPUs communicate with each other to complete the reduce. It is currently a test operator to verify and exam the communication between two GPUs.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
make example_cross_gpu_reduce -j
```
This will result in an executable `build/bin/example_cross_gpu_reduce`
