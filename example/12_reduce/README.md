# Instructions for ```example_reduce_blockwise```

## Run ```example_reduce_blockwise```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg2: run kernel # of times (>1)
./bin/example_reduce_blockwise -D 16,64,32,960 -v 1 1 10
```

Result
```
launch_and_time_kernel: grid_dim {240, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 3 times...
Perf: 0.23536 ms, 267.32 GB/s, DeviceReduceBlockWise<256,M_C4_S1,K_C64_S1,InSrcVectorDim_0_InSrcVectorSize_1_OutDstVectorSize_1>
error: 0
max_diff: 0, 529, 529
root@dc-smc-18:/data/composable_kernel/Build3# bin/example_reduce_blockwise -D 16,64,32,960 -v 1 1 10
launch_and_time_kernel: grid_dim {240, 1, 1}, block_dim {256, 1, 1} 
Warm up
Start running 10 times...
Perf: 0.23392 ms, 268.966 GB/s, DeviceReduceBlockWise<256,M_C4_S1,K_C64_S1,InSrcVectorDim_0_InSrcVectorSize_1_OutDstVectorSize_1>
error: 0
max_diff: 0, 528, 528
```
