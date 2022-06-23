# Instructions for ```example_reduce_blockwise```

## Run ```example_reduce_blockwise```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg2: time kernel (0=no, 1=yes)
./bin/example_reduce_blockwise -D 16,64,32,960 -v 1 1 1
```

Result
```
./bin/example_reduce_blockwise -D 16,64,32,960 -v 1 1 1
launch_and_time_kernel: grid_dim {240, 1, 1}, block_dim {256, 1, 1}
Warm up 1 time
Start running 10 times...
Perf: 0.282592 ms, 222.641 GB/s, DeviceReduceBlockWise<256,M_C4_S1,K_C64_S1,InSrcVectorDim_0_InSrcVectorSize_1_OutDstVectorSize_1>
```

# Instructions for ```example_reduce_blockwise_two_call```

## Run ```example_reduce_blockwise_two_call```
```bash
#arg1:  verification (0=no, 1=yes(
#arg2:  initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3:  time kernel (0=no, 1=yes)
./bin/example_reduce_blockwise_two_call 1 2 1
```

Result
```
./bin/example_reduce_blockwise_two_call 1 2 1
launch_and_time_kernel: grid_dim {204800, 1, 1}, block_dim {256, 1, 1}
Warm up 1 time
Start running 10 times...
launch_and_time_kernel: grid_dim {6400, 1, 1}, block_dim {256, 1, 1}
Warm up 1 time
Start running 10 times...
Perf: 2.1791 ms, 771.42 GB/s, DeviceReduceBlockWise<256,M_C32_S1,K_C8_S1,InSrcVectorDim_1_InSrcVectorSize_1_OutDstVectorSize_1> => DeviceReduceBlockWise<256,M_C256_S1,K_C1_S1,InSrcVectorDim_1_InSrcVectorSize_1_OutDstVectorSize_1>
```
