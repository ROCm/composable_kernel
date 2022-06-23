# Instructions for ```example_multiple_reduce```

## Run ```example_multiple_reduce```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg2: time kernel (0=no, 1=yes) 
./bin/example_multiple_reduce -D 16,64,32,960 -v 1 1 1
```

Result
```
./bin/example_multiple_reduce                          
launch_and_time_kernel: grid_dim {150, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 1.28875 ms, 186.886 GB/s, DeviceMultipleReduceMultiBlockAtomicAdd<256,M_C4_S1,K_C64_S1,InSrcVectorDim_1_InSrcVectorSize_1_OutDstVectorSize_1>
```

