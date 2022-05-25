# Instructions for ```example_lnorm_use_reduce```

## Run ```example_lnorm_use_reduce```
```bash
# -D <xxx> : input 4-d tensor lengths (nhwc layout)
# -v <x> :   verification (0=no, 1=yes)
#arg1: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg2: time kernel (0=no, 1=yes) 
./bin/example_lnorm_use_reduce -D 512,28,28,256 -v 1 1 1
```

Result
```
./bin/example_lnorm_use_reduce -D 512,28,28,256 -v 1 1 0 
```
root@dc-smc-18:/data/work/composable_kernel/Build3# bin/example_lnorm_use_reduce -D 512,28,28,256 -v 1 1 0
Perf: 0 ms, inf GB/s, DeviceReduceMultiBlockAtomicAdd<256,M_C16_S1,K_C16_S1,InSrcVectorDim_1_InSrcVectorSize_1_OutDstVectorSize_1> + DeviceReduceMultiBlockAtomicAdd<256,M_C16_S1,K_C16_S1,InSrcVectorDim_1_InSrcVectorSize_1_OutDstVectorSize_1>


