# Instructions for ```batchnorm nhwc``` Example

## Run ```batchnorm forward nhwc```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: save result mean/invVariance (0=no, 1=yes)
#arg2: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3: time kernel (0=no, 1=yes) 
./bin/example_batchnorm_forward -D 128,16,16,1024 -v 1 1 2 1
```

Result 
```
bin/example_batchnorm_forward -D 128,16,16,1024 -v 1 1 2 1
launch_and_time_kernel: grid_dim {64, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
launch_and_time_kernel: grid_dim {120, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
launch_and_time_kernel: grid_dim {120, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 2.42686 ms, 304.187 GB/s


## Run ```batchnorm infer nhwc```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg2: time kernel (0=no, 1=yes)
bin/example_batchnorm_infer -D 128,16,16,1024 -v 1  2 1
```

Result
```
./bin/example_batchnorm_infer -D 128,16,16,1024 -v 1  2 1
launch_and_time_kernel: grid_dim {120, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 1.47628 ms, 454.58 GB/s


## Run ```batchnorm backward nhwc```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: use saved mean/invVariance (0=no, 1=yes)
#arg2: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3: time kernel (0=no, 1=yes)
./bin/example_batchnorm_backward -D 128,16,16,1024 -v 1 1 3 0

Result (no timing since some kernel can only be executed once due to its using same buffer for both input and output)
```
./bin/example_batchnorm_backward -D 128,16,16,1024 -v 1 1 3 0
echo $?
0
