CFLAGS=-I ~/workspace/composable_kernel/include -I /opt/workspace/rocm-5.1.1/hip/include -I ~/workspace/composable_kernel/include/ -I ~/workspace/composable_kernel/include/ck/ -I ~/workspace/composable_kernel/example/01_gemm/ -I ~/workspace/composable_kernel/library/include/  -I ~/workspace/composable_kernel/library/src/utility/ -I ~/workspace/composable_kernel/include/ck/problem_transform/ -I ~/workspace/composable_kernel/include/ck/tensor/ -I ~/workspace/composable_kernel/include/ck/tensor_description/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/block/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/device/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/device/impl/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/element/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/grid/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/thread/ -I ~/workspace/composable_kernel/include/ck/tensor_operation/gpu/warp/ -I ~/workspace/composable_kernel/include/ck/host_utility -I /external/include/half/ -I ~/workspace/composable_kernel/library/include/ck/library/host/ -I ~/workspace/composable_kernel/library/include/ck/library/host_tensor/ -I ~/workspace/composable_kernel/library/include/ck/library/obselete_driver_offline/ -I ~/workspace/composable_kernel/library/include/ck/library/reference_tensor_operation/cpu/ -I ~/workspace/composable_kernel/library/include/ck/library/reference_tensor_operation/gpu/ -I ~/workspace/composable_kernel/library/include/ck/library/tensor_operation_instance/ -I ~/workspace/composable_kernel/library/include/ck/library/tensor_operation_instance/gpu/" + "reduce/ -I ~/workspace/composable_kernel/library/include/ck/library/tensor_op/ -I ~/workspace/composable_kernel/library/include/ck/library/utility/ -I ~/workspace/composable_kernel/profiler/include/ 

CXXFLAGS = -std=c++17
gemm: ex.o host_tensor.o device_memory.o
	hipcc $(CXXFLAGS) $(CFLAGS) ex.o host_tensor.o device_memory.o -o gemm

device_memory.o: ../../../../library/src/utility/device_memory.cpp
	hipcc $(CXXFLAGS) $(CFLAGS) -c ../../../../library/src/utility/device_memory.cpp

host_tensor.o: ../../../../library/src/utility/host_tensor.cpp
	hipcc $(CXXFLAGS) $(CFLAGS) -c ../../../../library/src/utility/host_tensor.cpp

ex.o: 
	hipcc -fPIC -fvisibility=hidden $(CXXFLAGS) -w /opt/rocm-5.3.0/amdgcn/bitcode/oclc_abi_version_400.bc $(CFLAGS) -L/opt/rocm-5.3.0/rocrand -lrocrand -x hip -c  ex.cpp
