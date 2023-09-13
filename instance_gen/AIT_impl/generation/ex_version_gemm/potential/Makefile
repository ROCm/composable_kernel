CC = /opt/rocm/bin/hipcc
CK_PATH=/dockerx/composable_kernel/
CFLAGS = -O3 -std=c++17 -DCK_AMD_GPU_GFX90A --offload-arch=gfx90a -I"${CK_PATH}/include" -I"${CK_PATH}/library/include" -I"${CK_PATH}/profiler/include"
OBJS = ex.o host_tensor.o device_memory.o

all: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o ex

device_memory.o: ../../library/src/utility/device_memory.cpp
	$(CC) $(CFLAGS) -c ../../library/src/utility/device_memory.cpp

host_tensor.o: ../../library/src/utility/host_tensor.cpp
	$(CC) $(CFLAGS) -c ../../library/src/utility/host_tensor.cpp