

# CK_TILE Toy Example

This repository demonstrates a toy example implemented using ck_tile

## Build Instructions

Follow these steps to build the examples:

```sh
cd composable_kernel
mkdir build
cd build

cmake -D CMAKE_PREFIX_PATH=/opt/rocm \
      -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -D CMAKE_BUILD_TYPE=Release \
      -D GPU_TARGETS="gfx942" \
      -Dkernel=N ..
```

### Compile Examples

#### **GEMM Softmax Example**
```sh
make -j128 tile_example_basic_gemm_softmax
```

## Running Examples

### **GEMM Softmax Example**
```sh
./bin/tile_example_basic_gemm_softmax 1 4096 256 7168
```

## Advanced part
#### **GEMM Example**
##### Follow these steps to build and run the different kernels: 
```sh

cd composable_kernel
mkdir build
cd build

# for naive kernel
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx942" -Dkernel=N .. && make -j128 tile_example_basic_gemm_softmax

# for kernel A
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx942" -Dkernel=A .. && make -j128 tile_example_basic_gemm_softmax

# for kernel B
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx942" -Dkernel=B .. && make -j128 tile_example_basic_gemm_softmax

...

# for kernel H
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx942" -Dkernel=H .. && make -j128 tile_example_basic_gemm_softmax