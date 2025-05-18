

# CK_TILE Tutorial Example

This folder demonstrates the examples implemented using ck_tile

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

#### **Basic Add Example**
```sh
make -j add_basic
```

#### **Elementwise Add Example**
```sh
make -j add
```

#### **GEMM Example**
```sh
make -j basic_gemm
```

#### **Flash Attention Forward Example**
```sh
make -j basic_flash_attention_fwd
```

## Running Examples

#### **Basic Add Example**
```sh
./bin/add_basic
```

### **Elementwise Add**
```sh
./bin/add
```

### **GEMM Example**
```sh
./bin/basic_gemm 1
```

### **Flash Attention Forward Example**
```sh
./bin/basic_flash_attention_fwd 1 1
```

## Advanced part
#### **GEMM Example**
##### Follow these steps to build and run the different kernels: 
```sh

cd composable_kernel
mkdir build
cd build

# for naive kernel
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx942" -Dkernel=N .. && make -j basic_gemm

# for kernel A
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx942" -Dkernel=A .. && make -j basic_gemm

# for kernel B
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx942" -Dkernel=B .. && make -j basic_gemm

...

# for kernel H
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx942" -Dkernel=H .. && make -j basic_gemm

```

```sh
./bin/basic_gemm 1
```

#### **Flash Attention Forward Example**
##### Follow these steps to build the kernels

```sh
# for naive kernel
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx942" .. && make -j basic_flash_attention_fwd

# for optimized kernel
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx942" -DENABLE_TOY_FA_FWD_OPT=ON .. && make -j basic_flash_attention_fwd
```
```sh
./bin/basic_flash_attention_fwd 1 1
```


##### Follow these steps to build the codegen instances

```sh
mkdir build
cd build
../script/cmake-ck-release.sh .. gfx942
make -j codegen_basic_flash_attention_fwd
```

```sh
./bin/codegen_basic_flash_attention_fwd 1 1 64 16384 16384 128 128
```
