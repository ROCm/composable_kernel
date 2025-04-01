

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
./bin/basic_flash_attention_fwd 1 0 1
```
