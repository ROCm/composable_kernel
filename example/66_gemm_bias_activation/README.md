### Build
```
mkdir -p build
cd build

sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...

make -j  example_gemm_bias_add_xdl_fp16
```
### Run Examples

#### args:
```
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: time kernel (0=no, 1=yes)
arg4 to 7: M (256x), N(128x), K(32x), op_type(Add = 0, Gelu = 1, Relu = 2, Silu = 3, Sigmoid = 4)
```
#### command:
```
./build/bin/example_gemm_bias_add_xdl_fp16
./build/bin/example_gemm_bias_add_xdl_fp16 1 1 1 64 3072 768 0
./build/bin/example_gemm_bias_add_xdl_fp16 1 1 1 64 3072 768 1
./build/bin/example_gemm_bias_add_xdl_fp16 1 1 1 64 3072 768 2
./build/bin/example_gemm_bias_add_xdl_fp16 1 1 1 64 3072 768 3
./build/bin/example_gemm_bias_add_xdl_fp16 1 1 1 64 3072 768 4
```
