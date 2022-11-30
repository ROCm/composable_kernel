# Instructions for ```example_gemm_xdl```

## Run ```example_gemm_xdl```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
./bin/example_gemm_xdl 0 1 5
```

Result (MI100 @ 1087Mhz, 133.5TFlops peak FP16)
```
a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
b_k_n: dim 2, lengths {4096, 4096}, strides {1, 4096}
c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
arg.a_grid_desc_k0_m_k1_{512, 3840, 8}
arg.b_grid_desc_k0_n_k1_{512, 4096, 8}
arg.c_grid_desc_m_n_{ 3840, 4096}
launch_and_time_kernel: grid_dim {480, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 5 times...
Perf: 1.19685 ms, 107.657 TFlops, 78.8501 GB/s
```

# Instruction for static shape split-k tiny gemm

## Switch branch
```bash
git checkout -b static_ck_small_gemm
```

## Run and test static shape kernel for tiny gemm
``` bash
cmake \
-D CMAKE_BUILD_TYPE=Release \
-D BUILD_DEV=OFF \
-D CMAKE_CXX_FLAGS=" -O3 " \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
-D CMAKE_PREFIX_PATH=/opt/rocm \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON \
-D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
-D AMDGPU_TARGETS=gfx90a \
..
DRIVER=example_gemm_xdl_fp16_splitk
export HIP_VISIBLE_DEVICES=1
export ROC_USE_FGS_KERNARG=0

make -j ${DRIVER}
./bin/${DRIVER} 1 1 1 16 1152 5120 5120 1152 1152 8
sleep 5
./bin/${DRIVER} 1 1 1 16 5120 384 384 5120 5120 4
sleep 5
./bin/${DRIVER} 1 1 1 16 1280 5120 5120 1280 1280 8
sleep 5
./bin/${DRIVER} 1 1 1 16 5120 1280 1280 5120 5120 5
```
