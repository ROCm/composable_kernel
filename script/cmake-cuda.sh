#!/bin/bash

rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=/home/chao/code/modular_convolution
MY_PROJECT_INSTALL=../install.dir

cmake                                                                                       \
-D CMAKE_INSTALL_PREFIX=${MY_PROJECT_INSTALL}                                               \
-D CMAKE_CXX_COMPILER=clang++                                                               \
-D CMAKE_BUILD_TYPE=Release                                                                 \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                           \
-D DEVICE_BACKEND=NVIDIA                                                                    \
-D CUDA_COMMON_INCLUDE_DIR="/package/install/cuda/10.1/NVIDIA_CUDA-10.1_Samples/common/inc" \
-D CMAKE_CUDA_FLAGS="-ccbin clang++ -m64 -Xcompiler -fopenmp -lineinfo --source-in-ptx -keep -Xptxas -v -gencode=arch=compute_61,code=sm_61 -Xptxas -v -maxrregcount=128" \
${MY_PROJECT_SOURCE}

#-D BOOST_ROOT="/package/install/boost_1.67.0"                                               \

#-D CMAKE_CUDA_COMPILER="/package/install/cuda_10.0/bin/nvcc"                                \
#-D CMAKE_CUDA_FLAGS="-ccbin clang++ -m64 -Xcompiler -fopenmp -lineinfo --source-in-ptx -keep -Xptxas -v -gencode=arch=compute_61,code=sm_61" \
#-D CMAKE_CUDA_FLAGS="-ccbin clang++ -m64 -Xcompiler -fopenmp -lineinfo --source-in-ptx -keep -Xptxas -v -gencode=arch=compute_61,code=sm_61 -Xptxas -v -maxrregcount=128" \
