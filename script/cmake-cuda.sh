#!/bin/bash

MY_PROJECT_SOURCE=../../../
 
 export CUDA_ROOT=/usr/local/cuda
 export CPATH=$CPATH:$CUDA_ROOT/include
 export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_ROOT/lib64
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

cmake                                                                                       \
-D CMAKE_CXX_COMPILER=clang++                                                               \
-D CMAKE_BUILD_TYPE=Release                                                                 \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                           \
-D DEVICE_BACKEND=NVIDIA                                                                    \
-D CMAKE_CUDA_FLAGS="-ccbin clang++ -m64 -Xcompiler -fopenmp -lineinfo --source-in-ptx -keep -Xptxas -v -gencode=arch=compute_61,code=sm_61 -Xptxas -v -maxrregcount=128" \
${MY_PROJECT_SOURCE}


#-D CMAKE_CUDA_FLAGS="-ccbin clang++ -m64 -Xcompiler -fopenmp -lineinfo --source-in-ptx -keep -Xptxas -v -gencode=arch=compute_70,code=sm_70" \
#-D CMAKE_CUDA_FLAGS="-ccbin clang++ -m64 -Xcompiler -fopenmp -lineinfo --source-in-ptx -keep -Xptxas -v -gencode=arch=compute_70,code=sm_70 -Xptxas -v -maxrregcount=128" \
