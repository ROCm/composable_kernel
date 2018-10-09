#!/bin/bash

rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=/package/code/github/test_feature/SpMV
MY_PROJECT_INSTALL=../install.dir

cmake                                                                                      \
-D CMAKE_INSTALL_PREFIX=${MY_PROJECT_INSTALL}                                              \
-D CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -std=c++11"                                         \
-D CMAKE_BUILD_TYPE=Release                                                                \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                          \
-D BOOST_ROOT="/package/install/boost_1.66.0-mpich_3.2"                                    \
-D CMAKE_CUDA_COMPILER="/package/install/cuda_9.0/bin/nvcc"                                \
-D CUDA_COMMON_INCLUDE_DIR="/package/code/github/test_feature/cuda_9.0_common/inc"         \
-D CMAKE_CUDA_FLAGS="-ccbin g++ -m64 -Xcompiler -fopenmp -lineinfo --source-in-ptx -keep -Xptxas -v -arch=sm_35 -Xptxas -v -maxrregcount=40"            \
${MY_PROJECT_SOURCE}

#-D CMAKE_CUDA_FLAGS="-lineinfo --source-in-ptx -keep -Xptxas -v -arch=sm_35 -Xptxas -v -maxrregcount=32"            \
#-D CMAKE_CUDA_FLAGS="-G -lineinfo --source-in-ptx -keep -Xptxas -v -arch=sm_35"            \
