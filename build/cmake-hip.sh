#!/bin/bash

rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=/home/chao/code/modular_convolution
MY_PROJECT_INSTALL=../install.dir

cmake                                                                                       \
-D CMAKE_INSTALL_PREFIX=${MY_PROJECT_INSTALL}                                               \
-D CMAKE_BUILD_TYPE=Release                                                                 \
-D DEVICE_BACKEND="HIP"                                                                     \
-D HIP_HIPCC_FLAGS="${HIP_HIPCC_FLAGS} -gline-tables-only"                                  \
-D CMAKE_CXX_FLAGS="-gline-tables-only"                                                     \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                   \
-D CMAKE_PREFIX_PATH="/opt/rocm;/home/package/build/mlopen_dep"                             \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                           \
${MY_PROJECT_SOURCE}
