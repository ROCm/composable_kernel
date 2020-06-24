#!/bin/bash
rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=../../../
MY_PROJECT_INSTALL=../install.dir

cmake                                                                                       \
-D CMAKE_INSTALL_PREFIX=${MY_PROJECT_INSTALL}                                               \
-D CMAKE_BUILD_TYPE=Release                                                                 \
-D DEVICE_BACKEND="AMD"                                                                     \
-D CMAKE_CXX_FLAGS="--amdgpu-target=gfx906"                                                 \
-D CMAKE_CXX_COMPILER=/opt/rocm/hip/bin/hipcc                                               \
-D CMAKE_PREFIX_PATH="/opt/rocm"                                                            \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                           \
${MY_PROJECT_SOURCE}

#-D CMAKE_CXX_FLAGS="-gline-tables-only -v --amdgpu-target=gfx906"                           \
