#!/bin/bash
rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles
AVX2_FLAGS='-m64 -mavx2 -mf16c -mfma -DHALF_ENABLE_F16C_INTRINSICS=0'

rm -rf build/
mkdir build && cd build

MY_PROJECT_SOURCE=..
MY_PROJECT_INSTALL=../install.dir
rm -rf $MY_PROJECT_INSTALL
mkdir $MY_PROJECT_INSTALL

cmake                                                                   \
-D CMAKE_INSTALL_PREFIX=${MY_PROJECT_INSTALL}                           \
-D BUILD_DEV=OFF                                                        \
-D CMAKE_BUILD_TYPE=Release                                             \
-D CMAKE_CXX_FLAGS=" $AVX2_FLAGS "                                      \
-D CMAKE_CXX_COMPILER=clang++                                           \
-D CMAKE_PREFIX_PATH=/usr/local                                         \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                       \
-D CK_NOGPU=ON                                                          \
${MY_PROJECT_SOURCE}

