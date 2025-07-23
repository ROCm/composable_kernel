#!/bin/bash
rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=$1

if [ $# -ge 2 ] && [[ "$2" =~ ^gfx ]]; then
    GPU_TARGETS=$2
    shift 2
    echo "GPU targets provided: $GPU_TARGETS"
    REST_ARGS=$@
else
    echo "No GPU targets provided, using default targets: gfx908;gfx90a;gfx942"
    GPU_TARGETS="gfx908;gfx90a;gfx942"
    shift 1
    REST_ARGS=$@
fi

cmake                                                                                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
-D CMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++                                                  \
-D CMAKE_CXX_FLAGS="-O3"                                                                          \
-D CMAKE_BUILD_TYPE=Release                                                                       \
-D BUILD_DEV=OFF                                                                                  \
-D GPU_TARGETS=$GPU_TARGETS                                                                       \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                                 \
-D USE_BITINT_EXTENSION_INT4=OFF                                                                  \
$REST_ARGS                                                                                        \
${MY_PROJECT_SOURCE}

