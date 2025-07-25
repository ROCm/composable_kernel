#!/bin/bash
rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=$1


if [ $# -ge 2 ]; then
    case "$2" in
        gfx*) 
            GPU_TARGETS=$2
            shift 2
            echo "GPU targets provided: $GPU_TARGETS"
            REST_ARGS=$@
            ;;
        *)
            echo "No GPU targets provided, using default targets: gfx908;gfx90a;gfx942"
            GPU_TARGETS="gfx908;gfx90a;gfx942"
            shift 1
            REST_ARGS=$@
            ;;
    esac
else
    echo "No GPU targets provided, using default targets: gfx908;gfx90a;gfx942"
    GPU_TARGETS="gfx908;gfx90a;gfx942"
    shift 1
    REST_ARGS=$@
fi

cmake                                                                                             \
-D CMAKE_PREFIX_PATH=/opt/rocm/                                                                   \
-D CMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++                                                  \
-D CMAKE_CXX_FLAGS="-std=c++20 -O3 -ftemplate-backtrace-limit=0  -fPIE  -Wno-gnu-line-marker"     \
-D CMAKE_BUILD_TYPE=Release                                                                       \
-D BUILD_DEV=ON                                                                                   \
-D GPU_TARGETS=$GPU_TARGETS                                                                       \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                                 \
-D USE_BITINT_EXTENSION_INT4=OFF                                                                  \
$REST_ARGS                                                                                        \
${MY_PROJECT_SOURCE}
