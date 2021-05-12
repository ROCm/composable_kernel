#!/bin/bash
rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=../
MY_PROJECT_INSTALL=../install.dir

cmake                                                                                                                              \
-D CMAKE_INSTALL_PREFIX=${MY_PROJECT_INSTALL}                                                                                      \
-D CMAKE_BUILD_TYPE=Release                                                                                                        \
-D DEVICE_BACKEND="AMD"                                                                                                            \
-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx1030 -gline-tables-only -save-temps=$CWD -ftemplate-backtrace-limit=0"                  \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                                                          \
-D CMAKE_PREFIX_PATH="/opt/rocm"                                                                                                   \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                                                                  \
${MY_PROJECT_SOURCE}

#-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx1030 -gline-tables-only -save-temps=$CWD -ftemplate-backtrace-limit=0"                   \
#-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx1030 -gline-tables-only -save-temps=$CWD -ftemplate-backtrace-limit=0 -mllvm -print-before=amdgpu-codegenprepare -mllvm -print-module-scope" \
#-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx906 -gline-tables-only -save-temps=$CWD"                                                \
#-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx906 -mllvm --amdgpu-spill-vgpr-to-agpr=0"                                               \
#-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx906 -mllvm --amdgpu-spill-vgpr-to-agpr=0 -save-temps=$CWD"                              \
#-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx906 -mllvm --amdgpu-enable-global-sgpr-addr -mllvm --amdgpu-spill-vgpr-to-agpr=0"       \
#-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx906 -mllvm --amdgpu-enable-global-sgpr-addr -mllvm --amdgpu-spill-vgpr-to-agpr=0 -save-temps=$CWD"       \
#-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx906 -mllvm --amdgpu-enable-global-sgpr-addr -mllvm --amdgpu-spill-vgpr-to-agpr=0 -v -gline-tables-only -save-temps=$CWD"       \
