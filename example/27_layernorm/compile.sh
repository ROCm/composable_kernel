#!/bin/bash

suf="out"

hipcc -std=c++17 --amdgpu-target=gfx90a \
-I /workspace/composable_kernel/include \
-I /workspace/composable_kernel/library/include \
$1 \
-o $1$suf \
/workspace/composable_kernel/library/src/utility/*.cpp \
2>&1 | tee log.txt
