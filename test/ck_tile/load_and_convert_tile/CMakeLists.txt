# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(LOAD_TILE_TRANSPOSE_COMPILE_OPTIONS)
if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_load_and_convert_tile test_load_and_convert_tile.cpp)
    list(APPEND LOAD_TILE_TRANSPOSE_COMPILE_OPTIONS -fverbose-asm --save-temps -Wno-gnu-line-marker)
    target_compile_options(test_load_and_convert_tile PRIVATE ${LOAD_TILE_TRANSPOSE_COMPILE_OPTIONS})
endif()
