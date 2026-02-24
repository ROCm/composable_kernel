# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Test for persistent async input GEMM - currently targeting gfx95
set(PERSISTENT_ASYNC_INPUT_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND PERSISTENT_ASYNC_INPUT_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
list(APPEND PERSISTENT_ASYNC_INPUT_COMPILE_OPTIONS
    -mllvm
    -enable-noalias-to-md-conversion=0
)

if(GPU_TARGETS MATCHES "gfx95")
    add_gtest_executable(test_ck_tile_gemm_persistent_async_input test_gemm_persistent_async_input.cpp)
    target_compile_options(test_ck_tile_gemm_persistent_async_input PRIVATE ${PERSISTENT_ASYNC_INPUT_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping test_ck_tile_gemm_persistent_async_input for current target - requires gfx95")
endif()
