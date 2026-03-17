# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Currently ck_tile_gemm is only built on gfx94/gfx95
set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx9|gfx12")
    add_gtest_executable(test_amdgcn_sparse_mma test_amdgcn_sparse_mma.cpp)
    target_compile_options(test_amdgcn_sparse_mma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()
if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_amdgcn_mma test_amdgcn_mma.cpp)
    target_compile_options(test_amdgcn_mma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping ck_tile_gemm tests for current target")
endif()

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_amdgcn_mma_layout test_amdgcn_mma_layout.cpp)
    target_compile_options(test_amdgcn_mma_layout PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping gfx9|gfx11|gfx12 mma layout validation tests for current target")
endif()

