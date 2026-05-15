# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TEST_MX_GEMM_COMPILE_OPTIONS -Wno-undefined-func-template)
if(CK_USE_OCP_FP8)
    list(APPEND TEST_MX_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx95")
    add_gtest_executable(test_ck_tile_mx_gemm_fp4 test_mx_gemm_fp4.cpp)
    target_compile_options(test_ck_tile_mx_gemm_fp4 PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_ck_tile_mx_gemm_fp8 test_mx_gemm_fp8.cpp)
    target_compile_options(test_ck_tile_mx_gemm_fp8 PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping ck_tile MX GEMM tests for current target")
endif()

if(GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_ck_tile_mx_gemm_pipeline_tdm_wmma test_mx_gemm_pipeline_comp_tdm_wmma.cpp)
    target_compile_options(test_ck_tile_mx_gemm_pipeline_tdm_wmma PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS})
endif()
