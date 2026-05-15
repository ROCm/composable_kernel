# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")

    add_custom_target(test_ck_tile_grouped_gemm)
    
    add_gtest_executable(test_ck_tile_grouped_gemm_f16 test_grouped_gemm_f16.cpp)
    add_gtest_executable(test_ck_tile_grouped_gemm_bf16 test_grouped_gemm_bf16.cpp)
    if(GPU_TARGETS MATCHES "gfx1250")
        add_gtest_executable(test_ck_tile_grouped_gemm_f8 test_grouped_gemm_f8.cpp)
    endif()
    
    add_dependencies(test_ck_tile_grouped_gemm
        test_ck_tile_grouped_gemm_f16
        test_ck_tile_grouped_gemm_bf16)
    if(GPU_TARGETS MATCHES "gfx1250")
        add_dependencies(test_ck_tile_grouped_gemm
            test_ck_tile_grouped_gemm_f8)
    endif()
endif()
