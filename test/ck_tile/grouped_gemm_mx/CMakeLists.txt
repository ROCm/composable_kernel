# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Currently ck_tile is only built on gfx1250
if(GPU_TARGETS MATCHES "gfx1250")
    add_gtest_executable(test_ck_tile_grouped_gemm_mx_tdm test_mx_grouped_gemm.cpp)
    # target_compile_options(test_ck_tile_grouped_gemm_mx_tdm PRIVATE --save-temps)
endif()
