# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx95")
    add_gtest_executable(test_ck_tile_wg_16x16x128_fp4 test_f32_16x16x128_fp4.cpp)
endif()

# Scale16 warp gemm tests for V_WMMA_SCALE16_F32_16X16X128_F8F6F4.
# Each test covers 4 type combos (fp8xfp8, bf8xbf8, fp8xbf8, bf8xfp8) x 3 cases
# (uniform scale, random scale, TransposeC) = 12 tests.
# 16x16: single warp gemm call. 32x32: pipeline-style 2x2 block loop with per-block scales.
if(GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_ck_tile_wg_16x16x128_fp8_scale16 test_f32_16x16x128_fp8_scale16.cpp)
    add_gtest_executable(test_ck_tile_wg_32x32x128_fp8_scale16 test_f32_32x32x128_fp8_scale16.cpp)
    add_gtest_executable(test_ck_tile_wmma_bf16_16x16x32_gfx1250 test_wmma_bf16_16x16x32_gfx1250.cpp)
    add_gtest_executable(test_ck_tile_wmma_bf16f32_16x16x32 test_wmma_bf16f32_16x16x32.cpp)
    add_gtest_executable(test_ck_tile_wg_32x16x128_fp4 test_f32_32x16x128_fp4.cpp)
endif()
