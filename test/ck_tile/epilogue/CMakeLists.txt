# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_ck_tile_cshuffle_epilogue_fp16 test_cshuffle_epilogue_fp16.cpp)
add_gtest_executable(test_ck_tile_cshuffle_epilogue_fp8 test_cshuffle_epilogue_fp8.cpp)
add_gtest_executable(test_ck_tile_cshuffle_epilogue_scale test_cshuffle_epilogue_scale.cpp)

if(CK_USE_OCP_FP8)
    target_compile_options(test_ck_tile_cshuffle_epilogue_fp8 PRIVATE -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx950")
    add_gtest_executable(test_ck_tile_cshuffle_epilogue_fp8_gfx950 test_cshuffle_epilogue_fp8_gfx950.cpp)
    if(CK_USE_OCP_FP8)
        target_compile_options(test_ck_tile_cshuffle_epilogue_fp8_gfx950 PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()
endif()
