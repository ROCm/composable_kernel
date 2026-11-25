# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Currently ck_tile is only built on gfx9
set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_gemm_multi_abd_cshuffle test_gemm_multi_abd_cshuffle.cpp)
    add_gtest_executable(test_ck_tile_gemm_multi_abd_default2d test_gemm_multi_abd_default2d.cpp)
    target_compile_definitions(test_ck_tile_gemm_multi_abd_cshuffle PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_definitions(test_ck_tile_gemm_multi_abd_default2d PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()

# The producer/consumer TDM pipeline (named barriers) is gfx1250-only; it takes one A and one B.
if(GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_ck_tile_gemm_multi_abd_pc_wmma test_gemm_multi_abd_pc_gfx1250_wmma.cpp)
    if(TARGET test_ck_tile_gemm_multi_abd_pc_wmma)
        ck_tile_restrict_archs(test_ck_tile_gemm_multi_abd_pc_wmma "^gfx125")
    endif()
endif()
