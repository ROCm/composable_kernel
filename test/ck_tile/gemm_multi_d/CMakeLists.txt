# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx12")
    add_gtest_executable(test_gemm_multi_d_cshuffle test_gemm_multi_d_cshuffle.cpp)
    add_gtest_executable(test_gemm_multi_d_default2d test_gemm_multi_d_default2d.cpp)
    target_compile_definitions(test_gemm_multi_d_cshuffle PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_definitions(test_gemm_multi_d_default2d PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()

# Multi-D GEMM with the TDM pipelines and TdmMultiDEpilogue (gfx1250 only).
# The "_wmma" suffix drops gfx9 targets; the remaining gfx11/gfx12 targets are removed below
# because the TDM pipelines are gfx125-only (and the 128x128x64 configs exceed 64 KB LDS there).
if(GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_gemm_multi_d_tdm test_gemm_multi_d_tdm_wmma.cpp)
    if(TARGET test_gemm_multi_d_tdm)
        ck_tile_restrict_archs(test_gemm_multi_d_tdm "^gfx125")
        target_compile_definitions(test_gemm_multi_d_tdm PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    endif()
endif()
