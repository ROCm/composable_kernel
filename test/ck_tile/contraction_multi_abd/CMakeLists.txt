# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx942|gfx950|gfx12-generic")
    add_gtest_executable(test_ck_tile_contraction_multi_abd_cshuffle test_contraction_multi_abd_cshuffle.cpp)
    add_gtest_executable(test_ck_tile_contraction_multi_abd_default2d test_contraction_multi_abd_default2d.cpp)
    target_compile_definitions(test_ck_tile_contraction_multi_abd_cshuffle PRIVATE ${COMPILE_OPTIONS})
    target_compile_definitions(test_ck_tile_contraction_multi_abd_default2d PRIVATE ${COMPILE_OPTIONS})
endif()
