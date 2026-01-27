# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx90a|gfx940|gfx941|gfx942|gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_mhc test_mhc.cpp)
    # Temporarily disable unused parameter warnings for debugging
    target_compile_options(test_ck_tile_mhc PRIVATE -Wno-unused-parameter)
endif()
