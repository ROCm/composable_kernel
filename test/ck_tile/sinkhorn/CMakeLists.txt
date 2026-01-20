# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_sinkhorn test_sinkhorn.cpp)
    target_link_libraries(test_ck_tile_sinkhorn)
endif()

