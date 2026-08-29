# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(SUPPORTED_GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_cluster_launch test_cluster_launch.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_cluster_launch PRIVATE utility)
    endif()
endif()
