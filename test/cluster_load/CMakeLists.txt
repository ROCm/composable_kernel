# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(SUPPORTED_GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_cluster_load test_cluster_load.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_cluster_load PRIVATE utility)
    endif()

    add_gtest_executable(test_cluster_load_async test_cluster_load_async.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_cluster_load_async PRIVATE utility)
    endif()
endif()
