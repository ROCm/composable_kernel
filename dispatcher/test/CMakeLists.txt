# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.16)

# Test executables
set(TEST_SOURCES
    test_kernel_key.cpp
    test_problem.cpp
    test_registry.cpp
)

foreach(test_source ${TEST_SOURCES})
    # Get test name from source file
    get_filename_component(test_name ${test_source} NAME_WE)
    
    # Create test executable
    add_executable(${test_name} ${test_source})
    
    # Link against dispatcher library
    target_link_libraries(${test_name} PRIVATE
        ck_tile_dispatcher
    )
    
    # Add to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Summary message
message(STATUS "Configured ${CMAKE_CURRENT_LIST_DIR} with ${CMAKE_CXX_COMPILER_ID} compiler")

