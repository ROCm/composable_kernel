# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.16)

# Include Google Test setup
# Note: gtest.cmake is in ${PROJECT_SOURCE_DIR}/cmake, should be on CMAKE_MODULE_PATH
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/gtest.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/gtest.cmake)
else()
    include(gtest)
endif()

# Mock kernel instance for testing (shared across tests)
add_library(dispatcher_test_utils STATIC
    test_mock_kernel.cpp
)

target_include_directories(dispatcher_test_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(dispatcher_test_utils PRIVATE
    ck_tile_dispatcher
)

# Test executables using Google Test
set(TEST_SOURCES
    test_kernel_key.cpp
    test_problem.cpp
    test_registry.cpp
    test_dispatcher.cpp
    test_tile_backend.cpp
    test_integration_e2e.cpp
)

foreach(test_source ${TEST_SOURCES})
    # Get test name from source file
    get_filename_component(test_name ${test_source} NAME_WE)
    
    # Create test executable
    add_executable(${test_name} ${test_source})
    
    # Link against dispatcher library and test utils
    target_link_libraries(${test_name} PRIVATE
        ck_tile_dispatcher
        dispatcher_test_utils
        GTest::gtest_main
    )
    
    # Suppress gtest warnings
    target_compile_options(${test_name} PRIVATE
        -Wno-global-constructors
        -Wno-undef
    )
    
    # Add to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Summary message
message(STATUS "Configured ${CMAKE_CURRENT_LIST_DIR} with ${CMAKE_CXX_COMPILER_ID} compiler")

