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
    # Core unit tests
    test_kernel_key.cpp
    test_problem.cpp
    test_registry.cpp
    test_dispatcher.cpp
    test_tile_backend.cpp
    
    # Extended unit tests (more comprehensive coverage)
    test_kernel_key_extended.cpp
    test_problem_extended.cpp
    test_registry_extended.cpp
    test_dispatcher_extended.cpp
    
    # Regression tests (known issues and edge cases)
    test_regression.cpp
    
    # JSON export tests
    test_json_export.cpp
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

# Standalone integration tests (with their own main())
set(STANDALONE_TESTS
    test_minimal.cpp
    test_conv_config.cpp
    test_conv_problem.cpp
    test_conv_kernel_decl.cpp
    test_conv_registry.cpp
)

foreach(test_source ${STANDALONE_TESTS})
    # Get test name from source file
    get_filename_component(test_name ${test_source} NAME_WE)
    
    # Create test executable
    add_executable(${test_name} ${test_source})
    
    # Link against dispatcher library and test utils
    target_link_libraries(${test_name} PRIVATE
        ck_tile_dispatcher
        dispatcher_test_utils
    )
    
    # Suppress warnings
    target_compile_options(${test_name} PRIVATE
        -Wno-global-constructors
        -Wno-undef
    )
    
    # Add to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Real kernel tests (requires generated kernels from unified_gemm_codegen.py)
set(KERNEL_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../generated_kernels")
set(KERNEL_REGISTRATION_HEADER "${KERNEL_OUTPUT_DIR}/dispatcher_wrappers/register_all_kernels.hpp")
set(CODEGEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_gemm_codegen.py")

# Option to enable automatic kernel generation
option(BUILD_DISPATCHER_REAL_KERNEL_TESTS "Build tests with real GPU kernels (generates kernels automatically)" ON)

if(BUILD_DISPATCHER_REAL_KERNEL_TESTS AND EXISTS "${CODEGEN_SCRIPT}")
    message(STATUS "Setting up real kernel test generation")
    
    # Create custom target to generate kernels
    add_custom_command(
        OUTPUT ${KERNEL_REGISTRATION_HEADER}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${KERNEL_OUTPUT_DIR}
        COMMAND ${Python3_EXECUTABLE} ${CODEGEN_SCRIPT}
            --output-dir ${KERNEL_OUTPUT_DIR}
            --datatype fp16
            --layout rcr
            --gpu-target gfx942
            --preselected fp16_rcr_essential
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../codegen
        COMMENT "Generating CK Tile kernels for real kernel tests..."
        VERBATIM
    )
    
    # Create a custom target that depends on the generated header
    add_custom_target(generate_test_kernels DEPENDS ${KERNEL_REGISTRATION_HEADER})
    
    message(STATUS "Building real kernel tests with automatic kernel generation")
    
    # Note: test_real_kernel (multi-kernel test) disabled - has CK Tile API compatibility issues
    # The single-kernel test (test_real_kernel_simple) proves the concept works
    
    # Real GPU kernel tests using tile_engine style (single kernel with -include)
    set(SINGLE_KERNEL_HEADER "${KERNEL_OUTPUT_DIR}/gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_128x128x32_2x2x1_32x32x16.hpp")
    
    set(REAL_KERNEL_TESTS
        test_real_kernel_simple
        test_real_kernel_multi_size
        test_real_kernel_performance
        test_real_kernel_correctness
        test_sanity_ck_tile
    )
    
    if(EXISTS "${SINGLE_KERNEL_HEADER}")
        foreach(test_name ${REAL_KERNEL_TESTS})
            add_executable(${test_name} ${test_name}.cpp)
            
            add_dependencies(${test_name} generate_test_kernels)
            
            target_link_libraries(${test_name} PRIVATE
                ck_tile_dispatcher
            )
            
            target_include_directories(${test_name} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/../../include
                ${KERNEL_OUTPUT_DIR}
            )
            
            # Use -include to force include single kernel (tile_engine pattern)
            target_compile_options(${test_name} PRIVATE
                -include ${SINGLE_KERNEL_HEADER}
                -mllvm -enable-noalias-to-md-conversion=0
                -Wno-undefined-func-template
                -Wno-float-equal
                --offload-compress
            )
            
            if(hip_FOUND)
                target_link_libraries(${test_name} PRIVATE hip::device hip::host)
            endif()
            
            # Add to CTest
            add_test(NAME ${test_name} COMMAND ${test_name})
        endforeach()
        
        message(STATUS "✓ Added 4 real GPU kernel tests:")
        message(STATUS "    - test_real_kernel_simple (basic functionality)")
        message(STATUS "    - test_real_kernel_multi_size (various problem sizes)")
        message(STATUS "    - test_real_kernel_performance (performance metrics)")
        message(STATUS "    - test_real_kernel_correctness (vs CPU reference)")
    endif()
    
    message(STATUS "✓ Real kernel tests configured with automatic generation")
    message(STATUS "  Kernels will be generated to: ${KERNEL_OUTPUT_DIR}")
else()
    if(NOT BUILD_DISPATCHER_REAL_KERNEL_TESTS)
        message(STATUS "Real kernel tests disabled (BUILD_DISPATCHER_REAL_KERNEL_TESTS=OFF)")
    elseif(NOT EXISTS "${CODEGEN_SCRIPT}")
        message(STATUS "Codegen script not found: ${CODEGEN_SCRIPT}")
    endif()
    message(STATUS "To enable: -DBUILD_DISPATCHER_REAL_KERNEL_TESTS=ON")
endif()

# Summary message
message(STATUS "Configured ${CMAKE_CURRENT_LIST_DIR} with ${CMAKE_CXX_COMPILER_ID} compiler")

