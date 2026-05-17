# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# CK Tile Dispatcher Tests (C++ and Python)
# =============================================================================

cmake_minimum_required(VERSION 3.16)

# Find Python
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# =============================================================================
# Python Tests
# =============================================================================

# Auto-correction and validation stress test
add_test(
    NAME dispatcher_test_autocorrect
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_autocorrect.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_autocorrect PROPERTIES
    LABELS "dispatcher;python;validation"
    TIMEOUT 120
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Verbose version of the test
add_test(
    NAME dispatcher_test_autocorrect_verbose
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_autocorrect.py -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_autocorrect_verbose PROPERTIES
    LABELS "dispatcher;python;validation;verbose"
    TIMEOUT 180
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Individual Python Test Categories
add_test(
    NAME dispatcher_test_gemm_validation
    COMMAND ${Python3_EXECUTABLE} -m unittest test_autocorrect.TestGemmValidation test_autocorrect.TestGemmExpansion -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_gemm_validation PROPERTIES
    LABELS "dispatcher;python;gemm;validation"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

add_test(
    NAME dispatcher_test_python_autocorrect
    COMMAND ${Python3_EXECUTABLE} -m unittest test_autocorrect.TestPythonAutoCorrect -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_python_autocorrect PROPERTIES
    LABELS "dispatcher;python;autocorrect"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

add_test(
    NAME dispatcher_test_stress
    COMMAND ${Python3_EXECUTABLE} -m unittest test_autocorrect.TestStressRandom -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_stress PROPERTIES
    LABELS "dispatcher;python;stress"
    TIMEOUT 120
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

add_test(
    NAME dispatcher_test_arch_support
    COMMAND ${Python3_EXECUTABLE} -m unittest test_autocorrect.TestArchitectureSupport -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_arch_support PROPERTIES
    LABELS "dispatcher;python;arch"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

add_test(
    NAME dispatcher_test_fmha_codegen
    COMMAND ${Python3_EXECUTABLE} -m unittest test_fmha_codegen -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_fmha_codegen PROPERTIES
    LABELS "dispatcher;python;fmha;codegen"
    TIMEOUT 120
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

add_test(
    NAME dispatcher_test_fmha_rules
    COMMAND ${Python3_EXECUTABLE} -m unittest test_fmha_rules -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_fmha_rules PROPERTIES
    LABELS "dispatcher;python;fmha;rules"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# FMHA parity test (requires GPU)
add_test(
    NAME dispatcher_test_fmha_parity
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_fmha_parity.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_test_fmha_parity PROPERTIES
    LABELS "dispatcher;python;fmha;parity;gpu"
    TIMEOUT 600
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Stress Test Script
add_test(
    NAME dispatcher_stress_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/stress_test_autocorrect.py
        --arch gfx942 --samples 30 --seed 42
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_stress_test PROPERTIES
    LABELS "dispatcher;python;stress;integration"
    TIMEOUT 180
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# =============================================================================
# Integration Tests (mimic examples)
# =============================================================================

# Full integration test suite
add_test(
    NAME dispatcher_integration_tests
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_examples_integration.py -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_integration_tests PROPERTIES
    LABELS "dispatcher;python;integration;examples"
    TIMEOUT 600
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Quick integration test (utilities only)
add_test(
    NAME dispatcher_integration_quick
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_examples_integration.py::TestUtilityImports -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_integration_quick PROPERTIES
    LABELS "dispatcher;python;integration;quick"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# GEMM examples integration
add_test(
    NAME dispatcher_integration_gemm
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_examples_integration.py::TestGemmPythonExamples -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_integration_gemm PROPERTIES
    LABELS "dispatcher;python;integration;gemm"
    TIMEOUT 300
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# =============================================================================
# C++ Tests (Google Test)
# =============================================================================

# Include Google Test setup
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
    test_fmha_problem.cpp
    test_fmha_dispatcher.cpp
    test_fmha_registry.cpp
    
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
    get_filename_component(test_name ${test_source} NAME_WE)
    
    add_executable(${test_name} ${test_source})
    
    target_link_libraries(${test_name} PRIVATE
        ck_tile_dispatcher
        dispatcher_test_utils
        GTest::gtest_main
    )
    
    target_compile_options(${test_name} PRIVATE
        -Wno-global-constructors
        -Wno-undef
    )
    
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS "dispatcher;cpp;unit")
endforeach()

# Standalone integration tests (with their own main())
set(STANDALONE_TESTS
    test_minimal.cpp
    test_grouped_conv_config.cpp
    test_grouped_conv_problem.cpp
    test_grouped_conv_kernel_decl.cpp
    test_grouped_conv_registry.cpp
    test_fmha_kernel_decl.cpp
)

foreach(test_source ${STANDALONE_TESTS})
    get_filename_component(test_name ${test_source} NAME_WE)
    
    add_executable(${test_name} ${test_source})
    
    target_link_libraries(${test_name} PRIVATE
        ck_tile_dispatcher
        dispatcher_test_utils
    )
    
    target_compile_options(${test_name} PRIVATE
        -Wno-global-constructors
        -Wno-undef
    )
    
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS "dispatcher;cpp;integration")
endforeach()

# =============================================================================
# Real Kernel Tests (requires generated kernels)
# =============================================================================

set(KERNEL_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../generated_kernels")
set(KERNEL_REGISTRATION_HEADER "${KERNEL_OUTPUT_DIR}/dispatcher_wrappers/register_all_kernels.hpp")
set(CODEGEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../codegen/unified_gemm_codegen.py")

option(BUILD_DISPATCHER_REAL_KERNEL_TESTS "Build tests with real GPU kernels" ON)

if(BUILD_DISPATCHER_REAL_KERNEL_TESTS AND EXISTS "${CODEGEN_SCRIPT}")
    message(STATUS "Setting up real kernel test generation")
    
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
    
    add_custom_target(generate_test_kernels DEPENDS ${KERNEL_REGISTRATION_HEADER})
    
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
            
            add_test(NAME ${test_name} COMMAND ${test_name})
            set_tests_properties(${test_name} PROPERTIES LABELS "dispatcher;cpp;gpu;kernel")
        endforeach()
    endif()
endif()

# =============================================================================
# Custom Targets
# =============================================================================

add_custom_target(run_dispatcher_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L dispatcher --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all dispatcher tests"
)

add_custom_target(test_dispatcher_python
    COMMAND ${CMAKE_CTEST_COMMAND} -L "dispatcher;python" --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running Python dispatcher tests"
)

add_custom_target(test_dispatcher_cpp
    COMMAND ${CMAKE_CTEST_COMMAND} -L "dispatcher;cpp" --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running C++ dispatcher tests"
)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "Dispatcher tests configured:")
message(STATUS "  Run all:        ctest -L dispatcher")
message(STATUS "  Run Python:     ctest -L 'dispatcher;python' or make test_dispatcher_python")
message(STATUS "  Run C++:        ctest -L 'dispatcher;cpp' or make test_dispatcher_cpp")
message(STATUS "  Run verbose:    ctest -R dispatcher_test_autocorrect_verbose")
