# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# =============================================================================
# CK Tile Dispatcher Tests
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

# Set test properties
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

# =============================================================================
# Individual Test Categories (for selective testing)
# =============================================================================

# GEMM validation tests only
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

# Conv validation tests only
add_test(
    NAME dispatcher_test_conv_validation
    COMMAND ${Python3_EXECUTABLE} -m unittest test_autocorrect.TestConvValidation test_autocorrect.TestConvExpansion -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(dispatcher_test_conv_validation PROPERTIES
    LABELS "dispatcher;python;conv;validation"
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

# Python auto-correction tests
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

# Stress tests
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

# Architecture support tests
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

# =============================================================================
# Custom Target for Running All Dispatcher Tests
# =============================================================================

add_custom_target(test_dispatcher
    COMMAND ${CMAKE_CTEST_COMMAND} -L dispatcher --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all dispatcher tests"
)

# =============================================================================
# Stress Test (scripts/stress_test_autocorrect.py)
# =============================================================================

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

# Stress test with verbose output
add_test(
    NAME dispatcher_stress_test_verbose
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/stress_test_autocorrect.py
        --arch gfx942 --samples 50 --seed 42 --verbose
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_tests_properties(dispatcher_stress_test_verbose PROPERTIES
    LABELS "dispatcher;python;stress;integration;verbose"
    TIMEOUT 300
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../codegen:${CMAKE_CURRENT_SOURCE_DIR}/../scripts"
)

message(STATUS "Dispatcher tests configured")
message(STATUS "  Run all: ctest -L dispatcher")
message(STATUS "  Run verbose: ctest -R dispatcher_test_autocorrect_verbose")
message(STATUS "  Run GEMM only: ctest -R dispatcher_test_gemm")
message(STATUS "  Run Conv only: ctest -R dispatcher_test_conv")
message(STATUS "  Run stress: ctest -R dispatcher_stress_test")

