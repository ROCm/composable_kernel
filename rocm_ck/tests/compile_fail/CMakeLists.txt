# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Expected compilation failure tests for rocm_ck schema types.
#
# Each .cpp file must FAIL to compile. The CTest test PASSES when
# compilation FAILS with the expected error message.
#
# The loop auto-discovers .cpp files and sets WILL_FAIL TRUE as a safe
# default. The overrides below upgrade each test to verify the specific
# consteval throw message via PASS_REGULAR_EXPRESSION (which replaces
# WILL_FAIL since the two interact badly — WILL_FAIL inverts the regex
# match result).
#
# Usage:
#   ctest --test-dir build -R compile_fail
#   ctest --test-dir build -L ROCM_CK_COMPILE_FAIL

file(GLOB COMPILE_FAIL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(source ${COMPILE_FAIL_SOURCES})
    get_filename_component(test_name ${source} NAME_WE)
    set(target_name "rocm_ck_compile_fail_${test_name}")

    add_library(${target_name} OBJECT EXCLUDE_FROM_ALL ${source})
    target_link_libraries(${target_name} PRIVATE rocm_ck)
    target_compile_features(${target_name} PRIVATE cxx_std_20)
    target_compile_options(${target_name} PRIVATE
        -Wno-zero-as-null-pointer-constant # C++20 <=> comparisons to 0
    )

    add_test(
        NAME ${target_name}
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${target_name}
    )
    set_tests_properties(${target_name} PROPERTIES
        WILL_FAIL TRUE
        LABELS "ROCM_CK_COMPILE_FAIL"
    )
endforeach()

# ---- Message verification overrides ----
# PASS_REGULAR_EXPRESSION verifies the compiler emits the expected throw
# message. It ignores exit code, so the regex match alone proves compilation
# failed with the correct consteval error. WILL_FAIL must be FALSE here
# because CMake applies WILL_FAIL after PASS_REGULAR_EXPRESSION, inverting
# the result.

# resolve.hpp errors
set_tests_properties(rocm_ck_compile_fail_conflicting_layout PROPERTIES
    WILL_FAIL FALSE
    PASS_REGULAR_EXPRESSION "conflicting layout for tensor")
set_tests_properties(rocm_ck_compile_fail_quantize_empty_scale_name PROPERTIES
    WILL_FAIL FALSE
    PASS_REGULAR_EXPRESSION "quantize has empty scale_name")
set_tests_properties(rocm_ck_compile_fail_no_dtype PROPERTIES
    WILL_FAIL FALSE
    PASS_REGULAR_EXPRESSION "tensor dtype unresolvable")
set_tests_properties(rocm_ck_compile_fail_empty_tensor_name PROPERTIES
    WILL_FAIL FALSE
    PASS_REGULAR_EXPRESSION "operator slot has empty tensor name")
set_tests_properties(rocm_ck_compile_fail_ssa_violation PROPERTIES
    WILL_FAIL FALSE
    PASS_REGULAR_EXPRESSION "tensor name produced by multiple operators")
set_tests_properties(rocm_ck_compile_fail_scale_scalar_not_found PROPERTIES
    WILL_FAIL FALSE
    PASS_REGULAR_EXPRESSION "ScaleOp.scale references undeclared Scalar")

# arch_properties.hpp errors
set_tests_properties(rocm_ck_compile_fail_target_set_mixed_wave PROPERTIES
    WILL_FAIL FALSE
    PASS_REGULAR_EXPRESSION "wavefront_size.*requires all targets")
set_tests_properties(rocm_ck_compile_fail_fp8_ocp_unsupported PROPERTIES
    WILL_FAIL FALSE
    PASS_REGULAR_EXPRESSION "FP8_OCP.*not yet supported")
set_tests_properties(rocm_ck_compile_fail_empty_target_set_wavefront PROPERTIES
    WILL_FAIL FALSE
    PASS_REGULAR_EXPRESSION "wavefront_size.*called on empty")

# fixed_string.hpp errors
set_tests_properties(rocm_ck_compile_fail_tensor_name_too_long PROPERTIES
    WILL_FAIL FALSE
    PASS_REGULAR_EXPRESSION "FixedString.*exceeds capacity")

# layout.hpp errors
set_tests_properties(rocm_ck_compile_fail_layout_auto_stride PROPERTIES
    WILL_FAIL FALSE
    PASS_REGULAR_EXPRESSION "leadingDimStride requires Row or Col")
