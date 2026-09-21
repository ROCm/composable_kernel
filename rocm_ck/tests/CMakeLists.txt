# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocm_ck tests
#
# Test tiers:
#   ROCM_CK_SMOKE        — Fast host-only tests (< 1s total). No GPU required.
#   ROCM_CK_COMPILE_FAIL — Compile-fail tests (~7s). No GPU required.
#   ROCM_CK_KERNEL       — GPU kernel tests. Require HIP and a GPU.
#
# Usage:
#   ninja smoke-rocm-ck        # build + run smoke tests
#   ninja build-smoke-rocm-ck  # build only (no run)
#   ninja check-rocm-ck        # run all rocm_ck tests
#
#   ctest -L ROCM_CK_SMOKE --output-on-failure

# Google Test (via CK's FetchContent wrapper)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/gtest.cmake)

# ---------------------------------------------------------------------------
# Helper function — reusable per-test setup
# ---------------------------------------------------------------------------
function(add_rocm_ck_test test_name)
    add_executable(${test_name} ${ARGN})
    target_link_libraries(${test_name} PRIVATE rocm_ck GTest::gtest_main GTest::gmock)
    target_compile_options(${test_name} PRIVATE
        -Wall -Werror
        -Wno-global-constructors           # GTest registration macros
        -Wno-undef                         # GTest internal headers
        -Wno-zero-as-null-pointer-constant # C++20 <=> comparisons to 0
    )
endfunction()

# ---------------------------------------------------------------------------
# Smoke tests (fast, host-only, no GPU)
# ---------------------------------------------------------------------------
set(ROCM_CK_UNIT_SOURCES
    unit/unit_arch_properties.cpp
    unit/unit_args.cpp
    unit/unit_datatype.cpp
    unit/unit_fixed_string.cpp
    unit/unit_fmha_bwd_common.cpp
    unit/unit_fmha_bwd_compat.cpp
    unit/unit_fmha_bwd_consteval.cpp
    unit/unit_fmha_bwd_convert_dq.cpp
    unit/unit_fmha_bwd_dqdkdv.cpp
    unit/unit_fmha_bwd_ograd_dot_o.cpp
    unit/unit_fmha_bwd_validate_args.cpp
    unit/unit_gemm_spec.cpp
    unit/unit_index_t.cpp
    unit/unit_layout.cpp
    unit/unit_physical_tensor.cpp
    unit/unit_resolve.cpp
    unit/unit_schema_compatibility.cpp
    unit/unit_signature.cpp
    unit/unit_validate.cpp
)
set_source_files_properties(${ROCM_CK_UNIT_SOURCES} PROPERTIES LANGUAGE CXX)

add_rocm_ck_test(rocm_ck_unit ${ROCM_CK_UNIT_SOURCES})
target_include_directories(rocm_ck_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# When built inside CK (hipcc/amdclang++ is the CXX compiler), the global
# link_libraries(hip::device) injects -x hip via INTERFACE_COMPILE_OPTIONS,
# forcing amdclang++ to run both host and device compilation passes on every
# TU.  The device pass defines __HIP_DEVICE_COMPILE__, which triggers the
# #error guards in host-only _api.hpp headers.  --offload-host-only suppresses
# the device pass for this pure-host test target.
if(TARGET hip::device)
    target_compile_options(rocm_ck_unit PRIVATE --offload-host-only)
endif()
add_test(NAME rocm_ck_unit COMMAND rocm_ck_unit)
set_tests_properties(rocm_ck_unit PROPERTIES LABELS "ROCM_CK_SMOKE")

# ---------------------------------------------------------------------------
# Convenience targets
# ---------------------------------------------------------------------------
add_custom_target(build-smoke-rocm-ck DEPENDS rocm_ck_unit)

add_custom_target(smoke-rocm-ck
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "ROCM_CK_SMOKE"
    DEPENDS build-smoke-rocm-ck
    USES_TERMINAL
    COMMENT "Running rocm_ck smoke tests...")

add_custom_target(check-rocm-ck
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "ROCM_CK"
    DEPENDS build-smoke-rocm-ck
    USES_TERMINAL
    COMMENT "Running all rocm_ck tests...")

# ---------------------------------------------------------------------------
# Compile-fail tests (expected failures)
# ---------------------------------------------------------------------------
add_subdirectory(compile_fail)
