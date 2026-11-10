# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_load_tile_transpose_distribution
    test_load_tile_transpose_distribution.cpp)

add_gtest_executable(test_tdm_real_lengths test_tdm_real_lengths.cpp)

add_gtest_executable(test_element_space_clamp test_element_space_clamp.cpp)

# Negative compile checks: a reference into a temporary tensor view must be rejected.
# The object libraries are excluded from "all"; a ctest entry builds each one and passes
# only when the build output contains the -Wdangling diagnostic for the bound reference.
# -Werror=dangling makes the build fail without BUILD_DEV too, so it is never up to date.
# Entries are "<macro>|<name of the bound reference>".
set(TENSOR_VIEW_LIFETIME_COMPILE_FAIL_CASES
    "DESCRIPTOR|descriptor"
    "BUFFER_VIEW|buffer_view"
    "CONST_BUFFER_VIEW|const_buffer_view")

# The diagnostic comes from the frontend, so one target is enough.
list(GET SUPPORTED_GPU_TARGETS 0 TENSOR_VIEW_LIFETIME_ARCH)
get_property(TENSOR_VIEW_LIFETIME_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

set_source_files_properties(test_tensor_view_lifetime_compile_fail.cpp PROPERTIES LANGUAGE HIP)
foreach(fail_case IN LISTS TENSOR_VIEW_LIFETIME_COMPILE_FAIL_CASES)
    string(REPLACE "|" ";" fail_case_parts "${fail_case}")
    list(GET fail_case_parts 0 fail_macro)
    list(GET fail_case_parts 1 fail_reference)
    string(TOLOWER "${fail_macro}" fail_suffix)
    set(fail_target test_ck_tile_tensor_view_lifetime_compile_fail_${fail_suffix})
    add_library(${fail_target} OBJECT EXCLUDE_FROM_ALL test_tensor_view_lifetime_compile_fail.cpp)
    set_property(TARGET ${fail_target} PROPERTY HIP_ARCHITECTURES ${TENSOR_VIEW_LIFETIME_ARCH})
    target_compile_definitions(${fail_target} PRIVATE CK_TENSOR_VIEW_LIFETIME_FAIL_${fail_macro})
    target_compile_options(${fail_target} PRIVATE -Werror=dangling)
    if(TENSOR_VIEW_LIFETIME_MULTI_CONFIG)
        add_test(NAME ${fail_target}
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${fail_target}
                    --config $<CONFIG>)
    else()
        add_test(NAME ${fail_target}
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${fail_target})
    endif()
    # RESOURCE_LOCK: these tests drive the shared build tree and must not run concurrently.
    set_tests_properties(${fail_target} PROPERTIES
        PASS_REGULAR_EXPRESSION "local reference '${fail_reference}' will be destroyed"
        RESOURCE_LOCK ck_tile_build_tree
        LABELS "SMOKE_TEST")
endforeach()
