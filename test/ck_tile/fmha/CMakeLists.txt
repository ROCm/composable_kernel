# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(FMHA_BWD_INSTANCES "tile_fmha_bwd_instances")
set(FMHA_FWD_INSTANCES "tile_fmha_fwd_instances")

set(TEST_NAME "test_ck_tile_fmha")

function(add_gtest_fwd test_group)
    set(V_TYPES "fp16" "bf16" "fp8bf16" "fp32" "mxfp8" "mxfp4")
    if(GPU_TARGETS MATCHES "gfx908|gfx90a" AND NOT CK_USE_FP8_ON_UNSUPPORTED_ARCH)
        # fp8 instances are built for all gfx9, do not test on archs without hardware support
        list(REMOVE_ITEM V_TYPES "fp8bf16")
    endif()
    set(CPP_TYPE_fp16 "FmhaFwdFp16")
    set(CPP_TYPE_bf16 "FmhaFwdBf16")
    set(CPP_TYPE_fp8bf16 "FmhaFwdFp8Bf16")
    set(CPP_TYPE_fp32 "FmhaFwdFp32")
    set(CPP_TYPE_mxfp8 "FmhaFwdMxFp8")
    set(CPP_TYPE_mxfp4 "FmhaFwdMxFp4")

    set(sources)
    if(TARGET ${FMHA_FWD_INSTANCES})
        get_target_property(sources ${FMHA_FWD_INSTANCES} SOURCES)
        message(VERBOSE "${FMHA_FWD_INSTANCES} SOURCES ${sources}")
    endif()

    set(all_tests)
    foreach(type ${V_TYPES})
        set(name "${test_group}_${type}")
        if(NOT sources MATCHES "_${type}_")
            message(STATUS "No FMHA FWD instances for ${type}, skip ${name}")
            continue()
        endif()
        add_gtest_executable(${name} test_fmha_fwd.cpp)
        get_test_property(${name} LABELS COMMON_LABELS)
        set_tests_properties(${name} PROPERTIES LABELS "${COMMON_LABELS};${TEST_NAME};${test_group};CK_TILE_FMHA_TESTS")
        target_compile_definitions(${name} PRIVATE DataTypeConfig=${CPP_TYPE_${type}})
        target_link_libraries(${name} PRIVATE ${FMHA_FWD_INSTANCES})
        list(APPEND all_tests ${name})
    endforeach()
    message(STATUS "FMHA FWD tests: ${all_tests}")
    add_custom_target(${test_group} DEPENDS ${all_tests})
endfunction()

function(add_gtest_bwd test_group)
    set(V_TYPES "fp16" "bf16" "fp32")
    set(CPP_TYPE_fp16 "FmhaBwdFp16")
    set(CPP_TYPE_bf16 "FmhaBwdBf16")
    set(CPP_TYPE_fp32 "FmhaBwdFp32")

    set(sources)
    if(TARGET ${FMHA_BWD_INSTANCES})
        get_target_property(sources ${FMHA_BWD_INSTANCES} SOURCES)
        message(VERBOSE "${FMHA_BWD_INSTANCES} SOURCES ${sources}")
    endif()

    set(all_tests)
    foreach(type ${V_TYPES})
        set(name "${test_group}_${type}")
        if(NOT sources MATCHES "_${type}_")
            message(STATUS "No FMHA BWD instances for ${type}, skip ${name}")
            continue()
        endif()
        add_gtest_executable(${name} test_fmha_bwd.cpp)
        get_test_property(${name} LABELS COMMON_LABELS)
        set_tests_properties(${name} PROPERTIES LABELS "${COMMON_LABELS};${TEST_NAME};${test_group};CK_TILE_FMHA_TESTS")
        target_compile_definitions(${name} PRIVATE DataTypeConfig=${CPP_TYPE_${type}})
        target_link_libraries(${name} PRIVATE ${FMHA_BWD_INSTANCES})
        list(APPEND all_tests ${name})
    endforeach()
    message(STATUS "FMHA BWD tests: ${all_tests}")
    add_custom_target(${test_group} DEPENDS ${all_tests})
endfunction()


add_gtest_fwd(${TEST_NAME}_fwd)
add_gtest_bwd(${TEST_NAME}_bwd)
add_custom_target(${TEST_NAME} DEPENDS ${TEST_NAME}_fwd ${TEST_NAME}_bwd)

# Umbrella target to build and run all ck_tile fmha tests
# Usage: ninja ck_tile_fmha_tests
add_custom_target(ck_tile_fmha_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_FMHA_TESTS"
    DEPENDS ${TEST_NAME}
    USES_TERMINAL
    COMMENT "Running all ck_tile fmha tests..."
)
