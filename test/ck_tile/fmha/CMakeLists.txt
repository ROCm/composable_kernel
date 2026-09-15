# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Keep in sync with example/ck_tile/01_fmha/CMakeLists.txt
if(NOT SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx12")
    return()
endif()

set(FMHA_BWD_INSTANCES "tile_fmha_bwd_instances")
set(FMHA_FWD_INSTANCES "tile_fmha_fwd_instances")

set(TEST_NAME "test_ck_tile_fmha")

function(add_gtest_fwd test_group)
    set(V_TYPES "fp16" "bf16" "fp8bf16" "fp32" "mxfp8" "mxfp4")
    # fp8/fp8fp32 targets and the quantization-scale sweep only have instances on gfx125x.
    # Keep them off every other target so their test time does not grow.
    if(GPU_TARGETS MATCHES "gfx125")
        list(APPEND V_TYPES "fp8" "fp8fp32")
    endif()
    if(GPU_TARGETS MATCHES "gfx908|gfx90a" AND NOT CK_USE_FP8_ON_UNSUPPORTED_ARCH)
        # fp8 instances are built for all gfx9, do not test on archs without hardware support
        list(REMOVE_ITEM V_TYPES "fp8bf16")
    endif()
    set(CPP_TYPE_fp16 "FmhaFwdFp16")
    set(CPP_TYPE_bf16 "FmhaFwdBf16")
    set(CPP_TYPE_fp8 "FmhaFwdFp8")
    set(CPP_TYPE_fp8bf16 "FmhaFwdFp8Bf16")
    set(CPP_TYPE_fp8fp32 "FmhaFwdFp8Fp32")
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
        # increase timeout limit for fmha_fwd tests to prevent timeout on a slower gfx11 machine
        set_tests_properties(${name} PROPERTIES TIMEOUT 3600)
        target_compile_definitions(${name} PRIVATE DataTypeConfig=${CPP_TYPE_${type}})
        if(GPU_TARGETS MATCHES "gfx125")
            target_compile_definitions(${name} PRIVATE CK_TILE_TEST_FMHA_QSCALE_SWEEP)
        endif()
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
        # increase timeout limit for fmha_bwd tests to prevent occasional timeout on slower machines
        set_tests_properties(${name} PROPERTIES TIMEOUT 3600)
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

set(FMHA_QR_TDM_PADDING_TEST_TARGET)
if(GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_ck_tile_fmha_qr_tdm_lds_padding
        test_qr_tdm_lds_padding.cpp)
    target_link_libraries(test_ck_tile_fmha_qr_tdm_lds_padding PRIVATE utility)
    # the qr_tdm pipeline folds log2(e) into scale_s and has no exp() fallback
    target_compile_definitions(test_ck_tile_fmha_qr_tdm_lds_padding
        PRIVATE CK_TILE_FMHA_FWD_FAST_EXP2=1)
    set_tests_properties(test_ck_tile_fmha_qr_tdm_lds_padding PROPERTIES
        LABELS "${TEST_NAME};CK_TILE_FMHA_TESTS")
    set(FMHA_QR_TDM_PADDING_TEST_TARGET test_ck_tile_fmha_qr_tdm_lds_padding)
endif()

# Python integration test for the fmha_fwd codegen path.
#
# Drives the real `generate.py` with a JSON tune-config
# (CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE) that declares two tiles
# identical in every field except F_occupancy, and asserts that codegen
# emits two distinct .cpp blobs (one with `_o2` suffix, one without).
# This covers both:
#   1. CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE JSON loading & dynamic
#      factory subclassing.
#   2. Per-occupancy kernel instantiation (kOccupancy_ template parameter)
#      exposed via FmhaFwdTileSize.name symbol suffix.
# Pure Python, no GPU required.
set(FMHA_FWD_TUNE_CONFIG_PY_TEST test_ck_tile_fmha_fwd_tune_config)
add_test(
    NAME ${FMHA_FWD_TUNE_CONFIG_PY_TEST}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_fmha_fwd_tune_config.py -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(${FMHA_FWD_TUNE_CONFIG_PY_TEST}
    PROPERTIES LABELS "${TEST_NAME};CK_TILE_FMHA_TESTS")

# Umbrella target to build and run all ck_tile fmha tests
# Usage: ninja ck_tile_fmha_tests
add_custom_target(ck_tile_fmha_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_FMHA_TESTS"
    DEPENDS ${TEST_NAME} ${FMHA_QR_TDM_PADDING_TEST_TARGET}
    USES_TERMINAL
    COMMENT "Running all ck_tile fmha tests..."
)
