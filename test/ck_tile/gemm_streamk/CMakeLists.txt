# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
set(EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS
    -mllvm
    -enable-noalias-to-md-conversion=0
)
set(EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS})

# Currently test_ck_tile_streamk_smoke is only built on gfx9
if(GPU_TARGETS MATCHES "gfx90a|gfx942|gfx950")

    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
 
    #TODO: support all arches
    #TODO: current c-shuffle only supports C layout as R
    add_gtest_executable(test_ck_tile_streamk_tile_partitioner test_streamk_tile_partitioner.cpp)
    set(STREAMK_EXTENDED_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_fp16_persistent_compv3.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_fp16_persistent_mem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_bf16_persistent_compv3.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_bf16_persistent_mem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_fp16_nonpersistent_compv3.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_fp16_nonpersistent_mem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_bf16_nonpersistent_compv3.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_bf16_nonpersistent_mem.cpp
        test_gemm_streamk_util.cpp)

    # We only test fp8 and bf8 on gfx942 and gfx950 since these types are not natively supported on gfx90a
    if(GPU_TARGETS MATCHES "gfx942|gfx950")
        list(APPEND STREAMK_EXTENDED_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_fp8_persistent_compv3.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_fp8_persistent_mem.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_bf8_persistent_compv3.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_bf8_persistent_mem.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_fp8_nonpersistent_compv3.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_fp8_nonpersistent_mem.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_bf8_nonpersistent_compv3.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_bf8_nonpersistent_mem.cpp)
    endif()

    add_gtest_executable(test_ck_tile_streamk_extended ${STREAMK_EXTENDED_SOURCES})
    target_compile_options(test_ck_tile_streamk_extended PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

    # Collect all test targets for umbrella label
    set(CK_TILE_GEMM_STREAMK_TEST_TARGETS
        test_ck_tile_streamk_tile_partitioner
        test_ck_tile_streamk_extended
    )

    # Label all ck_tile gemm_streamk tests with CK_TILE_GEMM_STREAMK_TESTS for selective execution
    foreach(test_target ${CK_TILE_GEMM_STREAMK_TEST_TARGETS})
        set_tests_properties(${test_target} PROPERTIES LABELS "CK_TILE_GEMM_STREAMK_TESTS")
    endforeach()

    # Umbrella target to build and run all ck_tile gemm_streamk tests
    # Usage: ninja ck_tile_gemm_streamk_tests
    add_custom_target(ck_tile_gemm_streamk_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_GEMM_STREAMK_TESTS"
        DEPENDS ${CK_TILE_GEMM_STREAMK_TEST_TARGETS}
        USES_TERMINAL
        COMMENT "Running all ck_tile gemm_streamk tests..."
    )
else()
    message(DEBUG "Skipping test_ck_tile_streamk unit tests for current target")
endif()
