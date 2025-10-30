# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# ================================================================================
# CK Tile Compiler CI Test Suite
# =================================================================================

# This test suite contains a curated set of CK Tile tests specifically designed for compiler
# CI nodes. The goal is to provide comprehensive coverage while completing within 20-30 minutes

set(COMPILER_TEST_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND COMPILER_TEST_COMPILE_OPTIONS -DCK_USE_OCP_FP8)
endif()

# ================================================================================
# Basic GEMM Functionality Tests - Core functionality
# ================================================================================
if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_gemm_pipeline_basic_fp16_compiler ../gemm/test_gemm_pipeline_basic_fp16.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_basic_fp16_compiler PRIVATE ${COMPILER_TEST_COMPILE_OPTIONS})
    set_tests_properties(test_ck_tile_gemm_pipeline_basic_fp16_compiler PROPERTIES LABELS "COMPILE_CI")
endif()

# ===============================================================================

if(GPU_TARGETS MATCHES "gfx94|gfx95|")
    # Test basic tests for MI300 series
    add_gtest_executable(test_ck_tile_gemm_pipeline_basic_fp8_compiler ../gemm/test_gemm_pipeline_basic_fp8.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_basic_fp8_compiler PRIVATE ${COMPILER_TEST_COMPILE_OPTIONS})
    set_tests_properties(test_ck_tile_gemm_pipeline_basic_fp8_compiler PROPERTIES LABELS "COMPILE_CI")
endif()

# ================================================================================
# Memory Pipeline Tests - Single test for memory operations
# ================================================================================

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_gemm_pipeline_mem_compiler ../gemm/test_gemm_pipeline_mem.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_mem_compiler PRIVATE ${COMPILER_TEST_COMPILE_OPTIONS})
    set_tests_properties(test_ck_tile_gemm_pipeline_mem_compiler PROPERTIES LABELS "COMPILE_CI")
endif()

if(GPU_TARGETS MATCHES "gfx11|gfx12")
    # mem_wmma for Radeon GPU's
    add_gtest_executable(test_ck_tile_gemm_pipeline_mem_wmma_compiler ../gemm/test_gemm_pipeline_mem_wmma.cpp)
    target_compile_options(test_ck_tile_gemm_pipeline_mem_compiler PRIVATE ${COMPILER_TEST_COMPILE_OPTIONS})
    set_tests_properties(test_ck_tile_gemm_pipeline_mem_compiler PROPERTIES LABELS "COMPILE_CI")
endif()

#================================================================================
# Basic Batched GEMM Functionality tests
#================================================================================

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_batched_gemm_compiler ../batched_gemm/test_batched_gemm.cpp)
    target_compile_options(test_ck_tile_batched_gemm_compiler PRIVATE ${COMPILER_TEST_COMPILE_OPTIONS})
    set_tests_properties(test_ck_tile_batched_gemm_compiler PROPERTIES LABELS "COMPILE_CI")
endif()

#================================================================================
# Grouped GEMM - Multi stream execution
#================================================================================

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_grouped_gemm_compiler ../grouped_gemm/test_grouped_gemm.cpp)

    set_tests_properties(test_ck_tile_grouped_gemm_compiler PROPERTIES LABELS "COMPILE_CI")
endif()

#================================================================================
# Reduce Operations - Important for Normalization
#================================================================================

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_reduce2d_compiler ../reduce/test_reduce2d.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_ck_tile_reduce2d_compiler PRIVATE utility)
    endif()
    set_tests_properties(test_ck_tile_reduce2d_compiler PROPERTIES LABELS "COMPILER_CI")
endif()

#================================================================================
# Data Type Tests - Quantization & Precision
#================================================================================

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_pk_int4_compiler ../data_type/test_pk_int4.cpp)
    set_tests_properties(test_ck_tile_pk_int4_compiler PROPERTIES LABELS "COMPILE_CI")
endif()

#================================================================================
# Epilogue Tests - Post Gemm operations
#================================================================================

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_cshuffle_epilogue_compiler ../epilogue/test_cshuffle_epilogue.cpp)
    set_tests_properties(test_ck_tile_cshuffle_epilogue_compiler PROPERTIES LABELS "COMPILE_CI")
endif()

#================================================================================
# ElementWise Operations - Basic element operations 
#================================================================================

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_elementwise_1d_compiler ../elementwise/test_elementwise_1d.cpp)
    set_tests_properties(test_ck_tile_elementwise_1d_compiler PROPERTIES LABELS "COMPILE_CI")
endif()

# ===============================================================================
# Container Tests - Core Infrastructure
# ===============================================================================

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_tuple_apply_compiler ../container/test_tuple_apply.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_ck_tile_tuple_apply_compiler PRIVATE utility)
    endif()
    set_tests_properties(test_ck_tile_tuple_apply_compiler PROPERTIES LABELS "COMPILE_CI")
endif()

#================================================================================
# Atomic Operations - Synchronization primitives
#================================================================================

add_gtest_executable(test_atomic_compiler ../atomic_add_op/test_atomic.cpp)
set_tests_properties(test_atomic_compiler PROPERTIES LABELS "COMPILE_CI")

# ==============================================================================
# Batched Transpose - Memory Layout transformation
# ==============================================================================

if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_ck_tile_batched_transpose_compiler ../batched_transpose/test_batched_transpose.cpp)
    set_tests_properties(test_ck_tile_batched_transpose_compiler PROPERTIES LABELS "COMPILE_CI")
endif()

#================================================================================
# Image to Column - Conv Preprocessing
#================================================================================

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_tile_image_to_column_compiler ../image_to_column/test_tile_image_to_column.cpp)
    set_tests_properties(test_tile_image_to_column_compiler PROPERTIES LABELS "COMPILE_CI")
endif()

#================================================================================
# Pooling - Important for CNN Operations
#================================================================================

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_pooling_compiler ../pooling/test_pooling.cpp)
    set_tests_properties(test_ck_tile_pooling_compiler PROPERTIES LABELS "COMPILE_CI")
endif()
