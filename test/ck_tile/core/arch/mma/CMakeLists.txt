# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Currently ck_tile_gemm is only built on gfx94/gfx95
set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx9|gfx120")
    add_gtest_executable(test_amdgcn_sparse_mma pipeline/test_amdgcn_sparse_mma.cpp)

    target_compile_options(test_amdgcn_sparse_mma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()
if(GPU_TARGETS MATCHES "gfx950")
    add_gtest_executable(test_amdgcn_scale_mma pipeline/test_amdgcn_scale_mma.cpp)
    target_compile_options(test_amdgcn_scale_mma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()
if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_amdgcn_mma test_amdgcn_mma.cpp)
    target_compile_options(test_amdgcn_mma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    add_gtest_executable(test_amdgcn_wavewise_mma pipeline/test_amdgcn_wavewise_mma.cpp)
    target_compile_options(test_amdgcn_wavewise_mma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping ck_tile_gemm tests for current target")
endif()

if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_amdgcn_mma_layout_gfx9 test_amdgcn_mma_layout_gfx9.cpp)
    target_compile_options(test_amdgcn_mma_layout_gfx9 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()

if(GPU_TARGETS MATCHES "gfx942|gfx950")
    add_gtest_executable(test_amdgcn_mma_layout_gfx942 test_amdgcn_mma_layout_gfx942.cpp)
    target_compile_options(test_amdgcn_mma_layout_gfx942 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()

if(GPU_TARGETS MATCHES "gfx950")
    add_gtest_executable(test_amdgcn_mma_layout_gfx950 test_amdgcn_mma_layout_gfx950.cpp)
    target_compile_options(test_amdgcn_mma_layout_gfx950 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()

if(GPU_TARGETS MATCHES "gfx11")
    add_gtest_executable(test_amdgcn_mma_layout_gfx11 test_amdgcn_mma_layout_gfx11.cpp)
    target_compile_options(test_amdgcn_mma_layout_gfx11 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()

if(GPU_TARGETS MATCHES "gfx120")
    add_gtest_executable(test_amdgcn_mma_layout_gfx12 test_amdgcn_mma_layout_gfx12.cpp)
    target_compile_options(test_amdgcn_mma_layout_gfx12 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()

add_gtest_executable(test_amdgcn_mma_pipeline pipeline/test_amdgcn_mma_pipeline.cpp)
target_compile_options(test_amdgcn_mma_pipeline PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

