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

macro(set_mma_test_arch_define target_name)
    if(GPU_TARGETS MATCHES "gfx950")
        target_compile_definitions(${target_name} PRIVATE CK_MMA_TEST_ARCH_GFX950)
    elseif(GPU_TARGETS MATCHES "gfx942")
        target_compile_definitions(${target_name} PRIVATE CK_MMA_TEST_ARCH_GFX942)
    elseif(GPU_TARGETS MATCHES "gfx90a")
        target_compile_definitions(${target_name} PRIVATE CK_MMA_TEST_ARCH_GFX90A)
    endif()
    # else: defaults to GFX908
endmacro()

if(GPU_TARGETS MATCHES "gfx9")
    add_gtest_executable(test_amdgcn_mma_layout_gfx9 test_amdgcn_mma_layout_gfx9.cpp)
    target_compile_options(test_amdgcn_mma_layout_gfx9 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    set_mma_test_arch_define(test_amdgcn_mma_layout_gfx9)
endif()

if(GPU_TARGETS MATCHES "gfx908|gfx90a")
    add_gtest_executable(test_amdgcn_mma_layout_gfx908_and_gfx90a test_amdgcn_mma_layout_gfx908_and_gfx90a.cpp)
    target_compile_options(test_amdgcn_mma_layout_gfx908_and_gfx90a PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    set_mma_test_arch_define(test_amdgcn_mma_layout_gfx908_and_gfx90a)
endif()

if(GPU_TARGETS MATCHES "gfx90a|gfx942|gfx950")
    add_gtest_executable(test_amdgcn_mma_layout_gfx90a_and_higher test_amdgcn_mma_layout_gfx90a_and_higher.cpp)
    target_compile_options(test_amdgcn_mma_layout_gfx90a_and_higher PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    set_mma_test_arch_define(test_amdgcn_mma_layout_gfx90a_and_higher)
endif()

if(GPU_TARGETS MATCHES "gfx942|gfx950")
    add_gtest_executable(test_amdgcn_mma_layout_gfx942_and_higher test_amdgcn_mma_layout_gfx942_and_higher.cpp)
    target_compile_options(test_amdgcn_mma_layout_gfx942_and_higher PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS} -Wno-header-hygiene)
    set_mma_test_arch_define(test_amdgcn_mma_layout_gfx942_and_higher)
endif()

if(GPU_TARGETS MATCHES "gfx950")
    add_gtest_executable(test_amdgcn_mma_layout_gfx950 test_amdgcn_mma_layout_gfx950.cpp)
    target_compile_options(test_amdgcn_mma_layout_gfx950 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    set_mma_test_arch_define(test_amdgcn_mma_layout_gfx950)
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

