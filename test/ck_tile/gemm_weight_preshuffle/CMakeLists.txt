# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Currently ck_tile_gemm is only built on gfx94/gfx95
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

# Helper function to add a generated test
function(add_wp_test TEST_CLASS_NAME KERNEL_TYPES TEST_SUFFIX)
    set(GENERATED_FILE "${CMAKE_CURRENT_BINARY_DIR}/test_gemm_pipeline_${TEST_SUFFIX}.cpp")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/test_gemm_pipeline_wp.cpp.in"
        "${GENERATED_FILE}"
        @ONLY
    )
    add_gtest_executable(test_ck_tile_gemm_pipeline_${TEST_SUFFIX} ${GENERATED_FILE})
    target_compile_options(test_ck_tile_gemm_pipeline_${TEST_SUFFIX} PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_include_directories(test_ck_tile_gemm_pipeline_${TEST_SUFFIX} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

# TestCkTileGemmPipelineWeightPreshuffle: gfx94/gfx95/gfx11/gfx12
if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx11|gfx12")
    if(NOT WIN32) #TODO: Figure out why this fails
        add_wp_test(
            TestCkTileGemmPipelineWeightPreshuffle
            KernelTypesWeightPreshuffle
            wp
        )
    endif()
endif()

# TestCkTileGemmPipelineWeightPreshuffleTDM: gfx125 only
if(GPU_TARGETS MATCHES "gfx125")
    add_wp_test(
        TestCkTileGemmPipelineWeightPreshuffleTDM
        KernelTypesWeightPreshuffleTDM
        wp_tdm
    )
endif()
