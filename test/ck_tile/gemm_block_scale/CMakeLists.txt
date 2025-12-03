# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TEST_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
  list(APPEND TEST_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

list(APPEND TEST_GEMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx12")
    # Typed Test Suite for GEMM Quantization - split into multiple files to reduce compile time
    
    # AQuant tests
    add_gtest_executable(test_tile_gemm_quant_aquant 
        test_gemm_quant_aquant.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # BQuant tests (without PreshuffleB)
    add_gtest_executable(test_tile_gemm_quant_bquant 
        test_gemm_quant_bquant.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # BQuant tests (with PreshuffleB)
    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffle 
        test_gemm_quant_bquant_preshuffle.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffle PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # RowColQuant tests
    add_gtest_executable(test_tile_gemm_quant_rowcol 
        test_gemm_quant_rowcol.cpp
    )
    target_compile_options(test_tile_gemm_quant_rowcol PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # TensorQuant tests
    add_gtest_executable(test_tile_gemm_quant_tensor 
        test_gemm_quant_tensor.cpp
    )
    target_compile_options(test_tile_gemm_quant_tensor PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping ck_tile quant gemm tests for current target")
endif()
