# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TEST_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
  list(APPEND TEST_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

list(APPEND TEST_GEMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx12")
    # Typed Test Suite for GEMM Quantization - split into multiple files to reduce compile time
    
    # AQuant tests - split into 10 files

    # AQuant Memory Pipeline tests
    add_gtest_executable(test_tile_gemm_quant_aquant_mem_prefill_interwave
        test_gemm_quant_aquant_mem_prefill_interwave.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant_mem_prefill_interwave PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_aquant_mem_decode_intrawave
        test_gemm_quant_aquant_mem_decode_intrawave.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant_mem_decode_intrawave PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_aquant_mem_decode_interwave
        test_gemm_quant_aquant_mem_decode_interwave.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant_mem_decode_interwave PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_aquant_base_rcr 
        test_gemm_quant_aquant_base_rcr.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant_base_rcr PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_aquant_base_rrr_crr 
        test_gemm_quant_aquant_base_rrr_crr.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant_base_rrr_crr PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_aquant_base_ccr 
        test_gemm_quant_aquant_base_ccr.cpp
    )

    target_compile_options(test_tile_gemm_quant_aquant_base_ccr PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_aquant_prefill 
        test_gemm_quant_aquant_prefill.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant_prefill PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_aquant_transpose_c 
        test_gemm_quant_aquant_transpose_c.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant_transpose_c PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_aquant_preshuffle 
        test_gemm_quant_aquant_preshuffle.cpp
    )
    target_compile_options(test_tile_gemm_quant_aquant_preshuffle PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # ABQuant tests split into 4 files
    add_gtest_executable(test_tile_gemm_quant_abquant_base
        test_gemm_quant_abquant_base.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_base PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_abquant_padding
        test_gemm_quant_abquant_padding.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_padding PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_abquant_preshuffle
        test_gemm_quant_abquant_preshuffle_2d.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_preshuffle PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_abquant_preshuffle_preshuffleQuant
        test_gemm_quant_abquant_preshuffle_preshuffleQuant.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_preshuffle_preshuffleQuant PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_abquant_eightwaves
        test_gemm_quant_abquant_eightwaves.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_eightwaves PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_abquant_eightwaves_padded_stride
        test_gemm_quant_abquant_eightwaves_padded_stride.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_eightwaves_padded_stride PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # ABQuant split-K tests
    add_gtest_executable(test_tile_gemm_quant_abquant_splitk_decode
        test_gemm_quant_abquant_splitk_decode.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_splitk_decode PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_abquant_splitk_prefill
        test_gemm_quant_abquant_splitk_prefill.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_splitk_prefill PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_abquant_a4w4_base
        test_gemm_quant_abquant_a4w4_base.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_a4w4_base PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_abquant_a4w4_padding
        test_gemm_quant_abquant_a4w4_padding.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_a4w4_padding PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_abquant_a4w4_preshuffle
        test_gemm_quant_abquant_a4w4_preshuffle.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_a4w4_preshuffle PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_abquant_preshuffleQuant
        test_gemm_quant_abquant_preshuffleQuant.cpp
    )
    target_compile_options(test_tile_gemm_quant_abquant_preshuffleQuant PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # BQuant tests (without PreshuffleB) - split into 6 files
    add_gtest_executable(test_tile_gemm_quant_bquant_1d_128 
        test_gemm_quant_bquant_1d_128.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_1d_128 PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_1d_64 
        test_gemm_quant_bquant_1d_64.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_1d_64 PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_2d_small_n 
        test_gemm_quant_bquant_2d_small_n.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_2d_small_n PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_2d_medium_n 
        test_gemm_quant_bquant_2d_medium_n.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_2d_medium_n PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_2d_large_n 
        test_gemm_quant_bquant_2d_large_n.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_2d_large_n PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_transpose 
        test_gemm_quant_bquant_transpose.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_transpose PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # BQuant split-K tests (no preshuffle)
    add_gtest_executable(test_tile_gemm_quant_bquant_splitk_decode 
        test_gemm_quant_bquant_splitk_decode.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_splitk_decode PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_splitk_prefill 
        test_gemm_quant_bquant_splitk_prefill.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_splitk_prefill PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # BQuant tests (with PreshuffleB) - split into 5 files
    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffle_decode_1d 
        test_gemm_quant_bquant_preshuffle_decode_1d.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffle_decode_1d PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffle_prefill_1d 
        test_gemm_quant_bquant_preshuffle_prefill_1d.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffle_prefill_1d PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffle_tiled_permute 
        test_gemm_quant_bquant_preshuffle_tiled_permute.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffle_tiled_permute PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffle_decode_2d 
        test_gemm_quant_bquant_preshuffle_decode_2d.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffle_decode_2d PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffle_prefill_2d 
        test_gemm_quant_bquant_preshuffle_prefill_2d.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffle_prefill_2d PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # BQuant tests (with PreshuffleQuant) - split into 4 files
    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffleQuant_decode_1d 
        test_gemm_quant_bquant_preshuffleQuant_decode_1d.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffleQuant_decode_1d PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffleQuant_prefill_1d 
        test_gemm_quant_bquant_preshuffleQuant_prefill_1d.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffleQuant_prefill_1d PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffleQuant_decode_2d 
        test_gemm_quant_bquant_preshuffleQuant_decode_2d.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffleQuant_decode_2d PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_preshuffleQuant_prefill_2d 
        test_gemm_quant_bquant_preshuffleQuant_prefill_2d.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_preshuffleQuant_prefill_2d PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    # BQuant microscale tests
    add_gtest_executable(test_tile_gemm_quant_bquant_microscale_rcr_1d_64 
        test_gemm_quant_bquant_microscale_rcr_1d_64.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_microscale_rcr_1d_64 PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_microscale_crr_1d_64 
        test_gemm_quant_bquant_microscale_crr_1d_64.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_microscale_crr_1d_64 PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_microscale_rrr_1d_64 
        test_gemm_quant_bquant_microscale_rrr_1d_64.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_microscale_rrr_1d_64 PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_microscale_ccr_1d_64 
        test_gemm_quant_bquant_microscale_ccr_1d_64.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_microscale_ccr_1d_64 PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_microscale_rcr_1d_128 
        test_gemm_quant_bquant_microscale_rcr_1d_128.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_microscale_rcr_1d_128 PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_microscale_crr_1d_128 
        test_gemm_quant_bquant_microscale_crr_1d_128.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_microscale_crr_1d_128 PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_microscale_rrr_1d_128 
        test_gemm_quant_bquant_microscale_rrr_1d_128.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_microscale_rrr_1d_128 PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_tile_gemm_quant_bquant_microscale_ccr_1d_128 
        test_gemm_quant_bquant_microscale_ccr_1d_128.cpp
    )
    target_compile_options(test_tile_gemm_quant_bquant_microscale_ccr_1d_128 PRIVATE ${TEST_GEMM_COMPILE_OPTIONS})

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

    # Collect all test targets for umbrella label
    set(CK_TILE_GEMM_BLOCK_SCALE_TEST_TARGETS
        # AQuant tests
        test_tile_gemm_quant_aquant_mem_prefill_interwave
        test_tile_gemm_quant_aquant_mem_decode_intrawave
        test_tile_gemm_quant_aquant_mem_decode_interwave
        test_tile_gemm_quant_aquant_base_rcr
        test_tile_gemm_quant_aquant_base_rrr_crr
        test_tile_gemm_quant_aquant_base_ccr
        test_tile_gemm_quant_aquant_prefill
        test_tile_gemm_quant_aquant_transpose_c
        test_tile_gemm_quant_aquant_preshuffle
        # ABQuant tests
        test_tile_gemm_quant_abquant_base
        test_tile_gemm_quant_abquant_padding
        test_tile_gemm_quant_abquant_preshuffle
        test_tile_gemm_quant_abquant_preshuffle_preshuffleQuant
        test_tile_gemm_quant_abquant_preshuffleQuant
        test_tile_gemm_quant_abquant_a4w4_base
        test_tile_gemm_quant_abquant_a4w4_padding
        test_tile_gemm_quant_abquant_a4w4_preshuffle
        test_tile_gemm_quant_abquant_eightwaves
        test_tile_gemm_quant_abquant_eightwaves_padded_stride
        # ABQuant split-K tests
        test_tile_gemm_quant_abquant_splitk_decode
        test_tile_gemm_quant_abquant_splitk_prefill
        # BQuant tests
        test_tile_gemm_quant_bquant_1d_128
        test_tile_gemm_quant_bquant_1d_64
        test_tile_gemm_quant_bquant_2d_small_n
        test_tile_gemm_quant_bquant_2d_medium_n
        test_tile_gemm_quant_bquant_2d_large_n
        test_tile_gemm_quant_bquant_transpose
        # BQuant split-K tests
        test_tile_gemm_quant_bquant_splitk_decode
        test_tile_gemm_quant_bquant_splitk_prefill
        # BQuant preshuffle tests
        test_tile_gemm_quant_bquant_preshuffle_decode_1d
        test_tile_gemm_quant_bquant_preshuffle_prefill_1d
        test_tile_gemm_quant_bquant_preshuffle_tiled_permute
        test_tile_gemm_quant_bquant_preshuffle_decode_2d
        test_tile_gemm_quant_bquant_preshuffle_prefill_2d
        # BQuant preshuffleQuant tests
        test_tile_gemm_quant_bquant_preshuffleQuant_decode_1d
        test_tile_gemm_quant_bquant_preshuffleQuant_prefill_1d
        test_tile_gemm_quant_bquant_preshuffleQuant_decode_2d
        test_tile_gemm_quant_bquant_preshuffleQuant_prefill_2d
        # Other quant tests
        test_tile_gemm_quant_rowcol
        test_tile_gemm_quant_tensor
    )

    # Label all ck_tile gemm_block_scale tests with CK_TILE_GEMM_BLOCK_SCALE_TESTS for selective execution
    foreach(test_target ${CK_TILE_GEMM_BLOCK_SCALE_TEST_TARGETS})
        set_tests_properties(${test_target} PROPERTIES LABELS "CK_TILE_GEMM_BLOCK_SCALE_TESTS")
    endforeach()

    # Target to build only AQuant memory pipeline tests
    add_custom_target(test_tile_gemm_aquant_mem_all)
    add_dependencies(test_tile_gemm_aquant_mem_all
        test_tile_gemm_quant_aquant_mem_prefill_interwave
        test_tile_gemm_quant_aquant_mem_decode_intrawave
        test_tile_gemm_quant_aquant_mem_decode_interwave
    )

    # Umbrella target to build all gemm quant tests
    add_custom_target(test_tile_gemm_quant_all)
    add_dependencies(test_tile_gemm_quant_all ${CK_TILE_GEMM_BLOCK_SCALE_TEST_TARGETS})

    # Umbrella target to build and run all ck_tile gemm_block_scale tests
    # Usage: ninja ck_tile_gemm_block_scale_tests
    add_custom_target(ck_tile_gemm_block_scale_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_GEMM_BLOCK_SCALE_TESTS"
        DEPENDS ${CK_TILE_GEMM_BLOCK_SCALE_TEST_TARGETS}
        USES_TERMINAL
        COMMENT "Running all ck_tile gemm_block_scale tests..."
    )
else()
    message(DEBUG "Skipping ck_tile quant gemm tests for current target")
endif()
