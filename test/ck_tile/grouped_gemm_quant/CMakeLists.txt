# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx12")
    # Split into three separate test executables for faster parallel compilation
    add_gtest_executable(test_ck_tile_grouped_gemm_quant_rowcol test_grouped_gemm_quant_rowcol.cpp)
    target_compile_options(test_ck_tile_grouped_gemm_quant_rowcol PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_ck_tile_grouped_gemm_quant_tensor test_grouped_gemm_quant_tensor.cpp)
    target_compile_options(test_ck_tile_grouped_gemm_quant_tensor PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_ck_tile_grouped_gemm_quant_aquant test_grouped_gemm_quant_aquant.cpp)
    target_compile_options(test_ck_tile_grouped_gemm_quant_aquant PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_ck_tile_grouped_gemm_quant_bquant test_grouped_gemm_quant_bquant.cpp)
    target_compile_options(test_ck_tile_grouped_gemm_quant_bquant PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

    add_gtest_executable(test_ck_tile_grouped_gemm_quant_bquant_preshuffleb test_grouped_gemm_quant_bquant_preshuffleb.cpp)
    target_compile_options(test_ck_tile_grouped_gemm_quant_bquant_preshuffleb PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

    # Collect all test targets for umbrella label
    set(CK_TILE_GROUPED_GEMM_QUANT_TEST_TARGETS
        test_ck_tile_grouped_gemm_quant_rowcol
        test_ck_tile_grouped_gemm_quant_tensor
        test_ck_tile_grouped_gemm_quant_aquant
        test_ck_tile_grouped_gemm_quant_bquant
        test_ck_tile_grouped_gemm_quant_bquant_preshuffleb
    )

    # Label all ck_tile grouped_gemm_quant tests with CK_TILE_GROUPED_GEMM_QUANT_TESTS for selective execution
    foreach(test_target ${CK_TILE_GROUPED_GEMM_QUANT_TEST_TARGETS})
        set_tests_properties(${test_target} PROPERTIES LABELS "CK_TILE_GROUPED_GEMM_QUANT_TESTS")
    endforeach()

    # Umbrella target to build and run all ck_tile grouped_gemm_quant tests
    # Usage: ninja ck_tile_grouped_gemm_quant_tests
    add_custom_target(ck_tile_grouped_gemm_quant_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_GROUPED_GEMM_QUANT_TESTS"
        DEPENDS ${CK_TILE_GROUPED_GEMM_QUANT_TEST_TARGETS}
        USES_TERMINAL
        COMMENT "Running all ck_tile grouped_gemm_quant tests..."
    )
endif()

