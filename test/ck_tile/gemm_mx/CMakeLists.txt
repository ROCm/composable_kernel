# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TEST_MX_GEMM_COMPILE_OPTIONS -Wno-undefined-func-template)
if(CK_USE_OCP_FP8)
    list(APPEND TEST_MX_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx95")
    add_gtest_executable(test_ck_tile_mx_gemm_async_rcr test_mx_gemm_async_rcr.cpp)
    target_compile_options(test_ck_tile_mx_gemm_async_rcr PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS})
    add_gtest_executable(test_ck_tile_mx_gemm_async_rrr test_mx_gemm_async_rrr.cpp)
    target_compile_options(test_ck_tile_mx_gemm_async_rrr PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS})
    add_gtest_executable(test_ck_tile_mx_gemm_async_crr test_mx_gemm_async_crr.cpp)
    target_compile_options(test_ck_tile_mx_gemm_async_crr PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS})
    add_gtest_executable(test_ck_tile_mx_gemm_async_ccr test_mx_gemm_async_ccr.cpp)
    target_compile_options(test_ck_tile_mx_gemm_async_ccr PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS})
    add_gtest_executable(test_ck_tile_mx_gemm_async_rcr_large_cases test_mx_gemm_async_rcr_large_cases.cpp)
    target_compile_options(test_ck_tile_mx_gemm_async_rcr_large_cases PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping ck_tile MX GEMM tests for current target")
endif()

if(GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_ck_tile_mx_gemm_splitk_support test_mx_gemm_splitk_support.cpp)
    target_compile_options(test_ck_tile_mx_gemm_splitk_support PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS})
    add_gtest_executable(test_ck_tile_mx_gemm_e8m0_gfx1250 test_mx_gemm_e8m0_gfx1250.cpp)
    target_compile_options(test_ck_tile_mx_gemm_e8m0_gfx1250 PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS})
    add_gtest_executable(test_ck_tile_mx_gemm_pipeline_tdm_wmma test_mx_gemm_pipeline_comp_tdm_wmma.cpp)
    target_compile_options(test_ck_tile_mx_gemm_pipeline_tdm_wmma PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS})
    # The TDM rows above with shapes sized for the gfx1250 emulator smoke stage
    # (test_mx_gemm_pipeline_smoke_cases.inc).
    add_gtest_executable(test_ck_tile_mx_gemm_pipeline_tdm_wmma_smoke test_mx_gemm_pipeline_comp_tdm_wmma_smoke.cpp)
    # bf16 C on TDM V1/V2 for FP8 and FP4, sized for the gfx1250 emulator smoke stage.
    add_gtest_executable(test_ck_tile_mx_gemm_tdm_bf16c test_mx_gemm_tdm_bf16c.cpp)
    # Host-only: the TDM MX shape gates of MxGemmKernel::IsSupportedArgument.
    add_gtest_executable(test_ck_tile_mx_gemm_tdm_gates test_mx_gemm_tdm_gates.cpp)
    # TDM exists only on gfx125. add_gtest_executable drops a source whose name matches a type missing
    # from DTYPES ("_bf16" for the bf16c test), so the targets may not exist.
    foreach(test_target test_ck_tile_mx_gemm_pipeline_tdm_wmma_smoke test_ck_tile_mx_gemm_tdm_bf16c
                        test_ck_tile_mx_gemm_tdm_gates)
        if(TARGET ${test_target})
            target_compile_options(${test_target} PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS})
            ck_tile_restrict_archs(${test_target} "^gfx125")
        endif()
    endforeach()
endif()

if(GPU_TARGETS MATCHES "gfx95|gfx125")
    # Host-only: the MX test helpers (6-bit fillers, E8M0 scales, tolerance). They serve the MX GEMM
    # tests of both gfx95 and gfx125, so they are built for either.
    add_gtest_executable(test_ck_tile_mx_gemm_host_helpers test_mx_gemm_host_helpers.cpp)
    if(TARGET test_ck_tile_mx_gemm_host_helpers)
        target_compile_options(test_ck_tile_mx_gemm_host_helpers PRIVATE ${TEST_MX_GEMM_COMPILE_OPTIONS})
    endif()
endif()
