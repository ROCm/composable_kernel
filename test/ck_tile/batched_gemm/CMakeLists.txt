# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12|amdgcnspirv")
    add_custom_target(test_ck_tile_batched_gemm)

    add_gtest_executable(test_ck_tile_batched_gemm_f16 test_batched_gemm_f16.cpp)
    add_gtest_executable(test_ck_tile_batched_gemm_bf16 test_batched_gemm_bf16.cpp)

    add_dependencies(test_ck_tile_batched_gemm
        test_ck_tile_batched_gemm_f16
        test_ck_tile_batched_gemm_bf16)

    # FP8 / BF8 batched GEMM relies on the fp8/bf8 MFMA or WMMA paths.
    # - gfx908 / gfx90a use the scalar f32 fallback in
    #   include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp.
    # - gfx942 / gfx950 use the native v_mfma_f32_32x32x16_{fp8,bf8}_{fp8,bf8} instructions.
    # - gfx12 uses the native v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8}_w32_gfx12 instructions
    #   (see include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_8bit_traits.hpp).
    # - gfx11 has no native fp8/bf8 WMMA instruction and no software fallback, so the
    #   warp-gemm silently returns CVecType{0} and the kernel produces all-zero output.
    # Skip the test there until an fp8/bf8 path is implemented for gfx11.
    #
    # On gfx950 / gfx12 the FP8/BF8 hardware uses the OCP encoding (bias 15 for
    # E5M2, bias 7 for E4M3). On gfx94x it uses FNUZ (bias 16 / 8). The device
    # code in include/ck_tile/core/numeric/float8.hpp picks the encoding from
    # CK_TILE_USE_OCP_FP8, which in config.hpp defaults to 1 only inside
    # __HIP_DEVICE_COMPILE__. For the host-side reference_gemm and tensor fill
    # to use the same encoding as the device kernel (otherwise A/B and the
    # reference output are produced in FNUZ while the device interprets the
    # same bytes as OCP, yielding a 4x error from the off-by-one bias on each
    # operand), propagate -DCK_TILE_USE_OCP_FP8 to host code when the global
    # CK_USE_OCP_FP8 toggle is on (set by the top-level CMakeLists.txt for
    # gfx950 / gfx12).
    if(GPU_TARGETS MATCHES "gfx9|gfx12")
        set(CK_TILE_BATCHED_GEMM_F8_COMPILE_OPTIONS)
        if(CK_USE_OCP_FP8)
            list(APPEND CK_TILE_BATCHED_GEMM_F8_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
        endif()
        add_gtest_executable(test_ck_tile_batched_gemm_f8 test_batched_gemm_f8.cpp)
        add_gtest_executable(test_ck_tile_batched_gemm_bf8 test_batched_gemm_bf8.cpp)
        target_compile_options(test_ck_tile_batched_gemm_f8
            PRIVATE ${CK_TILE_BATCHED_GEMM_F8_COMPILE_OPTIONS})
        target_compile_options(test_ck_tile_batched_gemm_bf8
            PRIVATE ${CK_TILE_BATCHED_GEMM_F8_COMPILE_OPTIONS})
        if(TARGET test_ck_tile_batched_gemm_f8)
            add_dependencies(test_ck_tile_batched_gemm test_ck_tile_batched_gemm_f8)
        endif()
        if(TARGET test_ck_tile_batched_gemm_bf8)
            add_dependencies(test_ck_tile_batched_gemm test_ck_tile_batched_gemm_bf8)
        endif()
    endif()

    # INT8 batched GEMM relies on the int32 MFMA / WMMA paths.
    # On gfx908 / gfx90a there is no native v_mfma_i32_32x32x16_i8 instruction and
    # the warp-gemm impl in include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp
    # falls back to v_mfma_f32_32x32x2f32, writing fp32 bit patterns into an int32_t
    # accumulator and ultimately into the int32_t C buffer (see how the existing
    # test_ck_tile_gemm_pipeline_compv3 INT8 cases are restricted to gfx94|gfx95).
    if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx11|gfx12")
        add_gtest_executable(test_ck_tile_batched_gemm_int8 test_batched_gemm_int8.cpp)
        if(TARGET test_ck_tile_batched_gemm_int8)
            add_dependencies(test_ck_tile_batched_gemm test_ck_tile_batched_gemm_int8)
        endif()
    endif()
endif()
