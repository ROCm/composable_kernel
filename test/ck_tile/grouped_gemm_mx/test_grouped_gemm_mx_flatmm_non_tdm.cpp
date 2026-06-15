// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// MXFlatmmPipelineAGmemBGmemCRegV1 grouped-GEMM tests on gfx950 and gfx1250.

#include "test_grouped_gemm_mx_flatmm_common.hpp"

// Compile-time arch dispatch via GetCurrentTargetId(). Only the selected arch's kernels are
// instantiated.

// clang-format off
using KernelTypes = std::conditional_t<
    GetCurrentTargetId() == ck_tile::core::arch::TargetId::GFX1250,
    ::testing::Types<
        //         ADataType, BDataType, CDataType, ArchTraits
        std::tuple<F8,        F8,        F16,       MXFlatmm_GFX1250_FP8FP8_Traits>,
        std::tuple<F4,        F4,        F16,       MXFlatmm_GFX1250_FP4FP4_Traits>,
        std::tuple<F6,        F6,        F16,       MXFlatmm_GFX1250_FP6FP6_Traits>,
        std::tuple<F8,        F4,        F16,       MXFlatmm_GFX1250_FP8FP4_Traits>,
        std::tuple<F4,        F8,        F16,       MXFlatmm_GFX1250_FP4FP8_Traits>
    >,
    ::testing::Types<
        std::tuple<F8,        F8,        F16,       MXFlatmm_GFX950_FP8FP8_Traits>,
        std::tuple<F4,        F4,        F16,       MXFlatmm_GFX950_FP4FP4_Traits>,
        std::tuple<F6,        F6,        F16,       MXFlatmm_GFX950_FP6FP6_Traits>,
        std::tuple<F8,        F4,        F16,       MXFlatmm_GFX950_FP8FP4_Traits>,
        std::tuple<F4,        F8,        F16,       MXFlatmm_GFX950_FP4FP8_Traits>
    >
>;
// clang-format on

TYPED_TEST_SUITE(TestGroupedGemmMXFlatmm, KernelTypes);

#include "test_grouped_gemm_mx_flatmm_ut_cases.inc"
