// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/**
 * @file
 * @brief MX FP8 GEMM on comp_tdm_v3 (gfx1250): fp8 x fp8 with 32-element E8M0 scales, with the
 *        TDM epilogue (k_batch 1, checked with the tolerance derived in
 *        test_mx_gemm_host_helpers.hpp) and with the CShuffle epilogue (k_batch 1 and 2, split-K
 *        tolerance).
 */

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileMxGemmTdmV3 : public TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmV3<T>>
{
    public:
    // The derived MX tolerance (one rounding of C plus the fp32 accumulation bound) for the TDM
    // epilogue, which runs k_batch 1 only. The CShuffle rows also run split-K, where every split
    // rounds its partial sum to C's type before the atomic add: they use calculate_rtol_atol,
    // whose split-K term covers those k_batch roundings.
    static constexpr bool kUseDerivedMxTolerance =
        std::tuple_element_t<14, T>::value != MxGemmPipelineType::CompTDMV3CShuffle;

    // Every row is chosen to run; an unsupported combination must not turn into a silent skip.
    static_assert(TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmTdmV3<T>>::check_data_type(),
                  "every comp_tdm_v3 MX row must have WMMA traits");
};

#define TEST_SUITE_NAME TestCkTileMxGemmTdmV3

TYPED_TEST_SUITE(TestCkTileMxGemmTdmV3, KernelTypesMxGemmTdmV3);

#include "test_mx_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
