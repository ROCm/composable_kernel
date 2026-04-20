// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "pipeline_tests_helper.hpp"

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/scale/scale_mma_pipeline.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/utility/functional.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;

using CompilerTargetGfx950 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX950>());

template <typename AType,
          typename BType,
          typename CType,
          std::uint32_t WaveTileM,
          std::uint32_t WaveTileN,
          std::uint32_t WaveTileK>
void ScaleMfmaGfx950Specialization_impl()
{
    using TestScaleMma = amdgcn_mma<AType,
                                    BType,
                                    CType,
                                    WaveTileM,
                                    WaveTileN,
                                    WaveTileK,
                                    DefaultScaleMfmaCtrlFlags,
                                    CompilerTargetGfx950,
                                    MmaOpFamily::SCALE>;

    static_assert(std::is_same_v<typename TestScaleMma::OpType, MfmaOp> &&
                      TestScaleMma::OpFamily == MmaOpFamily::SCALE,
                  "GFX950 scale intrinsic should have ScaleMFMAOp type");

    static_assert(is_mma_op_of_family_v<MmaOpFamily::SCALE, TestScaleMma>,
                  "GFX950 scale intrinsic should be detected as Scale");

    // Get its traits
    using TestTraits = MmaOpTraits<TestScaleMma>;

    // Verify trait detection
    static_assert(TestTraits::IsScale, "Scale MMA should be detected as scale");
    static_assert(TestTraits::IsSupported, "Scale MMA specialization should be supported");
    static_assert(TestTraits::IsMfma, "Scale MFMA should be detected as MFMA");
    static_assert(!TestTraits::IsWmma, "Scale MFMA should not be detected as WMMA");
}

TEST(ScaleMMATrait, ScaleMfmaGfx950Specialization)
{
    // Test fp8 → fp32 scale MFMA for GFX950 (16x16x128)
    ScaleMfmaGfx950Specialization_impl<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u>();
    // Test bf8 → fp32 scale MFMA for GFX950 (16x16x128)
    ScaleMfmaGfx950Specialization_impl<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u>();
    // Test fp4 → fp32 scale MFMA for GFX950 (16x16x128)
    ScaleMfmaGfx950Specialization_impl<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u>();
    // Test fp8 → fp32 scale MFMA for GFX950 (32x32x64)
    ScaleMfmaGfx950Specialization_impl<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u>();
    // Test bf8 → fp32 scale MFMA for GFX950 (32x32x64)
    ScaleMfmaGfx950Specialization_impl<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u>();
    // Test fp4 → fp32 scale MFMA for GFX950 (32x32x64)
    ScaleMfmaGfx950Specialization_impl<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 32u, 64u>();

    std::cout << "GFX950 scale MFMA specialization is correct" << std::endl;
}

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
template <typename AType,
          typename BType,
          typename CType,
          std::uint32_t WaveTileM,
          std::uint32_t WaveTileN,
          std::uint32_t WaveTileK>
void TestConceptRequirements_impl()
{
    using TestScaleMma = amdgcn_mma<AType,
                                    BType,
                                    CType,
                                    WaveTileM,
                                    WaveTileN,
                                    WaveTileK,
                                    DefaultScaleMfmaCtrlFlags,
                                    CompilerTargetGfx950,
                                    MmaOpFamily::SCALE>;
    static_assert(MmaOpI<TestScaleMma>);
}
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

TEST(ScaleMMATrait, TestConceptRequirements)
{
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
    TestConceptRequirements_impl<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u>();
    TestConceptRequirements_impl<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u>();
    TestConceptRequirements_impl<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u>();
    TestConceptRequirements_impl<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u>();
    TestConceptRequirements_impl<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u>();
    TestConceptRequirements_impl<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 32u, 64u>();
#else
    GTEST_SKIP() << "Not compiled with concepts. Skipping test.";
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
}

template <typename AType, typename BType, typename CType>
void ScaleSelector_impl()
{
    static_for<2, 14, 6>{}([](auto k_factor) {
        static_for<1, 33, 1>{}([&](auto i) {
            using Selected                = typename MmaDefaultSelector<AType,
                                                                        BType,
                                                                        CType,
                                                                        static_cast<std::uint32_t>(i),
                                                                        static_cast<std::uint32_t>(i),
                                                                        static_cast<std::uint32_t>(k_factor * i),
                                                                        CompilerTargetGfx950,
                                                                        MmaOpFamily::SCALE>::SelectedOp;
            static constexpr bool isValid = (i == 16 && k_factor == 8) || (i == 32);
            if constexpr(isValid)
            {
                // Selector should pick a scale MFMA implementation
                static_assert(MmaOpTraits<Selected>::IsScale);
                static_assert(MmaOpTraits<Selected>::IsMfma);
                static_assert(MmaOpTraits<Selected>::IsSupported);
                static_assert((std::is_same<typename Selected::OpType, MfmaOp>::value));
            }
            else
            {
                // Selector should pick the unsupported pass through
                static_assert(!MmaOpTraits<Selected>::IsSupported);
            }
        });
    });
}

TEST(ScaleMMATrait, ScaleSelector)
{
    ScaleSelector_impl<fp8_t, fp8_t, fp32_t>();
    ScaleSelector_impl<bf8_t, bf8_t, fp32_t>();
    ScaleSelector_impl<pk_fp4_t, pk_fp4_t, fp32_t>();
}

template <typename AType,
          typename BType,
          typename CType,
          typename ScaleAType,
          typename ScaleBType,
          std::uint32_t WaveTileM,
          std::uint32_t WaveTileN,
          std::uint32_t WaveTileK>
__global__ void
test_scale_accum_over_k(void* a, void* b, void* c, void* out, void* scale_A, void* scale_B)
{
    using Pipeline = ScaleMmaPipeline<AType, BType, CType, WaveTileM, WaveTileN, WaveTileK>;

    using AVecType = typename Pipeline::AVecType;
    using BVecType = typename Pipeline::BVecType;
    using CVecType = typename Pipeline::CVecType;

    // NOTE: WaveTileK is used as a Pipeline template parameter, but the K iteration is
    // happening outside the Pipeline. This is a bit incorrect currently.
    static constexpr std::uint32_t kIters = WaveTileK / Pipeline::MmaOp::kK;

    // Initialize the accumulator
    CVecType result = *reinterpret_cast<CVecType*>(c);

    // Accumulate input AxB over WaveTileK/FragK iterations
    for(std::uint32_t i = 0; i < kIters; ++i)
    {
        result = Pipeline::exec(*reinterpret_cast<AVecType*>(a),
                                *reinterpret_cast<BVecType*>(b),
                                result,
                                *reinterpret_cast<ScaleAType*>(scale_A),
                                *reinterpret_cast<ScaleBType*>(scale_B));
    }

    *reinterpret_cast<CVecType*>(out) = result;
}

template <typename AType,
          typename BType,
          typename CType,
          std::uint32_t WaveTileM,
          std::uint32_t WaveTileN,
          std::uint32_t WaveTileK>
void MmaSelector_Scale_Real_impl()
{
    using TestType = MmaPipelineTest<AType, BType, CType, WaveTileM, WaveTileN, WaveTileK>;
    TestType test;
    const auto should_skip = [](amdgcn_target_id currentArchId) {
        bool isSupportedWmma = false;
        bool isSupportedMfma = (currentArchId == amdgcn_target_id::GFX950);
        return ((currentArchId == amdgcn_target_id::HOST) || !(isSupportedWmma || isSupportedMfma));
    };
    const std::function<fp32_t(
        std::uint32_t, typename TestType::ScaleAType, typename TestType::ScaleBType)>
        validator =
            [](std::uint32_t fragK, TestType::ScaleAType scale_A, TestType::ScaleBType scale_B) {
                fp32_t actual_scale_A = std::powf(2.0f, scale_A - 127.0f);
                fp32_t actual_scale_B = std::powf(2.0f, scale_B - 127.0f);
                return static_cast<fp32_t>(fragK) * actual_scale_A * actual_scale_B;
            };
    const auto kernel = [](std::uint32_t waveSize,
                           void* a,
                           void* b,
                           void* c,
                           void* out,
                           void* scale_A,
                           void* scale_B) {
        test_scale_accum_over_k<typename TestType::AType,
                                typename TestType::BType,
                                typename TestType::CType,
                                typename TestType::ScaleAType,
                                typename TestType::ScaleBType,
                                TestType::WaveTileM,
                                TestType::WaveTileN,
                                TestType::WaveTileK>
            <<<1, waveSize>>>(a, b, c, out, scale_A, scale_B);
    };
    test.test_pipeline(should_skip, kernel, validator);
}

// Live test on real hardware for scale selection and execution.
TEST(ScaleMMATrait, MmaSelector_Scale_F8_F8_F32_16x16x128_Real)
{
    MmaSelector_Scale_Real_impl<fp8_t, fp8_t, fp32_t, 16u, 16u, 128u>();
}

// Live test on real hardware for scale selection and execution.
TEST(ScaleMMATrait, MmaSelector_Scale_BF8_BF8_F32_16x16x128_Real)
{
    MmaSelector_Scale_Real_impl<bf8_t, bf8_t, fp32_t, 16u, 16u, 128u>();
}

// Live test on real hardware for scale selection and execution.
TEST(ScaleMMATrait, MmaSelector_Scale_F4_F4_F32_16x16x128_Real)
{
    MmaSelector_Scale_Real_impl<pk_fp4_t, pk_fp4_t, fp32_t, 16u, 16u, 128u>();
}

// Live test on real hardware for scale selection and execution.
TEST(ScaleMMATrait, MmaSelector_Scale_F8_F8_F32_32x32x64_Real)
{
    MmaSelector_Scale_Real_impl<fp8_t, fp8_t, fp32_t, 32u, 32u, 64u>();
}

// Live test on real hardware for scale selection and execution.
TEST(ScaleMMATrait, MmaSelector_Scale_BF8_BF8_F32_32x32x64_Real)
{
    MmaSelector_Scale_Real_impl<bf8_t, bf8_t, fp32_t, 32u, 32u, 64u>();
}

// Live test on real hardware for scale selection and execution.
TEST(ScaleMMATrait, MmaSelector_Scale_F4_F4_F32_32x32x64_Real)
{
    MmaSelector_Scale_Real_impl<pk_fp4_t, pk_fp4_t, fp32_t, 32u, 32u, 64u>();
}
