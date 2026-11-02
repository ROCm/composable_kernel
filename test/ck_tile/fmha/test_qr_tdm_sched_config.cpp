// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "fmha_test_common.hpp"
#include "gtest/gtest.h"

#include <array>

namespace {

using namespace ck_tile;

template <typename T, int D, int N, int V>
struct Config
{
    using Problem  = typename test::Model<D, false, false, true, T, N, V>::Problem;
    using Geometry = FmhaTdmSchedGeometryFor<Problem>;
    using Policy   = FmhaTdmSchedPolicyFor<Problem>;
};

template <typename Policy>
struct WrongCountSchedule : Policy::Schedule
{
    static constexpr index_t kQkWmmasPerStage = Policy::Geometry::kQkWmmasPerStage - 1;
};
template <typename Policy>
struct WrongCountPolicy : Policy
{
    using Schedule = WrongCountSchedule<Policy>;
};

template <typename Problem, BlockAttentionBiasEnum Bias>
struct BiasProblem : Problem
{
    static constexpr auto BiasEnum = Bias;
};

// This fixture exercises only the max-policy predicate. Its mixed and FP8
// types do not imply that the corresponding full kernel problem is supported.
template <typename Q, typename K, typename V, bool Group, bool Mask, bool Lse>
struct CompilerMaxProblem
{
    using QDataType                    = Q;
    using KDataType                    = K;
    using VDataType                    = V;
    static constexpr bool kIsGroupMode = Group;
    static constexpr bool kStoreLSE    = Lse;
    struct FmhaMask
    {
        static constexpr bool IsMasking = Mask;
    };
};

template <typename Policy, typename Q, typename K, typename V>
void ExpectCompilerMaxFlagMatrix(const std::array<bool, 8>& expected, const char* type_case)
{
    SCOPED_TRACE(type_case);
    static_for<0, 8, 1>{}([&](auto state) {
        constexpr int s = decltype(state)::value;
        using Problem   = CompilerMaxProblem<Q, K, V, (s & 4) != 0, (s & 2) != 0, (s & 1) != 0>;
        EXPECT_EQ((Policy::template UseCompilerMax<Problem>()), expected[s])
            << "group=" << Problem::kIsGroupMode << " mask=" << Problem::FmhaMask::IsMasking
            << " lse=" << Problem::kStoreLSE;
    });
}

template <typename Policy, bool Early>
struct OutputRescalePolicy : Policy
{
    static constexpr bool kEarlyOutputRescale = Early;
};

using Configs = ::testing::Types<Config<bf16_t, 64, 64, 64>,
                                 Config<half_t, 64, 64, 64>,
                                 Config<bf16_t, 128, 128, 128>,
                                 Config<half_t, 128, 128, 128>,
                                 Config<bf16_t, 192, 64, 128>,
                                 Config<half_t, 192, 64, 128>,
                                 Config<bf16_t, 192, 128, 128>,
                                 Config<half_t, 192, 128, 128>>;

template <typename T>
class QrTdmSchedConfig : public ::testing::Test
{
};
TYPED_TEST_SUITE(QrTdmSchedConfig, Configs);

TYPED_TEST(QrTdmSchedConfig, GeometryAndDerivedTails)
{
    using G = typename TypeParam::Geometry;
    using P = typename TypeParam::Policy;
    using T = typename P::Tuning;
    EXPECT_EQ(G::kWaves, 4);
    EXPECT_EQ(G::kWaveSize, 32);
    EXPECT_EQ(G::kQueryRowsPerWave, 32);
    EXPECT_EQ(G::kQkStages, G::kN / 32);
    EXPECT_EQ(G::kPvStages, G::kN / 32);
    EXPECT_EQ(G::kKSuLoadCount, G::kHeadDimQK / 8);
    EXPECT_EQ(G::kVStageLoadCount, G::kDv / 8);
    EXPECT_EQ(G::kQkWmmasPerWaveKvTile, G::kN * G::kHeadDimQK / 256);
    EXPECT_EQ(G::kPvWmmasPerWaveKvTile, G::kN * G::kDv / 256);
    EXPECT_TRUE(G::PRepackLayout::kInThread);
    EXPECT_EQ(T::kKWait, 1);
    EXPECT_EQ(T::kVWait, 1);
    EXPECT_TRUE(T::kTailDrain);
    EXPECT_EQ(T::kBlockPerCu, G::kN == 128 ? 1 : G::kHeadDimQK == 64 ? 3 : 2);
    static_for<0, G::kQkStages, 1>{}([&](auto stage) {
        constexpr int s = decltype(stage)::value;
        EXPECT_EQ(T::template GetQkTailDsCount<s>(),
                  s + 1 == G::kQkStages ? G::kVStageLoadCount : G::kKSuLoadCount);
    });
    static_for<0, G::kPvStages, 1>{}([&](auto stage) {
        constexpr int s = decltype(stage)::value;
        EXPECT_EQ(T::template GetPvTailDsCount<s>(),
                  s + 1 == G::kPvStages ? G::kKSuLoadCount : G::kVStageLoadCount);
    });
}

TYPED_TEST(QrTdmSchedConfig, PackedLdsAndKernelContract)
{
    using G        = typename TypeParam::Geometry;
    using P        = typename TypeParam::Policy;
    using Problem  = typename TypeParam::Problem;
    using Pipeline = BlockFmhaPipelineQRKSVSTdmSched<Problem>;
    EXPECT_EQ(P::kKPhysicalStride, G::kHeadDimQK + 8);
    EXPECT_EQ(P::kVPhysicalStride, G::kDv + 16);
    EXPECT_EQ(P::kLdsOffsetK0, 0);
    EXPECT_EQ(P::kLdsOffsetK1, P::kKFootprintBytes);
    EXPECT_EQ(P::kLdsOffsetV0, 2 * P::kKFootprintBytes);
    EXPECT_EQ(P::kLdsOffsetV1, P::kLdsOffsetV0 + P::kVFootprintBytes);
    EXPECT_EQ(P::kLdsArenaSize, 2 * (P::kKFootprintBytes + P::kVFootprintBytes));
    EXPECT_EQ(P::kLdsOffsetK1 % 16, 0);
    EXPECT_EQ(P::kLdsOffsetV0 % 16, 0);
    EXPECT_EQ(P::kLdsOffsetV1 % 16, 0);
    EXPECT_LE(G::kM * G::kHeadDimQK * sizeof(typename G::QDataType),
              static_cast<std::size_t>(2 * P::kKFootprintBytes));
    EXPECT_EQ(Pipeline::GetSmemSize(), P::kLdsArenaSize);
    // The host compiler target uses wave64; gfx125 device code uses wave32.
    // Check the target-dependent expression here and the physical device shape below.
    EXPECT_EQ(Pipeline::kBlockSize, G::kWaves * get_warp_size());
#if defined(__HIP_DEVICE_COMPILE__)
    static_assert(Pipeline::kBlockSize == G::kWaves * G::kWaveSize);
    static_assert(Pipeline::kBlockSize == 128);
#endif
    EXPECT_TRUE((P::template IsSupportedProblem<Problem>()));
    EXPECT_FALSE((P::template IsSupportedProblem<
                  BiasProblem<Problem, BlockAttentionBiasEnum::ELEMENTWISE_BIAS>>()));
    EXPECT_FALSE(
        (P::template IsSupportedProblem<BiasProblem<Problem, BlockAttentionBiasEnum::ALIBI>>()));
    EXPECT_TRUE(detail::uses_untransposed_v_kernel_path_v<Pipeline>);
    EXPECT_TRUE(detail::uses_tdm_affine_dram_path_v<Pipeline>);
    EXPECT_FALSE(detail::uses_qr_tdm_lds_arena_v<Pipeline>);
}

TYPED_TEST(QrTdmSchedConfig, SelectedScheduleAndInvalidCounts)
{
    using G       = typename TypeParam::Geometry;
    using P       = typename TypeParam::Policy;
    using Problem = typename TypeParam::Problem;
    using Mode    = FmhaTdmSchedMode<Problem, P>;
    EXPECT_TRUE(Mode::kIsLegal);
    EXPECT_EQ(Mode::kEarlyOutputRescale, G::kHeadDimQK == 128);
    EXPECT_EQ(Mode::kStreamOutputRescale, (std::is_same_v<G, FmhaTdmSchedBf16D192N128Geometry>));
    EXPECT_EQ(P::OutputFragments::kNumFragments, G::kPvMIter * G::kPvNIter);
    EXPECT_FALSE((FmhaTdmSchedMode<Problem, WrongCountPolicy<P>>::kIsLegal));
}

TYPED_TEST(QrTdmSchedConfig, CompilerMaxScopeAcrossFlagsAndTypes)
{
    using G = typename TypeParam::Geometry;
    using P = typename TypeParam::Policy;
    // State order: batch dense, batch masked, group dense, group masked;
    // each pair has LSE disabled then enabled. Pin the shipped scope directly.
    constexpr std::array<bool, 8> d64_d128_scope{true, true, true, true, true, true, true, true};
    constexpr std::array<bool, 8> d192_scope{true, true, false, false, false, false, false, false};
    constexpr auto expected = G::kHeadDimQK == 192 ? d192_scope : d64_d128_scope;
    ExpectCompilerMaxFlagMatrix<P, bf16_t, bf16_t, bf16_t>(expected, "BF16");
    ExpectCompilerMaxFlagMatrix<P, half_t, half_t, half_t>(expected, "FP16");
    ExpectCompilerMaxFlagMatrix<P, const bf16_t&, bf16_t, bf16_t>(expected, "BF16 Q cvref");
    ExpectCompilerMaxFlagMatrix<P, fp8_t, fp8_t, fp8_t>(expected, "FP8 predicate only");
    ExpectCompilerMaxFlagMatrix<P, half_t, bf16_t, bf16_t>(expected, "mixed Q predicate only");
    ExpectCompilerMaxFlagMatrix<P, bf16_t, half_t, bf16_t>(expected, "mixed K predicate only");
    ExpectCompilerMaxFlagMatrix<P, bf16_t, bf16_t, half_t>(expected, "mixed V predicate only");
}

TYPED_TEST(QrTdmSchedConfig, OutputRescaleModesRejectConflictingOwners)
{
    using P       = typename TypeParam::Policy;
    using Problem = typename TypeParam::Problem;
    using Late    = FmhaTdmSchedMode<Problem, OutputRescalePolicy<P, false>>;
    using Early   = FmhaTdmSchedMode<Problem, OutputRescalePolicy<P, true>>;
    // Only the native BF16 D192/N128 schedule owns streaming rescale. Other
    // schedules permit early rescale or the post-softmax fragment alternative.
    constexpr bool streaming =
        std::is_same_v<typename TypeParam::Geometry, FmhaTdmSchedBf16D192N128Geometry>;
    EXPECT_FALSE(Late::kEarlyOutputRescale);
    EXPECT_EQ(Late::kStreamOutputRescale, streaming);
    EXPECT_EQ(Late::kPostSoftmaxFragmentRescale, !streaming);
    EXPECT_TRUE(Late::kIsLegal);
    EXPECT_TRUE(Early::kEarlyOutputRescale);
    EXPECT_EQ(Early::kStreamOutputRescale, streaming);
    EXPECT_FALSE(Early::kPostSoftmaxFragmentRescale);
    EXPECT_EQ(Early::kIsLegal, !streaming);
}

} // namespace
