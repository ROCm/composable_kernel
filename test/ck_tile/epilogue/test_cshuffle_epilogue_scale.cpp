// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_cshuffle_epilogue_common.hpp"

using namespace ck_tile;

namespace {
constexpr float kScaleEpsilon              = 0.001F;
constexpr float kTestScaleFactor           = 2.0F;
constexpr ck_tile::index_t kScaledColIndex = 1;
} // namespace

// Half precision test configuration for scale tests (128x128 fits in unique fp16 range)
using HalfConfig       = TileConfig<half_t, 128, 128, 2, 2, 32, 32, 8>;
using ScaleTestProblem = MakeProblem<HalfConfig>;

class CShuffleEpilogueScaleTest : public ::testing::Test
{
};

TEST_F(CShuffleEpilogueScaleTest, HalfTestWithRowColScale)
{
    // Run both unscaled and scaled tests
    auto results = run_scale_comparison_test<ScaleTestProblem,
                                             HalfConfig::kMPerBlock,
                                             HalfConfig::kNPerBlock,
                                             ScaleType::RowCol>();

    // With RowCol scaling, column kScaledColIndex is scaled by kTestScaleFactor
    // while other columns are scaled by kIdentityScale.
    // Verify scaling behavior for the first MPerXdl * MWave rows.
    const index_t rows_to_check =
        std::min(HalfConfig::kMPerBlock, HalfConfig::MPerXdl * HalfConfig::MWave);

    constexpr index_t kUnscaledCol = 0;
    constexpr index_t kScaledCol   = kScaledColIndex;

    size_t col0_unchanged_count = 0;
    size_t col1_scaled_count    = 0;

    for(index_t row = 0; row < rows_to_check; ++row)
    {
        const size_t col0_idx = static_cast<size_t>(row * HalfConfig::kNPerBlock + kUnscaledCol);
        const size_t col1_idx = static_cast<size_t>(row * HalfConfig::kNPerBlock + kScaledCol);

        const auto unscaled_col0 = type_convert<float>(results.first.mData[col0_idx]);
        const auto scaled_col0   = type_convert<float>(results.second.mData[col0_idx]);
        const auto unscaled_col1 = type_convert<float>(results.first.mData[col1_idx]);
        const auto scaled_col1   = type_convert<float>(results.second.mData[col1_idx]);

        // Count rows where column 0 is unchanged (scale = kIdentityScale)
        if(std::abs(scaled_col0 - unscaled_col0) < kScaleEpsilon)
        {
            col0_unchanged_count++;
        }

        // Count rows where column 1 is scaled by kTestScaleFactor
        const float expected_scaled = unscaled_col1 * kTestScaleFactor;
        if(std::abs(scaled_col1 - expected_scaled) < kScaleEpsilon)
        {
            col1_scaled_count++;
        }
    }

    // All rows must have correct scaling
    EXPECT_EQ(col0_unchanged_count, static_cast<size_t>(rows_to_check))
        << "RowCol: not all rows have unchanged col0";
    EXPECT_EQ(col1_scaled_count, static_cast<size_t>(rows_to_check))
        << "RowCol: not all rows have scaled col1";
}

TEST_F(CShuffleEpilogueScaleTest, HalfTestWithTensorScale)
{
    // Run both unscaled and scaled tests
    auto results = run_scale_comparison_test<ScaleTestProblem,
                                             HalfConfig::kMPerBlock,
                                             HalfConfig::kNPerBlock,
                                             ScaleType::Tensor>();

    // Convert both to sorted vectors using helper
    auto unscaled_vals = convert_and_sort_output(results.first);
    auto scaled_vals   = convert_and_sort_output(results.second);

    // With Tensor scaling (m_scale=kTestScaleFactor, n_scale=kIdentityScale),
    // all values should be scaled by kTestScaleFactor
    EXPECT_EQ(unscaled_vals.size(), scaled_vals.size()) << "Tensor scale: output sizes differ";

    for(size_t i = 0; i < unscaled_vals.size(); ++i)
    {
        const float expected = unscaled_vals[i] * kTestScaleFactor;
        EXPECT_NEAR(scaled_vals[i], expected, kScaleEpsilon)
            << "Tensor scale: sorted scaled[" << i << "]=" << scaled_vals[i] << " should be "
            << kTestScaleFactor << "x " << unscaled_vals[i];
    }
}

// Global test environment to check for wave32 devices
class Wave32CheckEnvironment : public ::testing::Environment
{
    public:
    void SetUp() override
    {
        int warp_size  = 0;
        hipError_t err = hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0);
        if(err == hipSuccess && warp_size == 32)
        {
            GTEST_SKIP() << "CShuffleEpilogue tests not supported on wave32 devices";
        }
    }
};

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new Wave32CheckEnvironment);
    return RUN_ALL_TESTS();
}
