// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_cshuffle_epilogue_common.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

using namespace ck_tile;

// Half precision test configuration for scale tests
using HalfConfig       = TileConfig<half_t, 256, 256, 2, 2, 32, 32, 8>;
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
    constexpr index_t kScaledCol   = verification::kScaledColIndex;

    size_t col0_unchanged_count = 0;
    size_t col1_scaled_count    = 0;

    for(index_t row = 0; row < rows_to_check; ++row)
    {
        const size_t col0_idx = static_cast<size_t>(row * HalfConfig::kNPerBlock + kUnscaledCol);
        const size_t col1_idx = static_cast<size_t>(row * HalfConfig::kNPerBlock + kScaledCol);

        const auto unscaled_col0 = type_convert<float>(results.unscaled.output.mData[col0_idx]);
        const auto scaled_col0   = type_convert<float>(results.scaled.output.mData[col0_idx]);
        const auto unscaled_col1 = type_convert<float>(results.unscaled.output.mData[col1_idx]);
        const auto scaled_col1   = type_convert<float>(results.scaled.output.mData[col1_idx]);

        // Count rows where column 0 is unchanged (scale = kIdentityScale)
        if(std::abs(scaled_col0 - unscaled_col0) < verification::kScaleEpsilon)
        {
            col0_unchanged_count++;
        }

        // Count rows where column 1 is scaled by kTestScaleFactor
        const float expected_scaled = unscaled_col1 * verification::kTestScaleFactor;
        if(std::abs(scaled_col1 - expected_scaled) < verification::kScaleEpsilon)
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
    auto unscaled_vals = convert_and_sort_output(results.unscaled.output);
    auto scaled_vals   = convert_and_sort_output(results.scaled.output);

    // With Tensor scaling (m_scale=kTestScaleFactor, n_scale=kIdentityScale),
    // all values should be scaled by kTestScaleFactor
    EXPECT_EQ(unscaled_vals.size(), scaled_vals.size()) << "Tensor scale: output sizes differ";

    for(size_t i = 0; i < unscaled_vals.size(); ++i)
    {
        const float expected = unscaled_vals[i] * verification::kTestScaleFactor;
        EXPECT_NEAR(scaled_vals[i], expected, verification::kScaleEpsilon)
            << "Tensor scale: sorted scaled[" << i << "]=" << scaled_vals[i] << " should be "
            << verification::kTestScaleFactor << "x " << unscaled_vals[i];
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
