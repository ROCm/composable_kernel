// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "test_cshuffle_epilogue_util.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

// Test configuration template for parameterized tests
// MfmaDataType is used for MFMA instruction selection (determines valid KPerXdl values)
// ODataType is the output data type
template <typename ODataType_,
          ck_tile::index_t MPerBlock_,
          ck_tile::index_t NPerBlock_,
          ck_tile::index_t MWave_,
          ck_tile::index_t NWave_,
          ck_tile::index_t MPerXdl_,
          ck_tile::index_t NPerXdl_,
          ck_tile::index_t KPerXdl_,
          typename MfmaDataType_ = ODataType_>
struct TileConfig
{
    using DataType                               = ODataType_;
    using MfmaDataType                           = MfmaDataType_;
    static constexpr ck_tile::index_t kMPerBlock = MPerBlock_;
    static constexpr ck_tile::index_t kNPerBlock = NPerBlock_;
    static constexpr ck_tile::index_t MWave      = MWave_;
    static constexpr ck_tile::index_t NWave      = NWave_;
    static constexpr ck_tile::index_t MPerXdl    = MPerXdl_;
    static constexpr ck_tile::index_t NPerXdl    = NPerXdl_;
    static constexpr ck_tile::index_t KPerXdl    = KPerXdl_;
};

// Helper to construct SimpleCShuffleEpilogueProblem from TileConfig
// Uses MfmaDataType for MFMA input types (A/B) and DataType for output
template <typename Config, typename AccDataType = float>
using MakeProblem = ck_tile::SimpleCShuffleEpilogueProblem<typename Config::MfmaDataType,
                                                           typename Config::MfmaDataType,
                                                           AccDataType,
                                                           typename Config::DataType,
                                                           Config::kMPerBlock,
                                                           Config::kNPerBlock,
                                                           Config::MWave,
                                                           Config::NWave,
                                                           Config::MPerXdl,
                                                           Config::NPerXdl,
                                                           Config::KPerXdl>;

// Verification helper: check that output contains valid data from the epilogue shuffle
// The C-shuffle epilogue broadcasts thread-local values to multiple output locations,
// so we verify: no NaN/zeros, reasonable value range, and at least kBlockSize unique values
// (since each thread generates unique values)
template <typename DataType,
          ck_tile::index_t kMPerBlock,
          ck_tile::index_t kNPerBlock,
          ck_tile::index_t kBlockSize>
void verify_permutation_output(const std::vector<float>& sorted_vals)
{
    constexpr size_t expected_size = static_cast<size_t>(kMPerBlock * kNPerBlock);

    // Verify output size matches expected
    ASSERT_EQ(sorted_vals.size(), expected_size) << "CShuffleEpilogue output size mismatch";

    // Verify no NaN values
    for(size_t i = 0; i < sorted_vals.size(); ++i)
    {
        ASSERT_FALSE(std::isnan(sorted_vals[i]))
            << "CShuffleEpilogue output contains NaN at index " << i;
    }

    // Verify all values are positive (no zeros from unwritten memory)
    EXPECT_GT(sorted_vals.front(), 0.0f) << "CShuffleEpilogue output contains zero values";

    // Count unique values and track occurrence counts for uniformity check
    std::vector<size_t> occurrence_counts;
    size_t current_count = 1;
    for(size_t i = 1; i < sorted_vals.size(); ++i)
    {
        if(std::abs(sorted_vals[i] - sorted_vals[i - 1]) > ck_tile::verification::kScaleEpsilon)
        {
            occurrence_counts.push_back(current_count);
            current_count = 1;
        }
        else
        {
            ++current_count;
        }
    }
    occurrence_counts.push_back(current_count); // Don't forget the last value

    const size_t num_unique = occurrence_counts.size();

    // Each thread generates unique values, so we expect at least kBlockSize unique values
    // This verifies that all threads contributed to the output
    EXPECT_GE(num_unique, static_cast<size_t>(kBlockSize))
        << "CShuffleEpilogue output has fewer unique values (" << num_unique
        << ") than threads per block (" << kBlockSize << ")";

    // Check if distribution is uniform (all values appear same number of times)
    const size_t first_count = occurrence_counts[0];
    bool is_uniform          = true;
    size_t min_count         = first_count;
    size_t max_count         = first_count;

    for(size_t count : occurrence_counts)
    {
        if(count != first_count)
        {
            is_uniform = false;
        }
        min_count = std::min(min_count, count);
        max_count = std::max(max_count, count);
    }

    if(is_uniform)
    {
        // Uniform distribution: verify exact counts
        const size_t expected_count = expected_size / num_unique;
        EXPECT_EQ(first_count, expected_count) << "Uniform distribution but count " << first_count
                                               << " != expected " << expected_count;
        EXPECT_EQ(expected_size % num_unique, 0u)
            << "Output size " << expected_size << " not evenly divisible by " << num_unique;
    }
    else
    {
        // Non-uniform distribution: log for investigation
        std::cout << "  [INFO] Non-uniform distribution detected: " << num_unique
                  << " unique values, counts range [" << min_count << ", " << max_count << "]"
                  << std::endl;
    }
}

// Type-parameterized test fixture
template <typename Config>
class CShuffleEpilogueTypedTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(CShuffleEpilogueTypedTest);

TYPED_TEST_P(CShuffleEpilogueTypedTest, BasicTest)
{
    using Config   = TypeParam;
    using DataType = typename Config::DataType;

    constexpr ck_tile::index_t kMPerBlock = Config::kMPerBlock;
    constexpr ck_tile::index_t kNPerBlock = Config::kNPerBlock;

    using TestProblem                     = MakeProblem<Config>;
    constexpr ck_tile::index_t kBlockSize = TestProblem::kBlockSize;

    auto test_result = ck_tile::run_cshuffle_epilogue_test<TestProblem, kMPerBlock, kNPerBlock>(
        ck_tile::ScaleType::None);

    // Convert output to sorted vector and verify
    auto output_vals = ck_tile::convert_and_sort_output(test_result.output);
    verify_permutation_output<DataType, kMPerBlock, kNPerBlock, kBlockSize>(output_vals);
}

REGISTER_TYPED_TEST_SUITE_P(CShuffleEpilogueTypedTest, BasicTest);

// Allow this test suite to be included without instantiation (e.g., in scale tests)
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(CShuffleEpilogueTypedTest);

// Macro to instantiate typed test suites with suppressed clang warnings
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CK_INSTANTIATE_TYPED_TEST_SUITE(Prefix, Suite, Types)            \
    _Pragma("clang diagnostic push")                                     \
        _Pragma("clang diagnostic ignored \"-Wused-but-marked-unused\"") \
            INSTANTIATE_TYPED_TEST_SUITE_P(Prefix, Suite, Types);        \
    _Pragma("clang diagnostic pop")
