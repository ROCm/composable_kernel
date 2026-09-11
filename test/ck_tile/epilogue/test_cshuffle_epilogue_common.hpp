// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * CShuffleEpilogue Test Infrastructure
 *
 * File organization:
 * - test_cshuffle_epilogue_common.hpp: TileConfig template, verification helpers,
 *   typed test suite definition (this file)
 * - test_cshuffle_epilogue_util.hpp: Kernel templates, launch helpers, test runners
 * - test_cshuffle_epilogue_fp16.cpp: FP16 tile configurations
 * - test_cshuffle_epilogue_fp8.cpp: FP8 tile configurations (standard)
 * - test_cshuffle_epilogue_fp8_gfx950.cpp: FP8 configurations for gfx950
 * - test_cshuffle_epilogue_scale.cpp: RowCol and Tensor scaling tests
 */

#pragma once

#include "test_cshuffle_epilogue_util.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

// TileConfig defines a test configuration for CShuffleEpilogue.
// - ODataType_: The output data type written to global memory
// - MfmaDataType_: The data type used for MFMA instruction selection (determines valid KPerXdl)
//   Defaults to ODataType_ but can differ (e.g., FP8 MFMA tiles with FP16 output
//   to avoid FP8 range limitations in test verification)
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

// Verification helper: check that output contains valid data from the epilogue shuffle.
// The C-shuffle epilogue loads thread-local values and writes them to output through LDS.
// We verify: correct output size, no NaN values, no unwritten zeros, and at least
// kBlockSize unique values (one per thread).
template <typename DataType,
          ck_tile::index_t kMPerBlock,
          ck_tile::index_t kNPerBlock,
          ck_tile::index_t kBlockSize>
void verify_permutation_output(const std::vector<float>& sorted_vals)
{
    constexpr size_t expected_size = static_cast<size_t>(kMPerBlock * kNPerBlock);

    ASSERT_EQ(sorted_vals.size(), expected_size) << "Output size mismatch";

    // Verify no NaN values
    for(size_t i = 0; i < sorted_vals.size(); ++i)
    {
        ASSERT_FALSE(std::isnan(sorted_vals[i])) << "NaN at index " << i;
    }

    // Count unique values using bit-exact comparison (sorted fp32 values from fp16 should be
    // distinct)
    size_t num_unique = 1;
    for(size_t i = 1; i < sorted_vals.size(); ++i)
    {
        if(ck_tile::bit_cast<uint32_t>(sorted_vals[i]) !=
           ck_tile::bit_cast<uint32_t>(sorted_vals[i - 1]))
        {
            ++num_unique;
        }
    }

    // Verify exact permutation: all input values should appear exactly once in output
    EXPECT_EQ(num_unique, expected_size) << "Expected exact permutation with " << expected_size
                                         << " unique values, got " << num_unique;
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

    auto host_output = ck_tile::run_cshuffle_epilogue_test<TestProblem, kMPerBlock, kNPerBlock>(
        ck_tile::ScaleType::None);

    // Convert output to sorted vector and verify using existing helper
    auto output_vals = ck_tile::convert_and_sort_output(host_output);
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
