// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "test_cshuffle_epilogue_util.hpp"
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

using namespace ck_tile;

class CShuffleEpilogueTest : public ::testing::Test
{
    protected:
    void SetUp() override {}
};

TEST_F(CShuffleEpilogueTest, BasicHalfTest)
{
    // Basic test configuration with half_t data types
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using ODataType   = ck_tile::half_t;

    constexpr index_t kMPerBlock = 256;
    constexpr index_t kNPerBlock = 256;
    constexpr index_t MWave      = 2;
    constexpr index_t NWave      = 2;
    constexpr index_t MPerXdl    = 32;
    constexpr index_t NPerXdl    = 32;
    constexpr index_t KPerXdl    = 8;

    using TestProblem = SimpleCShuffleEpilogueProblem<ADataType,
                                                      BDataType,
                                                      AccDataType,
                                                      ODataType,
                                                      kMPerBlock,
                                                      kNPerBlock,
                                                      MWave,
                                                      NWave,
                                                      MPerXdl,
                                                      NPerXdl,
                                                      KPerXdl>;

    bool result = run_cshuffle_epilogue_test<TestProblem, kMPerBlock, kNPerBlock>();
    EXPECT_TRUE(result) << "Basic CShuffleEpilogue test failed";
}

TEST_F(CShuffleEpilogueTest, BasicHalfTestWithScale)
{
    // Basic test configuration with half_t data types
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using ODataType   = ck_tile::half_t;

    constexpr index_t kMPerBlock = 256;
    constexpr index_t kNPerBlock = 256;
    constexpr index_t MWave      = 2;
    constexpr index_t NWave      = 2;
    constexpr index_t MPerXdl    = 32;
    constexpr index_t NPerXdl    = 32;
    constexpr index_t KPerXdl    = 8;

    using TestProblem = SimpleCShuffleEpilogueProblem<ADataType,
                                                      BDataType,
                                                      AccDataType,
                                                      ODataType,
                                                      kMPerBlock,
                                                      kNPerBlock,
                                                      MWave,
                                                      NWave,
                                                      MPerXdl,
                                                      NPerXdl,
                                                      KPerXdl>;

    bool result = run_cshuffle_epilogue_test<TestProblem, kMPerBlock, kNPerBlock>(true);
    EXPECT_TRUE(result) << "Scale CShuffleEpilogue test failed";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
