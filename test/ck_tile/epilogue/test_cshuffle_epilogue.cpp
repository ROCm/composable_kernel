// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_cshuffle_epilogue_util.hpp"
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

using namespace ck_tile;

// Test configuration template for parameterized tests
template <typename DataType_,
          index_t MPerBlock_,
          index_t NPerBlock_,
          index_t MWave_,
          index_t NWave_,
          index_t MPerXdl_,
          index_t NPerXdl_,
          index_t KPerXdl_>
struct TileConfig
{
    using DataType                      = DataType_;
    static constexpr index_t kMPerBlock = MPerBlock_;
    static constexpr index_t kNPerBlock = NPerBlock_;
    static constexpr index_t MWave      = MWave_;
    static constexpr index_t NWave      = NWave_;
    static constexpr index_t MPerXdl    = MPerXdl_;
    static constexpr index_t NPerXdl    = NPerXdl_;
    static constexpr index_t KPerXdl    = KPerXdl_;
};

// Type-parameterized test fixture
template <typename Config>
class CShuffleEpilogueTypedTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(CShuffleEpilogueTypedTest);

TYPED_TEST_P(CShuffleEpilogueTypedTest, BasicTest)
{
    using Config      = TypeParam;
    using DataType    = typename Config::DataType;
    using ADataType   = DataType;
    using BDataType   = DataType;
    using AccDataType = float;
    using ODataType   = DataType;

    constexpr index_t kMPerBlock = Config::kMPerBlock;
    constexpr index_t kNPerBlock = Config::kNPerBlock;
    constexpr index_t MWave      = Config::MWave;
    constexpr index_t NWave      = Config::NWave;
    constexpr index_t MPerXdl    = Config::MPerXdl;
    constexpr index_t NPerXdl    = Config::NPerXdl;
    constexpr index_t KPerXdl    = Config::KPerXdl;

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

    auto result = run_cshuffle_epilogue_test<TestProblem, kMPerBlock, kNPerBlock>(ScaleType::None);

    if constexpr(std::is_same_v<DataType, ck_tile::fp8_t>)
    {
        EXPECT_EQ(result[0], ck_tile::type_convert<ck_tile::fp8_t>(2.f))
            << "CShuffleEpilogue FP8 test failed";
    }
    else
    {
        EXPECT_FLOAT_EQ(ck_tile::type_convert<float>(result[0]), 2.0F)
            << "CShuffleEpilogue test failed";
    }
}

REGISTER_TYPED_TEST_SUITE_P(CShuffleEpilogueTypedTest, BasicTest);

// Half precision test configurations
using HalfConfig_256x256_2x2x1_32x32x8  = TileConfig<half_t, 256, 256, 2, 2, 32, 32, 8>;
using HalfConfig_128x128_1x4x1_16x16x16 = TileConfig<half_t, 128, 128, 1, 4, 16, 16, 16>;
using HalfConfig_128x128_2x2x1_16x16x16 = TileConfig<half_t, 128, 128, 2, 2, 16, 16, 16>;
using HalfConfig_128x128_4x1x1_16x16x16 = TileConfig<half_t, 128, 128, 4, 1, 16, 16, 16>;
using HalfConfig_128x128_2x2x1_32x32x16 = TileConfig<half_t, 128, 128, 2, 2, 32, 32, 16>;

// FP8 test configurations
using FP8Config_128x128_2x2x1_16x16x16 = TileConfig<fp8_t, 128, 128, 2, 2, 16, 16, 16>;
using FP8Config_128x128_1x4x1_16x16x16 = TileConfig<fp8_t, 128, 128, 1, 4, 16, 16, 16>;
using FP8Config_128x128_4x1x1_16x16x16 = TileConfig<fp8_t, 128, 128, 4, 1, 16, 16, 16>;
using FP8Config_128x128_2x2x1_32x32x16 = TileConfig<fp8_t, 128, 128, 2, 2, 32, 32, 16>;
using FP8Config_128x128_2x2x1_16x16x32 = TileConfig<fp8_t, 128, 128, 2, 2, 16, 16, 32>;
using FP8Config_128x128_2x2x1_32x32x32 = TileConfig<fp8_t, 128, 128, 2, 2, 32, 32, 32>;
using FP8Config_128x128_2x2x1_16x16x64 = TileConfig<fp8_t, 128, 128, 2, 2, 16, 16, 64>;

using HalfTestTypes = ::testing::Types<HalfConfig_256x256_2x2x1_32x32x8,
                                       HalfConfig_128x128_1x4x1_16x16x16,
                                       HalfConfig_128x128_2x2x1_16x16x16,
                                       HalfConfig_128x128_4x1x1_16x16x16,
                                       HalfConfig_128x128_2x2x1_32x32x16>;

using FP8TestTypes = ::testing::Types<FP8Config_128x128_2x2x1_16x16x16,
                                      FP8Config_128x128_1x4x1_16x16x16,
                                      FP8Config_128x128_4x1x1_16x16x16,
                                      FP8Config_128x128_2x2x1_32x32x16,
                                      FP8Config_128x128_2x2x1_16x16x32,
                                      FP8Config_128x128_2x2x1_32x32x32,
                                      FP8Config_128x128_2x2x1_16x16x64>;

// clang-format off
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
INSTANTIATE_TYPED_TEST_SUITE_P(Half, CShuffleEpilogueTypedTest, HalfTestTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(FP8, CShuffleEpilogueTypedTest, FP8TestTypes);
#pragma clang diagnostic pop
// clang-format on

// Additional tests for scale operations (not parameterized due to different verification logic)
class CShuffleEpilogueScaleTest : public ::testing::Test
{
};

TEST_F(CShuffleEpilogueScaleTest, HalfTestWithRowColScale)
{
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

    auto result =
        run_cshuffle_epilogue_test<TestProblem, kMPerBlock, kNPerBlock>(ScaleType::RowCol);
    EXPECT_FLOAT_EQ(result[0], 2.0F) << "RowCol scale test failed: first element not 2";
    EXPECT_FLOAT_EQ(result[1], 4.0F) << "RowCol scale test failed: second element not 2*2";
}

TEST_F(CShuffleEpilogueScaleTest, HalfTestWithTensorScale)
{
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

    auto result =
        run_cshuffle_epilogue_test<TestProblem, kMPerBlock, kNPerBlock>(ScaleType::Tensor);
    EXPECT_FLOAT_EQ(result[0], 4.0F) << "Tensor scale test failed: first element not 2*2=4";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
