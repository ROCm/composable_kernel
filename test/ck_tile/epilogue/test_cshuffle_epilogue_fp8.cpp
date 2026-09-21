// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_cshuffle_epilogue_common.hpp"

using namespace ck_tile;

// FP8 MFMA tile configurations with half_t output
// Using half_t output avoids FP8 range limitations while testing FP8-specific tile sizes
using FP8Config_128x128_2x2x1_16x16x16 = TileConfig<half_t, 128, 128, 2, 2, 16, 16, 16, fp8_t>;
using FP8Config_128x128_1x4x1_16x16x16 = TileConfig<half_t, 128, 128, 1, 4, 16, 16, 16, fp8_t>;
using FP8Config_128x128_4x1x1_16x16x16 = TileConfig<half_t, 128, 128, 4, 1, 16, 16, 16, fp8_t>;
using FP8Config_128x128_2x2x1_32x32x16 = TileConfig<half_t, 128, 128, 2, 2, 32, 32, 16, fp8_t>;
using FP8Config_128x128_2x2x1_16x16x32 = TileConfig<half_t, 128, 128, 2, 2, 16, 16, 32, fp8_t>;
using FP8Config_128x128_2x2x1_32x32x32 = TileConfig<half_t, 128, 128, 2, 2, 32, 32, 32, fp8_t>;
using FP8Config_128x128_2x2x1_16x16x64 = TileConfig<half_t, 128, 128, 2, 2, 16, 16, 64, fp8_t>;

using FP8TestTypes = ::testing::Types<FP8Config_128x128_2x2x1_16x16x16,
                                      FP8Config_128x128_1x4x1_16x16x16,
                                      FP8Config_128x128_4x1x1_16x16x16,
                                      FP8Config_128x128_2x2x1_32x32x16,
                                      FP8Config_128x128_2x2x1_16x16x32,
                                      FP8Config_128x128_2x2x1_32x32x32,
                                      FP8Config_128x128_2x2x1_16x16x64>;

CK_INSTANTIATE_TYPED_TEST_SUITE(FP8, CShuffleEpilogueTypedTest, FP8TestTypes)

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
