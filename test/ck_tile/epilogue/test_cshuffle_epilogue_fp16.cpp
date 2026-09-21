// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_cshuffle_epilogue_common.hpp"

using namespace ck_tile;

// Half precision test configurations (128x128 = 16384 elements fits in unique fp16 range)
using HalfConfig_128x128_2x2x1_32x32x8  = TileConfig<half_t, 128, 128, 2, 2, 32, 32, 8>;
using HalfConfig_128x128_1x4x1_16x16x16 = TileConfig<half_t, 128, 128, 1, 4, 16, 16, 16>;
using HalfConfig_128x128_2x2x1_16x16x16 = TileConfig<half_t, 128, 128, 2, 2, 16, 16, 16>;
using HalfConfig_128x128_4x1x1_16x16x16 = TileConfig<half_t, 128, 128, 4, 1, 16, 16, 16>;
using HalfConfig_128x128_2x2x1_32x32x16 = TileConfig<half_t, 128, 128, 2, 2, 32, 32, 16>;

using HalfTestTypes = ::testing::Types<HalfConfig_128x128_2x2x1_32x32x8,
                                       HalfConfig_128x128_1x4x1_16x16x16,
                                       HalfConfig_128x128_2x2x1_16x16x16,
                                       HalfConfig_128x128_4x1x1_16x16x16,
                                       HalfConfig_128x128_2x2x1_32x32x16>;

CK_INSTANTIATE_TYPED_TEST_SUITE(FP16, CShuffleEpilogueTypedTest, HalfTestTypes)

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
