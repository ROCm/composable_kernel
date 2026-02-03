// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_cshuffle_epilogue_common.hpp"

using namespace ck_tile;

// Half precision test configurations
using HalfConfig_256x256_2x2x1_32x32x8  = TileConfig<half_t, 256, 256, 2, 2, 32, 32, 8>;
using HalfConfig_128x128_1x4x1_16x16x16 = TileConfig<half_t, 128, 128, 1, 4, 16, 16, 16>;
using HalfConfig_128x128_2x2x1_16x16x16 = TileConfig<half_t, 128, 128, 2, 2, 16, 16, 16>;
using HalfConfig_128x128_4x1x1_16x16x16 = TileConfig<half_t, 128, 128, 4, 1, 16, 16, 16>;
using HalfConfig_128x128_2x2x1_32x32x16 = TileConfig<half_t, 128, 128, 2, 2, 32, 32, 16>;

using HalfTestTypes = ::testing::Types<HalfConfig_256x256_2x2x1_32x32x8,
                                       HalfConfig_128x128_1x4x1_16x16x16,
                                       HalfConfig_128x128_2x2x1_16x16x16,
                                       HalfConfig_128x128_4x1x1_16x16x16,
                                       HalfConfig_128x128_2x2x1_32x32x16>;

CK_INSTANTIATE_TYPED_TEST_SUITE(FP16, CShuffleEpilogueTypedTest, HalfTestTypes)

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
