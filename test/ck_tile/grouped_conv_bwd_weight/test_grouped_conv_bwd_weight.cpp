// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_grouped_conv_bwd_weight_util.hpp"

using GNHWC = ck_tile::tensor_layout::convolution::GNHWC;
using GKYXC = ck_tile::tensor_layout::convolution::GKYXC;
using GNHWK = ck_tile::tensor_layout::convolution::GNHWK;
using BF16  = ck_tile::bhalf_t;
using F16   = ck_tile::half_t;
using F32   = float;

struct GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2 
{
    static constexpr ck_tile::index_t M_Tile = 8;
    static constexpr ck_tile::index_t N_Tile = 128; 
    static constexpr ck_tile::index_t K_Tile = 64;
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 4;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
    static constexpr ck_tile::index_t VectorSizeA = 1;
    static constexpr ck_tile::index_t VectorSizeB = 1;
    static constexpr ck_tile::index_t VectorSizeC = 2;
};

using KernelTypes = ::testing::Types<
    // 2D Convolution Tests - FP16
    std::tuple<ck_tile::number<2>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, F16, F16, F32, F16, GNHWC, GKYXC, GNHWK, ck_tile::number<1>>,
    std::tuple<ck_tile::number<2>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, F16, F16, F32, F16, GNHWC, GKYXC, GNHWK, ck_tile::number<2>>,
    std::tuple<ck_tile::number<2>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, F16, F16, F32, F16, GNHWC, GKYXC, GNHWK, ck_tile::number<4>>,
    std::tuple<ck_tile::number<2>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, F16, F16, F32, F16, GNHWC, GKYXC, GNHWK, ck_tile::number<8>>,
    
    // 2D Convolution Tests - BF16  
    std::tuple<ck_tile::number<2>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, BF16, BF16, F32, BF16, GNHWC, GKYXC, GNHWK, ck_tile::number<1>>,
    std::tuple<ck_tile::number<2>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, BF16, BF16, F32, BF16, GNHWC, GKYXC, GNHWK, ck_tile::number<2>>,
    std::tuple<ck_tile::number<2>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, BF16, BF16, F32, BF16, GNHWC, GKYXC, GNHWK, ck_tile::number<4>>,
    std::tuple<ck_tile::number<2>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, BF16, BF16, F32, BF16, GNHWC, GKYXC, GNHWK, ck_tile::number<8>>,
    
    // 3D Convolution Tests - FP16
    std::tuple<ck_tile::number<3>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, F16, F16, F32, F16, GNHWC, GKYXC, GNHWK, ck_tile::number<1>>,
    std::tuple<ck_tile::number<3>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, F16, F16, F32, F16, GNHWC, GKYXC, GNHWK, ck_tile::number<2>>,
    std::tuple<ck_tile::number<3>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, F16, F16, F32, F16, GNHWC, GKYXC, GNHWK, ck_tile::number<4>>,
    std::tuple<ck_tile::number<3>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, F16, F16, F32, F16, GNHWC, GKYXC, GNHWK, ck_tile::number<8>>,
    
    // 3D Convolution Tests - BF16
    std::tuple<ck_tile::number<3>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, BF16, BF16, F32, BF16, GNHWC, GKYXC, GNHWK, ck_tile::number<1>>,
    std::tuple<ck_tile::number<3>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, BF16, BF16, F32, BF16, GNHWC, GKYXC, GNHWK, ck_tile::number<2>>,
    std::tuple<ck_tile::number<3>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, BF16, BF16, F32, BF16, GNHWC, GKYXC, GNHWK, ck_tile::number<4>>,
    std::tuple<ck_tile::number<3>, GemmConfig_8x128x64_2x2x1_4x64x16_1_1_2, BF16, BF16, F32, BF16, GNHWC, GKYXC, GNHWK, ck_tile::number<8>>
>;

TYPED_TEST_SUITE(TestCkTileGroupedConvBwdWeight, KernelTypes);

// Include the test cases

TYPED_TEST(TestCkTileGroupedConvBwdWeight, MergedConvGroups)
{
    GroupedConvBwdWeightHostArgs args;
    // TODO: Fill the arguments.
    this->run(args);
}