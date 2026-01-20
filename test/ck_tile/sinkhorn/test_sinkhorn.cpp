// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>
#include <cstring>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/sinkhorn_knopp.hpp"
#include "ck_tile/host/kernel_launch.hpp"

#include "test_sinkhorn_impl.hpp"

// Shape parameters for different test configurations
using Shape1_BlockWarps = ck_tile::sequence<4, 1>;
using Shape1_BlockTile  = ck_tile::sequence<128, 128>;
using Shape1_WarpTile   = ck_tile::sequence<32, 128>;
using Shape1_ThreadTile = ck_tile::sequence<8, 8>;

// Test configurations for different data types and input size
using TestConfig_F16 = std::tuple<
                                      ck_tile::half_t, // XDataType
                                      float,           // ComputeDataType
                                      float,           // YDataType
                                      Shape1_BlockWarps,
                                      Shape1_BlockTile,
                                      Shape1_WarpTile,
                                      Shape1_ThreadTile>;


using TestTypes = ::testing::Types<TestConfig_F16>;

TYPED_TEST_SUITE(TestCkTileSinkHorn, TestTypes);

TYPED_TEST(TestCkTileSinkHorn, Test_4x4)
{
    this->RunGenericTest({4, 4}, 10);
}
