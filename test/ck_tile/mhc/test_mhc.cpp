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
#include "ck_tile/host/kernel_launch.hpp"

#include "test_mhc_impl.hpp"

// Shape parameters for different test configurations
using Shape1_BlockWarps = ck_tile::sequence<4, 1>;
using Shape1_BlockTile  = ck_tile::sequence<128, 128>;
using Shape1_WarpTile   = ck_tile::sequence<32, 128>;
using Shape1_ThreadTile = ck_tile::sequence<8, 8>;

// Test configurations for different data types
using TestConfig_F16_Basic = std::tuple< // TODO,
    Shape1_BlockWarps,
    Shape1_BlockTile,
    Shape1_WarpTile,
    Shape1_ThreadTile>;

using TestTypes = ::testing::Types<TestConfig_F16_Basic>;

TYPED_TEST_SUITE(TestCkTileMHC, TestTypes);

TYPED_TEST(TestCkTileMHC, TestBasic)
{
    // this->RunTest2D_KeepDim0_ReduceDim1(64, 32);
    // TODO
}
