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

// Temporarily disable old tests that use runtime parameters
// TYPED_TEST(TestCkTileMHC, TestBasic) { this->RunExpansionParallelTest(); }

// TYPED_TEST(TestCkTileMHC, TestArbitraryBatchSize) { this->RunArbitraryBatchSizeTest(); }

// Explicit test cases for specific batch sizes (using template syntax)
TYPED_TEST(TestCkTileMHC, TestBatchSize1) { this->template RunBatchSizeTest<1>(); }

TYPED_TEST(TestCkTileMHC, TestBatchSize16) { this->template RunBatchSizeTest<16>(); }

TYPED_TEST(TestCkTileMHC, TestBatchSize17) { this->template RunBatchSizeTest<17>(); }

TYPED_TEST(TestCkTileMHC, TestBatchSize32) { this->template RunBatchSizeTest<32>(); }

TYPED_TEST(TestCkTileMHC, TestBatchSize64) { this->template RunBatchSizeTest<64>(); }

TYPED_TEST(TestCkTileMHC, TestBatchSize100) { this->template RunBatchSizeTest<100>(); }

// Test with different expansion factors (keeping nC <= 256)
TYPED_TEST(TestCkTileMHC, TestBatchSize16N2C128) { this->template RunBatchSizeTest<16, 2, 128>(); }

TYPED_TEST(TestCkTileMHC, TestBatchSize32N3C85) { this->template RunBatchSizeTest<32, 3, 85>(); }

TYPED_TEST(TestCkTileMHC, TestBatchSize8N8C32) { this->template RunBatchSizeTest<8, 8, 32>(); }
