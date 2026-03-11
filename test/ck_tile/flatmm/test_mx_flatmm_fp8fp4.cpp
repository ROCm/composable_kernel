// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include <gtest/gtest.h>
#include "test_mx_flatmm_fixtures.hpp"

// FP8 x FP4 -> FP16
// N_Tile = 256, K must be a multiple of lcm(32, 8) = 32.
// clang-format off
using FP8FP4Types = ::testing::Types<
    std::tuple<FP8, FP4, FP16, MXFlatmm_GFX950_FP8FP4_Traits>
>;
// clang-format on

TYPED_TEST_SUITE(TestMXFlatmm, FP8FP4Types);

// K=256 -> num_loop=1: has_hot_loop=false, tail=Odd
TYPED_TEST(TestMXFlatmm, SmallMNK)
{
    this->run_test_with_validation(128, 256, 256, 1, false, ck_tile::TailNumber::Odd);
}

// K=512 -> num_loop=2: has_hot_loop=false, tail=Even
TYPED_TEST(TestMXFlatmm, MediumMNK)
{
    this->run_test_with_validation(256, 512, 512, 1, false, ck_tile::TailNumber::Even);
}

// K=768 -> num_loop=3: has_hot_loop=true, tail=Odd
TYPED_TEST(TestMXFlatmm, LargeK_HotLoopOdd)
{
    this->run_test_with_validation(128, 256, 768, 1, true, ck_tile::TailNumber::Odd);
}

// K=1024 -> num_loop=4: has_hot_loop=true, tail=Even
TYPED_TEST(TestMXFlatmm, LargeK_HotLoopEven)
{
    this->run_test_with_validation(128, 256, 1024, 1, true, ck_tile::TailNumber::Even);
}
