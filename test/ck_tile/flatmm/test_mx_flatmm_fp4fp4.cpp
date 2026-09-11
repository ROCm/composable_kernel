// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include <gtest/gtest.h>
#include "test_mx_flatmm_fixtures.hpp"

// FP4 x FP4 -> FP16
// N_Tile = 512 (MXfp4_FlatmmConfig16), so N must be a multiple of 512.
// K must be a multiple of 32 (ScaleGranularityK) and 8 (FP4 PackedSize) -> multiple of 32.
//
// Compile-time arch dispatch via GetCurrentTargetId(). The GFX1250 branch is V1-only:
// MXFlatmmTDM_GFX1250_FP4FP4_Traits is intentionally omitted because the
// FP4xFP4 TDM path is confirmed numerically broken. The matching kernel
// instance is also skipped at the CMake level (test/ck_tile/flatmm/
// CMakeLists.txt).
// TODO: Re-enable both together once the kernel is fixed.
// clang-format off
using FP4FP4Types = std::conditional_t<
    GetCurrentTargetId() == ck_tile::core::arch::TargetId::GFX1250,
    ::testing::Types<
        std::tuple<FP4, FP4, FP16, MXFlatmm_GFX1250_FP4FP4_Traits>
    >,
    ::testing::Types<
        std::tuple<FP4, FP4, FP16, MXFlatmm_GFX950_FP4FP4_Traits>
    >
>;
// clang-format on

TYPED_TEST_SUITE(TestMXFlatmm, FP4FP4Types);

// K=256 -> num_loop=1: has_hot_loop=false, tail=Odd
TYPED_TEST(TestMXFlatmm, SmallMNK)
{
    this->run_test_with_validation(128, 512, 256, 1, false, ck_tile::TailNumber::Odd);
}

// K=512 -> num_loop=2: has_hot_loop=false, tail=Even
TYPED_TEST(TestMXFlatmm, MediumMNK)
{
    this->run_test_with_validation(256, 1024, 512, 1, false, ck_tile::TailNumber::Even);
}

// K=768 -> num_loop=3: has_hot_loop=true, tail=Odd
TYPED_TEST(TestMXFlatmm, LargeK_HotLoopOdd)
{
    this->run_test_with_validation(128, 512, 768, 1, true, ck_tile::TailNumber::Odd);
}

// K=1024 -> num_loop=4: has_hot_loop=true, tail=Even
TYPED_TEST(TestMXFlatmm, LargeK_HotLoopEven)
{
    this->run_test_with_validation(128, 512, 1024, 1, true, ck_tile::TailNumber::Even);
}
