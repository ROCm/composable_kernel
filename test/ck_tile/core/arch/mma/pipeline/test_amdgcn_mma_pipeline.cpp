// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_pipeline.hpp"

namespace {
using namespace ck_tile::core::arch::mma;
}

TEST(MmaPipelineOptionFlagsTests, ConversionTests)
{
    MmaPipelineOptionFlags flags_0{};
    MmaPipelineOptionFlags flags_1{MmaPipelineOptionFlag::ABSwap};
    MmaPipelineOptionFlags flags_2{MmaPipelineOptionFlag::COMPRESS_A};
    MmaPipelineOptionFlags flags_3{0b11}; // TODO c++20 - remove this

    EXPECT_TRUE(flags_0.testFlag(MmaPipelineOptionFlag::NONE));
    EXPECT_FALSE(flags_0.testFlag(MmaPipelineOptionFlag::ABSwap));
    EXPECT_FALSE(flags_0.testFlag(MmaPipelineOptionFlag::COMPRESS_A));

    EXPECT_TRUE(flags_1.testFlag(MmaPipelineOptionFlag::ABSwap));
    EXPECT_FALSE(flags_1.testFlag(MmaPipelineOptionFlag::NONE));
    EXPECT_FALSE(flags_1.testFlag(MmaPipelineOptionFlag::COMPRESS_A));

    EXPECT_TRUE(flags_2.testFlag(MmaPipelineOptionFlag::COMPRESS_A));
    EXPECT_FALSE(flags_2.testFlag(MmaPipelineOptionFlag::NONE));
    EXPECT_FALSE(flags_2.testFlag(MmaPipelineOptionFlag::ABSwap));

    EXPECT_TRUE(flags_3.testFlag(MmaPipelineOptionFlag::COMPRESS_A));
    EXPECT_TRUE(flags_3.testFlag(MmaPipelineOptionFlag::ABSwap));
    EXPECT_FALSE(flags_3.testFlag(MmaPipelineOptionFlag::NONE));
}

TEST(MmaPipelineOptionFlagsTests, OperatorsTests)
{
    MmaPipelineOptionFlags flags{};

    EXPECT_TRUE(flags.testFlag(MmaPipelineOptionFlag::NONE));

    flags |= MmaPipelineOptionFlag::ABSwap;

    EXPECT_FALSE(flags.testFlag(MmaPipelineOptionFlag::NONE));
    EXPECT_TRUE(flags.testFlag(MmaPipelineOptionFlag::ABSwap));

    flags |= MmaPipelineOptionFlag::COMPRESS_A;

    EXPECT_FALSE(flags.testFlag(MmaPipelineOptionFlag::NONE));
    EXPECT_TRUE(flags.testFlag(MmaPipelineOptionFlag::ABSwap));
    EXPECT_TRUE(flags.testFlag(MmaPipelineOptionFlag::COMPRESS_A));

    flags &= MmaPipelineOptionFlag::COMPRESS_A;

    EXPECT_FALSE(flags.testFlag(MmaPipelineOptionFlag::NONE));
    EXPECT_FALSE(flags.testFlag(MmaPipelineOptionFlag::ABSwap));
    EXPECT_TRUE(flags.testFlag(MmaPipelineOptionFlag::COMPRESS_A));

    EXPECT_FALSE((~flags).testFlag(MmaPipelineOptionFlag::NONE));
    EXPECT_TRUE((~flags).testFlag(MmaPipelineOptionFlag::ABSwap));
    EXPECT_FALSE((~flags).testFlag(MmaPipelineOptionFlag::COMPRESS_A));
}
