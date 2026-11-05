// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <limits>
#include <gtest/gtest.h>
#include "ck/tensor_operation/gpu/element/moe_silu.hpp"

TEST(MoeSilu, ClampBeforeSilu)
{
    const float expected = 1.0f / (1.0f + std::exp(-1.0f));
    EXPECT_NEAR(ck::MoeSilu::Run(8.0f, 4.0f, 1.0f), expected, 1e-6f);
    // Post-SiLU clamping would produce 1, not SiLU(1).
    EXPECT_GT(std::abs(expected - 1.0f), 0.2f);
    EXPECT_NEAR(ck::MoeSilu::Run(8.0f, -4.0f, 1.0f), -expected, 1e-6f);
    EXPECT_NEAR(ck::MoeSilu::Run(-2.0f, 4.0f, 1.0f), -2.0f / (1.0f + std::exp(2.0f)), 1e-6f);
}

TEST(MoeSilu, DisabledAndNumericContract)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(ck::MoeSilu::ValidLimit(0.0f));
    EXPECT_TRUE(ck::MoeSilu::ValidLimit(1.0f));
    EXPECT_TRUE(ck::MoeSilu::ValidLimit(inf));
    EXPECT_FALSE(ck::MoeSilu::ValidLimit(-1.0f));
    EXPECT_FALSE(ck::MoeSilu::ValidLimit(-inf));
    EXPECT_FALSE(ck::MoeSilu::ValidLimit(nan));
    EXPECT_TRUE(ck::MoeSilu::ValidRuntimeLimit(1.0f));
    EXPECT_FALSE(ck::MoeSilu::ValidRuntimeLimit(0.0f));
    EXPECT_FALSE(ck::MoeSilu::ValidRuntimeLimit(inf));
    EXPECT_FALSE(ck::MoeSilu::ValidRuntimeLimit(-1.0f));
    EXPECT_FALSE(ck::MoeSilu::ValidRuntimeLimit(-inf));
    EXPECT_FALSE(ck::MoeSilu::ValidRuntimeLimit(nan));
}

TEST(MoeSilu, Swiglustep)
{
    EXPECT_FLOAT_EQ(ck::MoeSilu::Swiglustep(8.0f, 8.0f), 49.0f);
    EXPECT_FLOAT_EQ(ck::MoeSilu::Swiglustep(8.0f, -8.0f), -49.0f);
    EXPECT_NE(ck::MoeSilu::Run(8.0f, 8.0f, 7.0f), 49.0f);
}
