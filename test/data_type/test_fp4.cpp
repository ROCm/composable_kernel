// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using ck::f4_t;
using ck::type_convert;

TEST(FP8, NumericLimits)
{
    // constants given for negative zero nan mode
    EXPECT_EQ(ck::NumericLimits<f4_t>::Min(), f4_t{0x2});
    EXPECT_EQ(ck::NumericLimits<f4_t>::Max(), f4_t{0x7});
    EXPECT_EQ(ck::NumericLimits<f4_t>::Lowest(), f4_t{0xF});
    EXPECT_EQ(ck::NumericLimits<f4_t>::MinSubnorm(), f4_t{0x1});
    EXPECT_EQ(ck::NumericLimits<f4_t>::MaxSubnorm(), f4_t{0x1});
}
