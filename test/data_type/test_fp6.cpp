// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/scaled_type_convert.hpp"

using ck::e8m0_bexp_t;
using ck::f6_t;
using ck::Number;
using ck::scaled_type_convert;
using ck::type_convert;
using ck::vector_type;

TEST(FP6, NumericLimits)
{
    EXPECT_EQ(ck::NumericLimits<f6_t>::Min(), f6_t(0b001000));
    EXPECT_EQ(ck::NumericLimits<f6_t>::Max(), f6_t(0b011111));
    EXPECT_EQ(ck::NumericLimits<f6_t>::Lowest(), f6_t(0b111111));
    EXPECT_EQ(ck::NumericLimits<f6_t>::MinSubnorm(), f6_t(0b000001));
    EXPECT_EQ(ck::NumericLimits<f6_t>::MaxSubnorm(), f6_t(0b000111));
}
