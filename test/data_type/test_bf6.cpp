// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/scaled_type_convert.hpp"

using ck::bf6_t;
using ck::e8m0_bexp_t;
using ck::Number;
using ck::scaled_type_convert;
using ck::type_convert;
using ck::vector_type;

using ck::utils::cast_from_float;
using ck::utils::cast_to_float;

TEST(BF6, NumericLimits)
{
    EXPECT_EQ(ck::NumericLimits<bf6_t>::Min(), bf6_t(0b001000));
    EXPECT_EQ(ck::NumericLimits<bf6_t>::Max(), bf6_t(0b011111));
    EXPECT_EQ(ck::NumericLimits<bf6_t>::Lowest(), bf6_t(0b111111));
    EXPECT_EQ(ck::NumericLimits<bf6_t>::MinSubnorm(), bf6_t(0b000001));
    EXPECT_EQ(ck::NumericLimits<bf6_t>::MaxSubnorm(), bf6_t(0b000011));
}
