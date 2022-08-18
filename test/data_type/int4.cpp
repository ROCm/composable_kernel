// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"

#include "ck/utility/data_type.hpp"
#include "ck/utility/math_v2.hpp"

using ck::int4_t;

TEST(Int4, BaseArithmetic)
{
  int4_t a{1};
  int4_t b{-2};
  EXPECT_EQ(a+a, int4_t{2});
  EXPECT_EQ(a-a, int4_t{0});
  EXPECT_EQ(a+b, int4_t{-1});
  EXPECT_EQ(a-b, int4_t{3});
  EXPECT_EQ(a*a, int4_t{1});
  EXPECT_EQ(a*b, int4_t{-2});
  EXPECT_EQ(b*b, int4_t{4});
  EXPECT_EQ(a/b, int4_t{0});
  a = int4_t{4};
  EXPECT_EQ(a/b, int4_t{-2});
  b = int4_t{2};
  EXPECT_EQ(a%b, int4_t{0});
}

TEST(Int4, NumericLimits)
{
  EXPECT_EQ(ck::NumericLimits<int4_t>::Min(), int4_t{-7});
  EXPECT_EQ(ck::NumericLimits<int4_t>::Max(), int4_t{7});
  EXPECT_EQ(ck::NumericLimits<int4_t>::Lowest(), int4_t{-7});
}

TEST(Int4, MathOpsV2)
{
  int4_t a{4};
  int4_t b{-5};

  EXPECT_EQ(ck::math::abs(a), int4_t{4});
  EXPECT_EQ(ck::math::abs(b), int4_t{5});
  EXPECT_FALSE(ck::math::isnan(b));
}
