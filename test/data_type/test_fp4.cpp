// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using ck::f4_convert_rne;
using ck::f4_convert_sr;
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

TEST(FP4, ConvertFP32Nearest)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum fp4 value
    float max_fp4 = 6.0f;
    // convert 0 float to fp4 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f4_convert_rne(0.0f)), abs_tol);
    // convert maximal f4_t to float and check if equal to 6.0
    ASSERT_NEAR(max_fp4, type_convert<float>(f4_convert_rne(max_fp4)), abs_tol);
    // convert maximal float to fp4 and back, check if clipped to 6.0
    ASSERT_NEAR(
        max_fp4, type_convert<float>(f4_convert_rne(std::numeric_limits<float>::max())), abs_tol);
    // positive norm float value to fp4 and back, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float, type_convert<float>(f4_convert_rne(pos_float)), abs_tol);
    // negative norm float value to fp4 and back, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(f4_convert_rne(neg_float)), abs_tol);
    // positive subnorm float value to fp4 and back, check if holds
    pos_float = 0.5f;
    ASSERT_NEAR(pos_float, type_convert<float>(f4_convert_rne(pos_float)), abs_tol);
    // negative subnorm float value to fp4 and back, check if holds
    neg_float = -0.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(f4_convert_rne(neg_float)), abs_tol);
}

TEST(FP4, ConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum fp4 value
    float max_fp4 = 6.0f;
    // convert 0 float to fp4 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f4_convert_sr(0.0f)), abs_tol);
    // convert maximal f4_t to float and check if equal to 6.0
    ASSERT_NEAR(max_fp4, type_convert<float>(f4_convert_sr(max_fp4)), abs_tol);
    // convert maximal float to fp4 and back, check if clipped to 6.0
    ASSERT_NEAR(
        max_fp4, type_convert<float>(f4_convert_sr(std::numeric_limits<float>::max())), abs_tol);
    // positive norm float value to fp4 and back, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float, type_convert<float>(f4_convert_sr(pos_float)), abs_tol);
    // negative norm float value to fp4 and back, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(f4_convert_sr(neg_float)), abs_tol);
    // positive subnorm float value to fp4 and back, check if holds
    pos_float = 0.5f;
    ASSERT_NEAR(pos_float, type_convert<float>(f4_convert_sr(pos_float)), abs_tol);
    // negative subnorm float value to fp4 and back, check if holds
    neg_float = -0.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(f4_convert_sr(neg_float)), abs_tol);
}
