// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/scaled_type_convert.hpp"

using ck::bf6_convert_rne;
using ck::bf6_convert_sr;
using ck::bf6_t;
using ck::e8m0_bexp_t;
using ck::Number;
using ck::scaled_type_convert;
using ck::type_convert;
using ck::vector_type;

TEST(BF6, NumericLimits)
{
    EXPECT_EQ(ck::NumericLimits<bf6_t>::Min(), bf6_t(0b001000));
    EXPECT_EQ(ck::NumericLimits<bf6_t>::Max(), bf6_t(0b011111));
    EXPECT_EQ(ck::NumericLimits<bf6_t>::Lowest(), bf6_t(0b111111));
    EXPECT_EQ(ck::NumericLimits<bf6_t>::MinSubnorm(), bf6_t(0b000001));
    EXPECT_EQ(ck::NumericLimits<bf6_t>::MaxSubnorm(), bf6_t(0b000011));
}

TEST(BF6, ConvertFP32Nearest)
{
    // set maximum bf6 value
    float max_bf6 = 28.0f;
    // convert 0 float to bf6 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(bf6_convert_rne(0.0f)), 0.0f);
    // convert max_bf6 to float and check if equal to max_bf6
    ASSERT_NEAR(max_bf6, type_convert<float>(bf6_convert_rne(max_bf6)), 0.0f);
    // convert maximal float to bf6 and back, check if clipped to max_bf6
    ASSERT_NEAR(
        max_bf6, type_convert<float>(bf6_convert_rne(std::numeric_limits<float>::max())), 0.0f);
    // convert float Inf to bf6 and back, check if clipped to max_bf6
    ASSERT_NEAR(max_bf6,
                type_convert<float>(bf6_convert_rne(std::numeric_limits<float>::infinity())),
                0.0f);
    // convert float value less than bf6 subnorm to bf6 and back, check if equal to 0.0
    float less_than_subnorm = 0.03125f;
    ASSERT_NEAR(0.0f, type_convert<float>(bf6_convert_rne(less_than_subnorm)), 0.0f);
    // convert float NaN to bf6 and back, check if clipped to max_bf6
    ASSERT_NEAR(max_bf6,
                type_convert<float>(bf6_convert_rne(std::numeric_limits<float>::quiet_NaN())),
                0.0f);
    // positive norm float value to bf6 and back, check if holds
    float pos_float = 0.25f;
    ASSERT_NEAR(pos_float, type_convert<float>(bf6_convert_rne(pos_float)), 0.0f);
    // negative norm float value to bf6 and back, check if holds
    float neg_float = -0.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(bf6_convert_rne(neg_float)), 0.0f);
    // positive subnorm float value to bf6 and back, check if holds
    pos_float = 0.1875f;
    ASSERT_NEAR(pos_float, type_convert<float>(bf6_convert_rne(pos_float)), 0.0f);
    // negative subnorm float value to bf6 and back, check if holds
    neg_float = -0.0625f;
    ASSERT_NEAR(neg_float, type_convert<float>(bf6_convert_rne(neg_float)), 0.0f);
}

TEST(BF6, ConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum bf6 value
    float max_bf6 = 28.0f;
    // convert 0 float to bf6 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(bf6_convert_sr(0.0f)), abs_tol);
    // convert maximal bf6_t to float and check if equal to max_bf6
    ASSERT_NEAR(max_bf6, type_convert<float>(bf6_convert_sr(max_bf6)), abs_tol);
    // convert maximal float to bf6 and back, check if clipped to max_bf6
    ASSERT_NEAR(
        max_bf6, type_convert<float>(bf6_convert_sr(std::numeric_limits<float>::max())), abs_tol);
    // convert float Inf to bf6 and back, check if clipped to max_bf6
    ASSERT_NEAR(max_bf6,
                type_convert<float>(bf6_convert_rne(std::numeric_limits<float>::infinity())),
                0.0f);
    // convert float NaN to bf6 and back, check if clipped to max_bf6
    ASSERT_NEAR(max_bf6,
                type_convert<float>(bf6_convert_rne(std::numeric_limits<float>::quiet_NaN())),
                0.0f);
    // positive norm float value to bf6 and back, check if holds
    float pos_float = 0.25f;
    ASSERT_NEAR(pos_float, type_convert<float>(bf6_convert_sr(pos_float)), abs_tol);
    // negative norm float value to bf6 and back, check if holds
    float neg_float = -0.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(bf6_convert_sr(neg_float)), abs_tol);
    // positive subnorm float value to bf6 and back, check if holds
    pos_float = 0.1875f;
    ASSERT_NEAR(pos_float, type_convert<float>(bf6_convert_sr(pos_float)), abs_tol);
    // negative subnorm float value to bf6 and back, check if holds
    neg_float = -0.0625f;
    ASSERT_NEAR(neg_float, type_convert<float>(bf6_convert_sr(neg_float)), abs_tol);
}

TEST(BF6, ScaledConvertFP32Nearest)
{
    // set maximum scale
    float max_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Max()); // 0xFE -> float
    // set minimum scale
    float min_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Min()); // 0x00 -> float
    // set arbitrary scale to 256.0
    float test_scale = 256.0f; // 0b10000111
    // convert 0 float to bf6 and back with maximal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_rne(0.0f)), 0.0f);
    // convert 0 float to bf6 and back with minimal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_rne(0.0f)), 0.0f);
    // positive norm float value to bf6 and back with various scales, check if holds
    float pos_float = 0.25f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_rne(pos_float)),
                0.0f);
    // negative norm float value to bf6 and back with various scales, check if holds
    float neg_float = -0.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_rne(neg_float)),
                0.0f);
    // positive subnorm float value to bf6 and back with various scales, check if holds
    pos_float = 0.1875f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_rne(pos_float)),
                0.0f);
    // negative subnorm float value to bf6 and back with various scales, check if holds
    neg_float = -0.0625f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_rne(neg_float)),
                0.0f);
}

TEST(BF6, ScaledConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum scale
    float max_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Max()); // 0xFE -> float
    // set minimum scale
    float min_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Min()); // 0x00 -> float
    // set arbitrary scale to 256.0
    float test_scale = 256.0f; // 0b10000111
    // convert 0 float to bf6 and back with maximal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_sr(0.0f)), abs_tol);
    // convert 0 float to bf6 and back with minimal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_sr(0.0f)), abs_tol);
    // positive norm float value to bf6 and back with various scales, check if holds
    float pos_float = 0.25f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_sr(pos_float)),
                abs_tol);
    // negative norm float value to bf6 and back with various scales, check if holds
    float neg_float = -0.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_sr(neg_float)),
                abs_tol);
    // positive subnorm float value to bf6 and back with various scales, check if holds
    pos_float = 0.1875f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_sr(pos_float)),
                abs_tol);
    // negative subnorm float value to bf6 and back with various scales, check if holds
    neg_float = -0.0625f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_sr(neg_float)),
                abs_tol);
}
