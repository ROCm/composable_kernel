// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using ck::e8m0_scale_t;
using ck::f4_convert_rne;
using ck::f4_convert_sr;
using ck::f4_t;
using ck::scaled_type_convert;
using ck::type_convert;

using ck::utils::cast_from_float;
using ck::utils::cast_to_float;

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

TEST(FP4, ScaledConvertFP32Nearest)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum fp4 value
    float max_fp4 = 6.0f;
    // set maximum scale
    float max_scale = std::pow(2,
                               ck::NumericLimits<e8m0_scale_t>::Max() -
                                   ck::NumericUtils<e8m0_scale_t>::bias); // 0xFE -> float
    // set minimum scale
    float min_scale = std::pow(2, -ck::NumericUtils<e8m0_scale_t>::bias); // 0x00 -> float
    // set arbitrary scale to 256.0
    float test_scale = 256.0f; // 0b10000111
    // convert 0 float to fp4 and back with maximal scale, check if holds
    ASSERT_NEAR(0.0f,
                scaled_type_convert<float>(cast_from_float(max_scale), f4_convert_rne(0.0f)),
                abs_tol);
    // convert 0 float to fp4 and back with minimal scale, check if holds
    ASSERT_NEAR(0.0f,
                scaled_type_convert<float>(cast_from_float(min_scale), f4_convert_rne(0.0f)),
                abs_tol);
    // convert maximal f4_t with minimal scale to float and check if equal to minimal float
    ASSERT_NEAR(ck::NumericLimits<float>::Min(),
                scaled_type_convert<float>(cast_from_float(min_scale), f4_convert_rne(max_fp4)),
                abs_tol);
    // positive norm float value to fp4 and back with various scales, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(cast_from_float(test_scale), f4_convert_rne(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(cast_from_float(max_scale), f4_convert_rne(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(cast_from_float(min_scale), f4_convert_rne(pos_float)),
                abs_tol);
    // negative norm float value to fp4 and back with various scales, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(cast_from_float(test_scale), f4_convert_rne(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(cast_from_float(max_scale), f4_convert_rne(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(cast_from_float(min_scale), f4_convert_rne(neg_float)),
                abs_tol);
    // positive subnorm float value to fp4 and back with various scales, check if holds
    pos_float = 0.5f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(cast_from_float(test_scale), f4_convert_rne(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(cast_from_float(max_scale), f4_convert_rne(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(cast_from_float(min_scale), f4_convert_rne(pos_float)),
                abs_tol);
    // negative subnorm float value to fp4 and back with various scales, check if holds
    neg_float = -0.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(cast_from_float(test_scale), f4_convert_rne(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(cast_from_float(max_scale), f4_convert_rne(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(cast_from_float(min_scale), f4_convert_rne(neg_float)),
                abs_tol);
}

TEST(FP4, ScaledConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum fp4 value
    float max_fp4 = 6.0f;
    // set maximum scale
    float max_scale = std::pow(2,
                               ck::NumericLimits<e8m0_scale_t>::Max() -
                                   ck::NumericUtils<e8m0_scale_t>::bias); // 0xFE -> float
    // set minimum scale
    float min_scale = std::pow(2, -ck::NumericUtils<e8m0_scale_t>::bias); // 0x00 -> float
    // set arbitrary scale to 256.0
    float test_scale = 256.0f; // 0b10000111
    // convert 0 float to fp4 and back with maximal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(cast_from_float(max_scale), f4_convert_sr(0.0f)), abs_tol);
    // convert 0 float to fp4 and back with minimal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(cast_from_float(min_scale), f4_convert_sr(0.0f)), abs_tol);
    // convert maximal f4_t with minimal scale to float and check if equal to minimal float
    ASSERT_NEAR(ck::NumericLimits<float>::Min(),
                scaled_type_convert<float>(cast_from_float(min_scale), f4_convert_sr(max_fp4)),
                abs_tol);
    // positive norm float value to fp4 and back with various scales, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(cast_from_float(test_scale), f4_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(cast_from_float(max_scale), f4_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(cast_from_float(min_scale), f4_convert_sr(pos_float)),
                abs_tol);
    // negative norm float value to fp4 and back with various scales, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(cast_from_float(test_scale), f4_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(cast_from_float(max_scale), f4_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(cast_from_float(min_scale), f4_convert_sr(neg_float)),
                abs_tol);
    // positive subnorm float value to fp4 and back with various scales, check if holds
    pos_float = 0.5f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(cast_from_float(test_scale), f4_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(cast_from_float(max_scale), f4_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(cast_from_float(min_scale), f4_convert_sr(pos_float)),
                abs_tol);
    // negative subnorm float value to fp4 and back with various scales, check if holds
    neg_float = -0.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(cast_from_float(test_scale), f4_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(cast_from_float(max_scale), f4_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(cast_from_float(min_scale), f4_convert_sr(neg_float)),
                abs_tol);
}
