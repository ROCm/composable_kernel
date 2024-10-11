// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using ck::f8_convert_rne;
using ck::f8_convert_sr;
using ck::f8_ocp_t;
using ck::half_t;
using ck::type_convert;

TEST(FP8OCP, NumericLimits)
{
    // constants given for OCP FP8
    EXPECT_EQ(ck::NumericLimits<f8_ocp_t>::Min(),
              type_convert<f8_ocp_t>(0x08)); // 0b00001000 = 2^-6
    EXPECT_EQ(ck::NumericLimits<f8_ocp_t>::Max(), type_convert<f8_ocp_t>(0x7E)); // 0b01111110 = 448
    EXPECT_EQ(ck::NumericLimits<f8_ocp_t>::Lowest(),
              type_convert<f8_ocp_t>(0xFE)); // 0b11111110 = -448
    EXPECT_EQ(ck::NumericLimits<f8_ocp_t>::QuietNaN().data,
              type_convert<f8_ocp_t>(0x7F).data); // 0b01111111
    EXPECT_FALSE(ck::NumericLimits<f8_ocp_t>::QuietNaN() ==
                 ck::NumericLimits<f8_ocp_t>::QuietNaN());
}

TEST(FP8OCP, ConvertFP32Nearest)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // convert 0 float to fp8 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f8_convert_rne<f8_ocp_t>(0.0f)), 0.0f);

    // convert minimal float to fp8 and back, check if holds
    ASSERT_NEAR(std::numeric_limits<float>::min(),
                type_convert<float>(f8_convert_rne<f8_ocp_t>(std::numeric_limits<float>::min())),
                abs_tol);

    const auto max_f8_t_float = type_convert<float>(ck::NumericLimits<f8_ocp_t>::Max());

    // convert maximal f8_ocp_t to float and check if equal to fp8 max
    ASSERT_NEAR(
        max_f8_t_float, type_convert<float>(f8_convert_rne<f8_ocp_t>(max_f8_t_float)), 0.0f);

    // convert maximal float to fp8 and back, check if clipped to fp8 max (saturation to finite)
    ASSERT_NEAR(max_f8_t_float,
                type_convert<float>(f8_convert_rne<f8_ocp_t>(std::numeric_limits<float>::max())),
                0.0f);

    // convert float infinity to f8_ocp_t and check if it is max value (saturation to finite)
    ASSERT_EQ(ck::NumericLimits<f8_ocp_t>::Max(),
              f8_convert_rne<f8_ocp_t>(std::numeric_limits<float>::infinity()));

    // positive norm float value to fp8 and back, check if holds
    float pos_float = 0.017578125f;
    ASSERT_NEAR(pos_float, type_convert<float>(f8_convert_rne<f8_ocp_t>(pos_float)), abs_tol);

    // smallest normal fp8 value to fp8 and back, check if holds
    float neg_float = -0.015625f; //-2^-6
    ASSERT_NEAR(neg_float, type_convert<float>(f8_convert_rne<f8_ocp_t>(neg_float)), 0.0f);

    // positive subnorm float value to fp8 and back, check if holds
    pos_float = 0.00390625f;
    ASSERT_NEAR(pos_float, type_convert<float>(f8_convert_rne<f8_ocp_t>(pos_float)), abs_tol);

    // min subnorm fp8 value to fp8 and back, check if holds
    neg_float = -0.001953125f; //-2^-9
    ASSERT_NEAR(neg_float, type_convert<float>(f8_convert_rne<f8_ocp_t>(neg_float)), 0.0f);

    // smaller than min subnorm fp8 value to fp8 must be zero
    auto less_than_min_subnorm = 0.0009765625f; // 2^-10
    ASSERT_EQ(0.0f, type_convert<float>(f8_convert_rne<f8_ocp_t>(less_than_min_subnorm)));

    // convert quiet NaN to f8_ocp_t and check if it is quiet NaN
    auto f8_nan = f8_convert_rne<f8_ocp_t>(std::numeric_limits<float>::quiet_NaN());
    ASSERT_TRUE((f8_nan.data & 0x7f) == 0x7f);
}

TEST(FP8OCP, ConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // convert 0 float to fp8 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f8_convert_sr<f8_ocp_t>(0.0f)), 0.0f);

    // convert minimal float to fp8 and back, check if holds
    ASSERT_NEAR(std::numeric_limits<float>::min(),
                type_convert<float>(f8_convert_sr<f8_ocp_t>(std::numeric_limits<float>::min())),
                abs_tol);

    const auto max_f8_t_float = type_convert<float>(ck::NumericLimits<f8_ocp_t>::Max());

    // convert maximal f8_ocp_t to float and check if equal to fp8 max
    ASSERT_NEAR(max_f8_t_float, type_convert<float>(f8_convert_sr<f8_ocp_t>(max_f8_t_float)), 0.0f);

    // convert maximal float to fp8 and back, check if clipped to fp8 max (saturation to finite)
    ASSERT_NEAR(max_f8_t_float,
                type_convert<float>(f8_convert_sr<f8_ocp_t>(std::numeric_limits<float>::max())),
                0.0f);

    // convert float infinity to f8_ocp_t and check if it is max value (saturation to finite)
    ASSERT_EQ(ck::NumericLimits<f8_ocp_t>::Max(),
              f8_convert_sr<f8_ocp_t>(std::numeric_limits<float>::infinity()));

    // positive norm float value to fp8 and back, check if holds
    float pos_float = 0.017578125f;
    ASSERT_NEAR(pos_float, type_convert<float>(f8_convert_sr<f8_ocp_t>(pos_float)), abs_tol);

    // smallest normal fp8 value to fp8 and back, check if holds
    float neg_float = -0.015625f; //-2^-6
    ASSERT_NEAR(neg_float, type_convert<float>(f8_convert_sr<f8_ocp_t>(neg_float)), 0.0f);

    // positive subnorm float value to fp8 and back, check if holds
    pos_float = 0.00390625f;
    ASSERT_NEAR(pos_float, type_convert<float>(f8_convert_sr<f8_ocp_t>(pos_float)), abs_tol);

    // min subnorm fp8 value to fp8 and back, check if holds
    constexpr auto min_subnorm_fp8 = -0.001953125f; //-2^-9
    ASSERT_NEAR(
        min_subnorm_fp8, type_convert<float>(f8_convert_sr<f8_ocp_t>(min_subnorm_fp8)), 0.0f);

    // smaller than min subnorm fp8 value to fp8 alternates between 0 and 2^-9
    auto less_than_min_subnorm = 0.0009765625f; // 2^-10
    ASSERT_NEAR(
        0.0f, type_convert<float>(f8_convert_sr<f8_ocp_t>(less_than_min_subnorm)), 0.001953125f);

    // convert quiet NaN to f8_ocp_t and check if it is quiet NaN
    auto f8_nan = f8_convert_sr<f8_ocp_t>(std::numeric_limits<float>::quiet_NaN());
    ASSERT_TRUE((f8_nan.data & 0x7f) == 0x7f);
}

TEST(FP8OCP, ConvertFP16Nearest) {}

TEST(FP8OCP, ConvertFP16Stochastic) {}
