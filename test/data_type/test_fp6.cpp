// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/scaled_type_convert.hpp"

using ck::e8m0_bexp_t;
using ck::f6_convert_rne;
using ck::f6_convert_sr;
using ck::f6_t;
using ck::f6x16_pk_t;
using ck::f6x32_pk_t;
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

TEST(FP6, ConvertFP32Nearest)
{
    // set maximum fp6 value
    float max_fp6 = 7.5f;
    // convert 0 float to fp6 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f6_convert_rne(0.0f)), 0.0f);
    // convert maximal f6_t to float and check if equal to max_fp6
    ASSERT_NEAR(max_fp6, type_convert<float>(f6_convert_rne(max_fp6)), 0.0f);
    // convert maximal float to fp6 and back, check if clipped to max_fp6
    ASSERT_NEAR(
        max_fp6, type_convert<float>(f6_convert_rne(std::numeric_limits<float>::max())), 0.0f);
    // convert float Inf to fp6 and back, check if clipped to max_fp6
    ASSERT_NEAR(
        max_fp6, type_convert<float>(f6_convert_rne(std::numeric_limits<float>::infinity())), 0.0f);
    // convert float value less than fp6 subnorm to fp6 and back, check if equal to 0.0
    float less_than_subnorm = 0.0625f;
    ASSERT_NEAR(0.0f, type_convert<float>(f6_convert_rne(less_than_subnorm)), 0.0f);
    // convert float NaN to fp6 and back, check if clipped to max_fp6
    ASSERT_NEAR(max_fp6,
                type_convert<float>(f6_convert_rne(std::numeric_limits<float>::quiet_NaN())),
                0.0f);
    // positive norm float value to fp6 and back, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float, type_convert<float>(f6_convert_rne(pos_float)), 0.0f);
    // negative norm float value to fp6 and back, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(f6_convert_rne(neg_float)), 0.0f);
    // positive subnorm float value to fp6 and back, check if holds
    pos_float = 0.125f;
    ASSERT_NEAR(pos_float, type_convert<float>(f6_convert_rne(pos_float)), 0.0f);
    // negative subnorm float value to fp6 and back, check if holds
    neg_float = -0.25f;
    ASSERT_NEAR(neg_float, type_convert<float>(f6_convert_rne(neg_float)), 0.0f);
}

TEST(FP6, ConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum fp6 value
    float max_fp6 = 7.5f;
    // convert 0 float to fp6 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f6_convert_sr(0.0f)), abs_tol);
    // convert maximal f6_t to float and check if equal to max_fp6
    ASSERT_NEAR(max_fp6, type_convert<float>(f6_convert_sr(max_fp6)), abs_tol);
    // convert maximal float to fp6 and back, check if clipped to max_fp6
    ASSERT_NEAR(
        max_fp6, type_convert<float>(f6_convert_sr(std::numeric_limits<float>::max())), abs_tol);
    // convert float Inf to fp6 and back, check if clipped to max_fp6
    ASSERT_NEAR(max_fp6,
                type_convert<float>(f6_convert_sr(std::numeric_limits<float>::infinity())),
                abs_tol);
    // convert float NaN to fp6 and back, check if clipped to max_fp6
    ASSERT_NEAR(max_fp6,
                type_convert<float>(f6_convert_sr(std::numeric_limits<float>::quiet_NaN())),
                abs_tol);
    // positive norm float value to fp6 and back, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float, type_convert<float>(f6_convert_sr(pos_float)), abs_tol);
    // negative norm float value to fp6 and back, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(f6_convert_sr(neg_float)), abs_tol);
    // positive subnorm float value to fp6 and back, check if holds
    pos_float = 0.125f;
    ASSERT_NEAR(pos_float, type_convert<float>(f6_convert_sr(pos_float)), abs_tol);
    // negative subnorm float value to fp6 and back, check if holds
    neg_float = -0.25f;
    ASSERT_NEAR(neg_float, type_convert<float>(f6_convert_sr(neg_float)), abs_tol);
}

TEST(FP6, ScaledConvertFP32Nearest)
{
    // set maximum scale
    float max_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Max()); // 0xFE -> float
    // set minimum scale
    float min_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Min()); // 0x00 -> float
    // set arbitrary scale to 256.0
    float test_scale = 256.0f; // 0b10000111
    // convert 0 float to fp6 and back with maximal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_rne(0.0f)), 0.0f);
    // convert 0 float to fp6 and back with minimal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_rne(0.0f)), 0.0f);
    // positive norm float value to fp6 and back with various scales, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_rne(pos_float)),
                0.0f);
    // negative norm float value to fp6 and back with various scales, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_rne(neg_float)),
                0.0f);
    // positive subnorm float value to fp6 and back with various scales, check if holds
    pos_float = 0.125f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_rne(pos_float)),
                0.0f);
    // negative subnorm float value to fp6 and back with various scales, check if holds
    neg_float = -0.25f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_rne(neg_float)),
                0.0f);
}

TEST(FP6, ScaledConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum scale
    float max_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Max()); // 0xFE -> float
    // set minimum scale
    float min_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Min()); // 0x00 -> float
    // set arbitrary scale to 256.0
    float test_scale = 256.0f; // 0b10000111
    // convert 0 float to fp6 and back with maximal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_sr(0.0f)), abs_tol);
    // convert 0 float to fp6 and back with minimal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_sr(0.0f)), abs_tol);
    // positive norm float value to fp6 and back with various scales, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_sr(pos_float)),
                abs_tol);
    // negative norm float value to fp6 and back with various scales, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_sr(neg_float)),
                abs_tol);
    // positive subnorm float value to fp6 and back with various scales, check if holds
    pos_float = 0.125f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_sr(pos_float)),
                abs_tol);
    // negative subnorm float value to fp6 and back with various scales, check if holds
    neg_float = -0.25f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_sr(neg_float)),
                abs_tol);
}

TEST(FP6, TestSize)
{
    ASSERT_EQ(1, sizeof(f6_t));
    ASSERT_EQ(12, sizeof(f6x16_pk_t));
    ASSERT_EQ(24, sizeof(f6x32_pk_t));
    ASSERT_EQ(16, sizeof(vector_type<f6x16_pk_t, 1>));
    ASSERT_EQ(32, sizeof(vector_type<f6x16_pk_t, 2>));
    ASSERT_EQ(32, sizeof(vector_type<f6x32_pk_t, 1>));
}

TEST(FP6, TestAlignment)
{
    ASSERT_EQ(1, alignof(f6_t));
    ASSERT_EQ(4, alignof(f6x16_pk_t));
    ASSERT_EQ(4, alignof(f6x32_pk_t));
    ASSERT_EQ(16, alignof(vector_type<f6x16_pk_t, 1>));
    ASSERT_EQ(32, alignof(vector_type<f6x16_pk_t, 2>));
    ASSERT_EQ(32, alignof(vector_type<f6x32_pk_t, 1>));
}

// test vector of 1 f6x16_pk_t, contains 16 f6_t
TEST(FP6, TestAsType16x1)
{
    // test size
    const int vector_size = 1;
    const int packed_size = 16;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(16)));
    test_vec_t test_vec = {f6_t(0b000000),
                           f6_t(0b100000),
                           f6_t(0b000001),
                           f6_t(0b100001),
                           f6_t(0b000010),
                           f6_t(0b100010),
                           f6_t(0b000011),
                           f6_t(0b100011),
                           f6_t(0b000100),
                           f6_t(0b100100),
                           f6_t(0b000101),
                           f6_t(0b100101),
                           f6_t(0b000110),
                           f6_t(0b100110),
                           f6_t(0b001011),
                           f6_t(0b101011)};
    // reference vector
    vector_type<f6x16_pk_t, vector_size> right_vec;
    // check default CTOR
    ck::static_for<0, packed_size, 1>{}([&](auto i) {
        ASSERT_EQ(
            right_vec.template AsType<f6x16_pk_t>()(Number<0>{}).template unpack<>(Number<i>{}), 0);
    });
    // assign test values to the vector
    ck::static_for<0, vector_size, 1>{}([&](auto i) {
        right_vec.template AsType<f6x16_pk_t>()(Number<i>{}) = f6x16_pk_t{}.pack(test_vec);
    });
    // copy the vector
    vector_type<f6x16_pk_t, vector_size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, packed_size, 1>{}([&](auto i) {
        ASSERT_EQ(
            left_vec.template AsType<f6x16_pk_t>()(Number<0>{}).template unpack<>(Number<i>{}),
            static_cast<f6_t>(test_vec[static_cast<int>(i)]));
    });
}

// test vector of 2 f6x16_pk_t, contains 32 f6_t
TEST(FP6, TestAsType16x2)
{
    // test size
    const int vector_size = 2;
    const int packed_size = 16;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(16)));
    test_vec_t test_vec[2];
    test_vec[0] = {f6_t(0b000000),
                   f6_t(0b100000),
                   f6_t(0b000001),
                   f6_t(0b100001),
                   f6_t(0b000010),
                   f6_t(0b100010),
                   f6_t(0b000011),
                   f6_t(0b100011),
                   f6_t(0b000100),
                   f6_t(0b100100),
                   f6_t(0b000101),
                   f6_t(0b100101),
                   f6_t(0b000110),
                   f6_t(0b100110),
                   f6_t(0b001011),
                   f6_t(0b101011)};
    test_vec[1] = {f6_t(0b010000),
                   f6_t(0b110000),
                   f6_t(0b010001),
                   f6_t(0b110001),
                   f6_t(0b010010),
                   f6_t(0b110010),
                   f6_t(0b010011),
                   f6_t(0b110011),
                   f6_t(0b010100),
                   f6_t(0b110100),
                   f6_t(0b010101),
                   f6_t(0b110101),
                   f6_t(0b010110),
                   f6_t(0b110110),
                   f6_t(0b011011),
                   f6_t(0b111011)};
    // reference vector
    vector_type<f6x16_pk_t, vector_size> right_vec;
    // check default CTOR
    ck::static_for<0, vector_size, 1>{}([&](auto idx_vector) {
        ck::static_for<0, packed_size, 1>{}([&](auto idx_element) {
            ASSERT_EQ(right_vec.template AsType<f6x16_pk_t>()(Number<idx_vector>{})
                          .template unpack<>(Number<idx_element>{}),
                      0);
        });
    });
    // assign test values to the vector
    ck::static_for<0, vector_size, 1>{}([&](auto i) {
        right_vec.template AsType<f6x16_pk_t>()(Number<i>{}) = f6x16_pk_t{}.pack(test_vec[i]);
    });
    // copy the vector
    vector_type<f6x16_pk_t, vector_size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, vector_size, 1>{}([&](auto idx_vector) {
        ck::static_for<0, packed_size, 1>{}([&](auto idx_element) {
            ASSERT_EQ(left_vec.template AsType<f6x16_pk_t>()(Number<idx_vector>{})
                          .template unpack<>(Number<idx_element>{}),
                      static_cast<f6_t>(test_vec[idx_vector][static_cast<int>(idx_element)]));
        });
    });
}

// test vector of 1 f6x32_pk_t, contains 32 f6_t
TEST(FP6, TestAsType32x1)
{
    // test size
    const int vector_size = 1;
    const int packed_size = 32;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(32)));
    test_vec_t test_vec = {f6_t(0b000000), f6_t(0b100000), f6_t(0b000001), f6_t(0b100001),
                           f6_t(0b000010), f6_t(0b100010), f6_t(0b000011), f6_t(0b100011),
                           f6_t(0b000100), f6_t(0b100100), f6_t(0b000101), f6_t(0b100101),
                           f6_t(0b000110), f6_t(0b100110), f6_t(0b001011), f6_t(0b101011),
                           f6_t(0b010000), f6_t(0b110000), f6_t(0b010001), f6_t(0b110001),
                           f6_t(0b010010), f6_t(0b110010), f6_t(0b010011), f6_t(0b110011),
                           f6_t(0b010100), f6_t(0b110100), f6_t(0b010101), f6_t(0b110101),
                           f6_t(0b010110), f6_t(0b110110), f6_t(0b011011), f6_t(0b111011)};
    // reference vector
    vector_type<f6x32_pk_t, vector_size> right_vec;
    // check default CTOR
    ck::static_for<0, packed_size, 1>{}([&](auto i) {
        ASSERT_EQ(
            right_vec.template AsType<f6x32_pk_t>()(Number<0>{}).template unpack<>(Number<i>{}), 0);
    });
    // assign test values to the vector
    ck::static_for<0, vector_size, 1>{}([&](auto i) {
        right_vec.template AsType<f6x32_pk_t>()(Number<i>{}) = f6x32_pk_t{}.pack(test_vec);
    });
    // copy the vector
    vector_type<f6x32_pk_t, vector_size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, packed_size, 1>{}([&](auto i) {
        ASSERT_EQ(
            left_vec.template AsType<f6x32_pk_t>()(Number<0>{}).template unpack<>(Number<i>{}),
            static_cast<f6_t>(test_vec[static_cast<int>(i)]));
    });
}
