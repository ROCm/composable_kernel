// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/utility/e8m0.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using namespace ck;

TEST(E8M0, DefaultConstructor)
{
    e8m0_bexp_t exp;
    EXPECT_EQ(exp.data, 0);
}

TEST(E8M0, InitConstructor)
{
    e8m0_bexp_t exp(0x7F);
    EXPECT_EQ(exp.data, 0x7F);
}

TEST(E8M0, FloatConstructor)
{
    e8m0_bexp_t exp(1.0f);
    EXPECT_EQ(exp.data, 0x7F);
}

TEST(E8M0, FloatConstructorNaN)
{
    e8m0_bexp_t exp(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(exp.data, 0xFF);
}

TEST(E8M0, FloatConstructorZero)
{
    e8m0_bexp_t exp(0.0f);
    EXPECT_EQ(exp.data, 0);
}

TEST(E8M0, ConversionToFloat)
{
    e8m0_bexp_t exp(0x7F);
    float value = type_convert<float>(exp);
    EXPECT_EQ(value, 1.0f);
}

TEST(E8M0, ConversionToFloatNaN)
{
    e8m0_bexp_t exp(0xFF);
    float value = type_convert<float>(exp);
    EXPECT_TRUE(std::isnan(value));
}

TEST(E8M0, MinValue)
{
    e8m0_bexp_t exp(0);
    EXPECT_TRUE(exp == ck::NumericLimits<e8m0_bexp_t>::Min());

    float value = type_convert<float>(exp);
    EXPECT_EQ(value, std::powf(2, -ck::NumericUtils<e8m0_bexp_t>::bias));
}

TEST(E8M0, MaxValue)
{
    e8m0_bexp_t exp(254);
    EXPECT_TRUE(exp == ck::NumericLimits<e8m0_bexp_t>::Max());

    float value = type_convert<float>(exp);
    EXPECT_EQ(value,
              std::powf(2,
                        ck::NumericLimits<e8m0_bexp_t>::Max().data -
                            ck::NumericUtils<e8m0_bexp_t>::bias));
}

TEST(E8M0, EqualityOperator)
{
    e8m0_bexp_t exp1(0x7F);
    e8m0_bexp_t exp2(0x7F);
    EXPECT_TRUE(exp1 == exp2);
}

TEST(E8M0, InequalityOperator)
{
    e8m0_bexp_t exp1(0x7F);
    e8m0_bexp_t exp2(0x80);
    EXPECT_FALSE(exp1 == exp2);
}

TEST(E8M0, EqualityOperatorNaN)
{
    e8m0_bexp_t exp1(0xFF);
    e8m0_bexp_t exp2(0xFF);
    EXPECT_FALSE(exp1 == exp2);
}

TEST(E8M0, GetExponentValue)
{
    e8m0_bexp_t exp(0x7F);
    int value = ck::utils::get_exponent_value(exp);
    EXPECT_EQ(value, 0x7F);
}

// Round-trip conversion tests
TEST(E8M0, RoundTripPowersOfTwo)
{
    // E8M0 represents exact powers of two
    for(int e = -126; e <= 127; ++e)
    {
        float input = std::powf(2.0f, static_cast<float>(e));
        e8m0_bexp_t exp(input);
        float output = type_convert<float>(exp);
        EXPECT_EQ(output, input) << "Failed for 2^" << e;
    }
}

TEST(E8M0, FloatToE8M0Truncation)
{
    // E8M0 only stores exponent, mantissa is lost
    // 1.5 = 2^0 * 1.5, stored as 2^0 = 1.0
    e8m0_bexp_t exp1(1.5f);
    EXPECT_EQ(type_convert<float>(exp1), 1.0f);

    // 3.0 = 2^1 * 1.5, stored as 2^1 = 2.0
    e8m0_bexp_t exp2(3.0f);
    EXPECT_EQ(type_convert<float>(exp2), 2.0f);

    // 6.0 = 2^2 * 1.5, stored as 2^2 = 4.0
    e8m0_bexp_t exp3(6.0f);
    EXPECT_EQ(type_convert<float>(exp3), 4.0f);
}

// Edge case tests
TEST(E8M0, InfinityInput)
{
    e8m0_bexp_t exp(std::numeric_limits<float>::infinity());
    EXPECT_TRUE(exp.is_nan());
}

TEST(E8M0, NegativeInfinityInput)
{
    // Negative infinity extracts the exponent bits which are all 1s
    e8m0_bexp_t exp(-std::numeric_limits<float>::infinity());
    EXPECT_TRUE(exp.is_nan());
}

TEST(E8M0, IsNaN)
{
    e8m0_bexp_t nan_exp(0xFF);
    EXPECT_TRUE(nan_exp.is_nan());

    e8m0_bexp_t normal_exp(0x7F);
    EXPECT_FALSE(normal_exp.is_nan());

    e8m0_bexp_t zero_exp(0x00);
    EXPECT_FALSE(zero_exp.is_nan());
}

TEST(E8M0, SpecificExponentValues)
{
    // Test specific exponent values
    // 0x7F = 127 (bias) => 2^0 = 1.0
    EXPECT_EQ(type_convert<float>(e8m0_bexp_t(0x7F)), 1.0f);

    // 0x80 = 128 => 2^1 = 2.0
    EXPECT_EQ(type_convert<float>(e8m0_bexp_t(0x80)), 2.0f);

    // 0x7E = 126 => 2^-1 = 0.5
    EXPECT_EQ(type_convert<float>(e8m0_bexp_t(0x7E)), 0.5f);

    // 0x81 = 129 => 2^2 = 4.0
    EXPECT_EQ(type_convert<float>(e8m0_bexp_t(0x81)), 4.0f);

    // 0x87 = 135 => 2^8 = 256.0
    EXPECT_EQ(type_convert<float>(e8m0_bexp_t(0x87)), 256.0f);
}

TEST(E8M0, ExtremeExponents)
{
    // Test extreme exponent values
    // 0x01 = 1 => 2^-126
    float min_normal = std::powf(2.0f, -126.0f);
    EXPECT_EQ(type_convert<float>(e8m0_bexp_t(0x01)), min_normal);

    // 0xFE = 254 => 2^127
    float max_normal = std::powf(2.0f, 127.0f);
    EXPECT_EQ(type_convert<float>(e8m0_bexp_t(0xFE)), max_normal);
}

TEST(E8M0, BiasValue)
{
    // Verify bias is 127
    EXPECT_EQ(e8m0_bexp_t::bias, 127);
}

TEST(E8M0, IntConstructor)
{
    e8m0_bexp_t exp1(127);
    EXPECT_EQ(exp1.data, 127);

    e8m0_bexp_t exp2(255);
    EXPECT_EQ(exp2.data, 255);

    // Test masking behavior
    e8m0_bexp_t exp3(256); // Should be masked to 0
    EXPECT_EQ(exp3.data, 0);
}

TEST(E8M0, UInt32Constructor)
{
    e8m0_bexp_t exp1(uint32_t(127));
    EXPECT_EQ(exp1.data, 127);

    e8m0_bexp_t exp2(uint32_t(0x100)); // Should be masked
    EXPECT_EQ(exp2.data, 0);
}

TEST(E8M0, ZeroExponentSpecialCase)
{
    // Exponent 0 represents denormal/zero in float
    e8m0_bexp_t exp(0x00);
    float value = type_convert<float>(exp);
    // Zero exponent should produce 2^-127
    EXPECT_EQ(value, std::powf(2.0f, -127.0f));
}
