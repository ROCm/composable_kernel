#include <gtest/gtest.h>
#include "ck/utility/e4m3.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using namespace ck;

TEST(E4M3, DefaultConstructor)
{
    e4m3_scale_t scale;
    EXPECT_EQ(scale.data, 0);
}

TEST(E4M3, InitConstructor)
{
    e4m3_scale_t scale(0x7F);
    EXPECT_EQ(scale.data, 0x7F);
}

TEST(E4M3, FloatConstructor)
{
    e4m3_scale_t scale(1.0f);
    EXPECT_EQ(scale.data, 0x38);
}

TEST(E4M3, FloatConstructorNaN)
{
    e4m3_scale_t scale(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(scale.data, 0x7F);
}

TEST(E4M3, FloatConstructorZero)
{
    e4m3_scale_t scale(0.0f);
    EXPECT_EQ(scale.data, 0);
}

TEST(E4M3, ConversionToFloat)
{
    e4m3_scale_t scale(0x40);
    float value = float(scale);
    EXPECT_EQ(value, 2.0f);
}

TEST(E4M3, ConversionToFloatNaN)
{
    e4m3_scale_t scale(0x7F);
    float value = float(scale);
    EXPECT_TRUE(std::isnan(value));
}

TEST(E4M3, MinValue)
{
    e4m3_scale_t scale(0x01);
    EXPECT_TRUE(scale == ck::NumericLimits<e4m3_scale_t>::Min());

    float value = float(scale);
    EXPECT_EQ(value, std::powf(2, -9));
}

TEST(E4M3, MaxValue)
{
    e4m3_scale_t scale(0x7E);
    EXPECT_TRUE(scale == ck::NumericLimits<e4m3_scale_t>::Max());

    float value = float(scale);
    EXPECT_EQ(value, 448.0f);
}

TEST(E4M3, EqualityOperator)
{
    e4m3_scale_t scale1(0x08);
    e4m3_scale_t scale2(0x08);
    EXPECT_TRUE(scale1 == scale2);
}

TEST(E4M3, InequalityOperator)
{
    e4m3_scale_t scale1(0x08);
    e4m3_scale_t scale2(0x09);
    EXPECT_FALSE(scale1 == scale2);
}

TEST(E4M3, EqualityOperatorNaN)
{
    e4m3_scale_t scale1(0xFF);
    e4m3_scale_t scale2(0xFF);
    EXPECT_FALSE(scale1 == scale2);
}

TEST(E4M3, GetExponentValue)
{
    e4m3_scale_t scale(0x78);
    int value = ck::utils::get_exponent_value(scale);
    EXPECT_EQ(value, 0x0F);
}

// Round-trip conversion tests
TEST(E4M3, RoundTripPowersOfTwo)
{
    // Test powers of two that are exactly representable
    for(int exp = -6; exp <= 8; ++exp)
    {
        float input = std::powf(2.0f, static_cast<float>(exp));
        e4m3_scale_t scale(input);
        float output = float(scale);
        EXPECT_EQ(output, input) << "Failed for 2^" << exp;
    }
}

TEST(E4M3, RoundTripSmallValues)
{
    // Test small representable values
    std::vector<float> values = {0.0625f, 0.125f, 0.25f, 0.5f};
    for(float v : values)
    {
        e4m3_scale_t scale(v);
        float output = float(scale);
        EXPECT_EQ(output, v) << "Failed for " << v;
    }
}

// Rounding tests (round-to-nearest-even)
TEST(E4M3, RoundingToNearestEven)
{
    // Test values that should round to even
    // 1.0625 = 1 + 1/16, rounds to 1.0 (even)
    e4m3_scale_t scale1(1.0625f);
    EXPECT_EQ(float(scale1), 1.0f);

    // 1.1875 = 1 + 3/16, rounds to 1.25
    e4m3_scale_t scale2(1.1875f);
    EXPECT_EQ(float(scale2), 1.25f);
}

TEST(E4M3, RoundingUp)
{
    // 1.15 should round up to 1.25
    e4m3_scale_t scale(1.2f);
    float output = float(scale);
    EXPECT_EQ(output, 1.25f);
}

TEST(E4M3, RoundingDown)
{
    // 1.05 should round down to 1.0
    e4m3_scale_t scale(1.05f);
    float output = float(scale);
    EXPECT_EQ(output, 1.0f);
}

// Edge case tests
TEST(E4M3, NegativeInput)
{
    e4m3_scale_t scale(-1.0f);
    EXPECT_TRUE(scale.is_nan());
}

TEST(E4M3, InfinityInput)
{
    e4m3_scale_t scale(std::numeric_limits<float>::infinity());
    EXPECT_TRUE(scale.is_nan());
}

TEST(E4M3, NegativeInfinityInput)
{
    e4m3_scale_t scale(-std::numeric_limits<float>::infinity());
    EXPECT_TRUE(scale.is_nan());
}

TEST(E4M3, OverflowClampsToMax)
{
    // Values larger than max should clamp to max_finite
    e4m3_scale_t scale(1000.0f);
    EXPECT_EQ(scale.data, e4m3_scale_t::max_finite);
    EXPECT_EQ(float(scale), 448.0f);
}

TEST(E4M3, UnderflowFlushesToZero)
{
    // Very small values should flush to zero
    e4m3_scale_t scale(1e-10f);
    EXPECT_EQ(scale.data, 0);
}

TEST(E4M3, DenormalValues)
{
    // Test denormal representation (exponent = 0)
    // Smallest denormal: 2^-9
    e4m3_scale_t scale1(std::powf(2.0f, -9.0f));
    EXPECT_EQ(scale1.data, 0x01);

    // Second smallest denormal: 2^-8
    e4m3_scale_t scale2(std::powf(2.0f, -8.0f));
    EXPECT_EQ(scale2.data, 0x02);
}

TEST(E4M3, SmallestNormal)
{
    // Smallest normal value: 2^-6 (exponent = 1, mantissa = 0)
    e4m3_scale_t scale(std::powf(2.0f, -6.0f));
    EXPECT_EQ(scale.data, 0x08);
}

TEST(E4M3, IsNaN)
{
    e4m3_scale_t nan_scale(0x7F);
    EXPECT_TRUE(nan_scale.is_nan());

    e4m3_scale_t normal_scale(0x38);
    EXPECT_FALSE(normal_scale.is_nan());
}

TEST(E4M3, SpecificBitPatterns)
{
    // Test specific bit patterns and their float values
    // 0x38 = exp=7, mant=0 => 2^(7-7) * 1.0 = 1.0
    EXPECT_EQ(float(e4m3_scale_t(0x38)), 1.0f);

    // 0x40 = exp=8, mant=0 => 2^(8-7) * 1.0 = 2.0
    EXPECT_EQ(float(e4m3_scale_t(0x40)), 2.0f);

    // 0x30 = exp=6, mant=0 => 2^(6-7) * 1.0 = 0.5
    EXPECT_EQ(float(e4m3_scale_t(0x30)), 0.5f);

    // 0x3C = exp=7, mant=4 => 2^(7-7) * 1.5 = 1.5
    EXPECT_EQ(float(e4m3_scale_t(0x3C)), 1.5f);
}
