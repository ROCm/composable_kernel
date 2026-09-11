#include <gtest/gtest.h>
#include "ck/utility/e5m3.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using namespace ck;

TEST(E5M3, DefaultConstructor)
{
    e5m3_scale_t scale;
    EXPECT_EQ(scale.data, 0);
}

TEST(E5M3, InitConstructor)
{
    e5m3_scale_t scale(0x7F);
    EXPECT_EQ(scale.data, 0x7F);
}

TEST(E5M3, FloatConstructor)
{
    e5m3_scale_t scale(1.0f);
    EXPECT_EQ(scale.data, 0x78);
}

TEST(E5M3, FloatConstructorNaN)
{
    e5m3_scale_t scale(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(scale.data, 0xFF);
}

TEST(E5M3, FloatConstructorZero)
{
    e5m3_scale_t scale(0.0f);
    EXPECT_EQ(scale.data, 0);
}

TEST(E5M3, ConversionToFloat)
{
    e5m3_scale_t scale(0x78);
    float value = float(scale);
    EXPECT_EQ(value, 1.0f);
}

TEST(E5M3, ConversionToFloatNaN)
{
    e5m3_scale_t scale(0xFF);
    float value = float(scale);
    EXPECT_TRUE(std::isnan(value));
}

TEST(E5M3, MinValue)
{
    e5m3_scale_t scale(0x01);
    EXPECT_TRUE(scale == ck::NumericLimits<e5m3_scale_t>::Min());

    float value = float(scale);
    EXPECT_EQ(value, std::powf(2, -17));
}

TEST(E5M3, MaxValue)
{
    e5m3_scale_t scale(0xFE);
    EXPECT_TRUE(scale == ck::NumericLimits<e5m3_scale_t>::Max());

    float value = float(scale);
    EXPECT_EQ(value, 114688.0f);
}

TEST(E5M3, EqualityOperator)
{
    e5m3_scale_t scale1(0x08);
    e5m3_scale_t scale2(0x08);
    EXPECT_TRUE(scale1 == scale2);
}

TEST(E5M3, InequalityOperator)
{
    e5m3_scale_t scale1(0x08);
    e5m3_scale_t scale2(0x09);
    EXPECT_FALSE(scale1 == scale2);
}

TEST(E5M3, EqualityOperatorNaN)
{
    e5m3_scale_t scale1(0xFF);
    e5m3_scale_t scale2(0xFF);
    EXPECT_FALSE(scale1 == scale2);
}

TEST(E5M3, GetExponentValue)
{
    e5m3_scale_t scale(0xF8);
    int value = ck::utils::get_exponent_value(scale);
    EXPECT_EQ(value, 0x1F);
}

// Round-trip conversion tests
TEST(E5M3, RoundTripPowersOfTwo)
{
    // Test powers of two that are exactly representable
    for(int exp = -14; exp <= 15; ++exp)
    {
        float input = std::powf(2.0f, static_cast<float>(exp));
        e5m3_scale_t scale(input);
        float output = float(scale);
        EXPECT_EQ(output, input) << "Failed for 2^" << exp;
    }
}

TEST(E5M3, RoundTripSmallValues)
{
    // Test small representable values
    std::vector<float> values = {0.0625f, 0.125f, 0.25f, 0.5f};
    for(float v : values)
    {
        e5m3_scale_t scale(v);
        float output = float(scale);
        EXPECT_EQ(output, v) << "Failed for " << v;
    }
}

// Rounding tests (round-to-nearest-even)
TEST(E5M3, RoundingToNearestEven)
{
    // Test values that should round to even
    // 1.0625 = 1 + 1/16, rounds to 1.0 (even)
    e5m3_scale_t scale1(1.0625f);
    EXPECT_EQ(float(scale1), 1.0f);

    // 1.1875 = 1 + 3/16, rounds to 1.25
    e5m3_scale_t scale2(1.1875f);
    EXPECT_EQ(float(scale2), 1.25f);
}

TEST(E5M3, RoundingUp)
{
    // 1.15 should round up to 1.25
    e5m3_scale_t scale(1.2f);
    float output = float(scale);
    EXPECT_EQ(output, 1.25f);
}

TEST(E5M3, RoundingDown)
{
    // 1.05 should round down to 1.0
    e5m3_scale_t scale(1.05f);
    float output = float(scale);
    EXPECT_EQ(output, 1.0f);
}

// Edge case tests
TEST(E5M3, NegativeInput)
{
    e5m3_scale_t scale(-1.0f);
    EXPECT_TRUE(scale.is_nan());
}

TEST(E5M3, InfinityInput)
{
    e5m3_scale_t scale(std::numeric_limits<float>::infinity());
    EXPECT_TRUE(scale.is_nan());
}

TEST(E5M3, NegativeInfinityInput)
{
    e5m3_scale_t scale(-std::numeric_limits<float>::infinity());
    EXPECT_TRUE(scale.is_nan());
}

TEST(E5M3, OverflowClampsToMax)
{
    // Values larger than max should clamp to max_finite
    e5m3_scale_t scale(200000.0f);
    EXPECT_EQ(scale.data, e5m3_scale_t::max_finite);
    EXPECT_EQ(float(scale), 114688.0f);
}

TEST(E5M3, UnderflowFlushesToZero)
{
    // Very small values should flush to zero
    e5m3_scale_t scale(1e-20f);
    EXPECT_EQ(scale.data, 0);
}

TEST(E5M3, DenormalValues)
{
    // Test denormal representation (exponent = 0)
    // Smallest denormal: 2^-17
    e5m3_scale_t scale1(std::powf(2.0f, -17.0f));
    EXPECT_EQ(scale1.data, 0x01);

    // Second smallest denormal: 2^-16
    e5m3_scale_t scale2(std::powf(2.0f, -16.0f));
    EXPECT_EQ(scale2.data, 0x02);
}

TEST(E5M3, SmallestNormal)
{
    // Smallest normal value: 2^-14 (exponent = 1, mantissa = 0)
    e5m3_scale_t scale(std::powf(2.0f, -14.0f));
    EXPECT_EQ(scale.data, 0x08);
}

TEST(E5M3, IsNaN)
{
    e5m3_scale_t nan_scale(0xFF);
    EXPECT_TRUE(nan_scale.is_nan());

    e5m3_scale_t normal_scale(0x78);
    EXPECT_FALSE(normal_scale.is_nan());
}

TEST(E5M3, SpecificBitPatterns)
{
    // Test specific bit patterns and their float values
    // 0x78 = exp=15, mant=0 => 2^(15-15) * 1.0 = 1.0
    EXPECT_EQ(float(e5m3_scale_t(0x78)), 1.0f);

    // 0x80 = exp=16, mant=0 => 2^(16-15) * 1.0 = 2.0
    EXPECT_EQ(float(e5m3_scale_t(0x80)), 2.0f);

    // 0x70 = exp=14, mant=0 => 2^(14-15) * 1.0 = 0.5
    EXPECT_EQ(float(e5m3_scale_t(0x70)), 0.5f);

    // 0x7C = exp=15, mant=4 => 2^(15-15) * 1.5 = 1.5
    EXPECT_EQ(float(e5m3_scale_t(0x7C)), 1.5f);
}

TEST(E5M3, LargerDynamicRange)
{
    // E5M3 has larger dynamic range than E4M3
    // Test values that E4M3 cannot represent
    e5m3_scale_t scale1(std::powf(2.0f, 10.0f)); // 1024
    EXPECT_EQ(float(scale1), 1024.0f);

    e5m3_scale_t scale2(std::powf(2.0f, -10.0f)); // 1/1024
    EXPECT_EQ(float(scale2), std::powf(2.0f, -10.0f));
}
