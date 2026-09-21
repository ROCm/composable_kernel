// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/datatype.hpp>

#include <gtest/gtest.h>

#include <string>

using ::rocm_ck::DataType;
using ::rocm_ck::dataTypeBits;
using ::rocm_ck::dataTypeName;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;

namespace {

// ============================================================================
// Parameterized: one row per DataType variant
// ============================================================================

struct DataTypeEntry
{
    DataType dt;
    int bits;
    const char* name;
};

class DataTypeTest : public TestWithParam<DataTypeEntry>
{
};

TEST_P(DataTypeTest, ReportsCorrectBits)
{
    EXPECT_EQ(dataTypeBits(GetParam().dt), GetParam().bits);
}

TEST_P(DataTypeTest, MapsToExpectedName)
{
    EXPECT_STREQ(dataTypeName(GetParam().dt), GetParam().name);
}

INSTANTIATE_TEST_SUITE_P(
    AllTypes,
    DataTypeTest,
    Values(DataTypeEntry{.dt = DataType::FP64, .bits = 64, .name = "FP64"},
           DataTypeEntry{.dt = DataType::FP32, .bits = 32, .name = "FP32"},
           DataTypeEntry{.dt = DataType::FP16, .bits = 16, .name = "FP16"},
           DataTypeEntry{.dt = DataType::BF16, .bits = 16, .name = "BF16"},
           DataTypeEntry{.dt = DataType::FP8_FNUZ, .bits = 8, .name = "FP8_FNUZ"},
           DataTypeEntry{.dt = DataType::BF8_FNUZ, .bits = 8, .name = "BF8_FNUZ"},
           DataTypeEntry{.dt = DataType::FP8_OCP, .bits = 8, .name = "FP8_OCP"},
           DataTypeEntry{.dt = DataType::BF8_OCP, .bits = 8, .name = "BF8_OCP"},
           DataTypeEntry{.dt = DataType::I4, .bits = 4, .name = "I4"},
           DataTypeEntry{.dt = DataType::I8, .bits = 8, .name = "I8"},
           DataTypeEntry{.dt = DataType::I16, .bits = 16, .name = "I16"},
           DataTypeEntry{.dt = DataType::I32, .bits = 32, .name = "I32"},
           DataTypeEntry{.dt = DataType::I64, .bits = 64, .name = "I64"},
           DataTypeEntry{.dt = DataType::U8, .bits = 8, .name = "U8"},
           DataTypeEntry{.dt = DataType::U16, .bits = 16, .name = "U16"},
           DataTypeEntry{.dt = DataType::U32, .bits = 32, .name = "U32"},
           DataTypeEntry{.dt = DataType::U64, .bits = 64, .name = "U64"}),
    [](const TestParamInfo<DataTypeEntry>& p) { return std::string(p.param.name); });

// ============================================================================
// constexpr validation
// ============================================================================

TEST(DataType, EvaluatesBitsAndNameAtCompileTime)
{
    constexpr int fp32_bits = dataTypeBits(DataType::FP32);
    EXPECT_EQ(fp32_bits, 32);

    constexpr const char* fp32_name = dataTypeName(DataType::FP32);
    EXPECT_STREQ(fp32_name, "FP32");
}

} // namespace
