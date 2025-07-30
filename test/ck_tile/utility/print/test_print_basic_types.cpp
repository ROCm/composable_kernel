// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include "ck_tile/core/utility/print.hpp"

namespace ck_tile {

class PrintBasicTypesTest : public ::testing::Test
{
    protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PrintBasicTypesTest, PrintIntArray)
{
    int arr[4] = {1, 2, 3, 4};

    testing::internal::CaptureStdout();
    print(arr);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(output, "[1, 2, 3, 4]");
}

TEST_F(PrintBasicTypesTest, PrintFloatArray)
{
    float arr[3] = {1.5f, 2.5f, 3.5f};

    testing::internal::CaptureStdout();
    print(arr);
    std::string output = testing::internal::GetCapturedStdout();

    // Note: floating point formatting may vary, so we check for key elements
    EXPECT_TRUE(output.find("[") == 0);
    EXPECT_TRUE(output.find("1.5") != std::string::npos);
    EXPECT_TRUE(output.find("2.5") != std::string::npos);
    EXPECT_TRUE(output.find("3.5") != std::string::npos);
    EXPECT_TRUE(output.back() == ']');
    EXPECT_TRUE(output.find(", ") != std::string::npos);
}

TEST_F(PrintBasicTypesTest, PrintDoubleArray)
{
    double arr[2] = {10.123, 20.456};

    testing::internal::CaptureStdout();
    print(arr);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("[") == 0);
    EXPECT_TRUE(output.find("10.123") != std::string::npos);
    EXPECT_TRUE(output.find("20.456") != std::string::npos);
    EXPECT_TRUE(output.back() == ']');
}

TEST_F(PrintBasicTypesTest, PrintUnsignedIntArray)
{
    unsigned int arr[3] = {100u, 200u, 300u};

    testing::internal::CaptureStdout();
    print(arr);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(output, "[100, 200, 300]");
}

TEST_F(PrintBasicTypesTest, PrintCharArray)
{
    char arr[5] = {'a', 'b', 'c', 'd', 'e'};

    testing::internal::CaptureStdout();
    print(arr);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(output, "[a, b, c, d, e]");
}

TEST_F(PrintBasicTypesTest, PrintSingleElementArray)
{
    int arr[1] = {42};

    testing::internal::CaptureStdout();
    print(arr);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(output, "[42]");
}

} // namespace ck_tile
