// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include "ck_tile/core/algorithm/static_encoding_pattern.hpp"
#include "ck_tile/core/utility/print.hpp"

namespace ck_tile {

class PrintStaticEncodingPatternTest : public ::testing::Test
{
    protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PrintStaticEncodingPatternTest, PrintThreadRakedPattern)
{
    // Test printing thread raked pattern
    using PatternType = TileDistributionEncodingPattern2D<64, 8, 16, 4, tile_distribution_pattern::thread_raked>;
    PatternType pattern;

    testing::internal::CaptureStdout();
    print(pattern);
    std::string output = testing::internal::GetCapturedStdout();

    // Verify the output contains expected information
    EXPECT_TRUE(output.find("TileDistributionEncodingPattern2D") != std::string::npos);
    EXPECT_TRUE(output.find("BlockSize:64") != std::string::npos);
    EXPECT_TRUE(output.find("YPerTile:8") != std::string::npos);
    EXPECT_TRUE(output.find("XPerTile:16") != std::string::npos);
    EXPECT_TRUE(output.find("VecSize:4") != std::string::npos);
    EXPECT_TRUE(output.find("thread_raked") != std::string::npos);
}

TEST_F(PrintStaticEncodingPatternTest, PrintWarpRakedPattern)
{
    // Test printing warp raked pattern
    using PatternType = TileDistributionEncodingPattern2D<128, 16, 32, 8, tile_distribution_pattern::warp_raked>;
    PatternType pattern;

    testing::internal::CaptureStdout();
    print(pattern);
    std::string output = testing::internal::GetCapturedStdout();

    // Verify the output contains expected information
    EXPECT_TRUE(output.find("TileDistributionEncodingPattern2D") != std::string::npos);
    EXPECT_TRUE(output.find("BlockSize:128") != std::string::npos);
    EXPECT_TRUE(output.find("YPerTile:16") != std::string::npos);
    EXPECT_TRUE(output.find("XPerTile:32") != std::string::npos);
    EXPECT_TRUE(output.find("VecSize:8") != std::string::npos);
    EXPECT_TRUE(output.find("warp_raked") != std::string::npos);
}

TEST_F(PrintStaticEncodingPatternTest, PrintBlockRakedPattern)
{
    // Test printing block raked pattern
    using PatternType = TileDistributionEncodingPattern2D<256, 32, 64, 16, tile_distribution_pattern::block_raked>;
    PatternType pattern;

    testing::internal::CaptureStdout();
    print(pattern);
    std::string output = testing::internal::GetCapturedStdout();

    // Verify the output contains expected information
    EXPECT_TRUE(output.find("TileDistributionEncodingPattern2D") != std::string::npos);
    EXPECT_TRUE(output.find("BlockSize:256") != std::string::npos);
    EXPECT_TRUE(output.find("YPerTile:32") != std::string::npos);
    EXPECT_TRUE(output.find("XPerTile:64") != std::string::npos);
    EXPECT_TRUE(output.find("VecSize:16") != std::string::npos);
    EXPECT_TRUE(output.find("block_raked") != std::string::npos);
}

} // namespace ck_tile
