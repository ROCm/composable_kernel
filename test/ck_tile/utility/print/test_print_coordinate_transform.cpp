// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/utility/print.hpp"

namespace ck_tile {

class PrintCoordinateTransformTest : public ::testing::Test
{
    protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PrintCoordinateTransformTest, PrintPassThrough)
{
    // Test printing pass_through transform
    auto pt = make_pass_through_transform(number<32>{});

    testing::internal::CaptureStdout();
    print(pt);
    std::string output = testing::internal::GetCapturedStdout();

    // Verify it contains the pass_through identifier and some structure
    EXPECT_TRUE(output.find("pass_through{") == 0);
    EXPECT_TRUE(output.find("up_lengths_") != std::string::npos);
    EXPECT_TRUE(output.back() == '}');
}

TEST_F(PrintCoordinateTransformTest, PrintEmbed)
{
    // Test printing embed transform
    auto embed_transform = make_embed_transform(make_tuple(number<4>{}, number<8>{}),
                                                make_tuple(number<1>{}, number<4>{}));

    testing::internal::CaptureStdout();
    print(embed_transform);
    std::string output = testing::internal::GetCapturedStdout();

    // Verify it contains the embed identifier and key fields
    EXPECT_TRUE(output.find("embed{") == 0);
    EXPECT_TRUE(output.find("up_lengths_") != std::string::npos);
    EXPECT_TRUE(output.find("coefficients_") != std::string::npos);
    EXPECT_TRUE(output.back() == '}');
}

TEST_F(PrintCoordinateTransformTest, PrintMerge)
{
    // Test printing merge transform
    auto merge_transform = make_merge_transform(make_tuple(number<4>{}, number<8>{}));

    testing::internal::CaptureStdout();
    print(merge_transform);
    std::string output = testing::internal::GetCapturedStdout();

    // Verify it contains merge identifier and key fields
    EXPECT_TRUE(output.find("merge") ==
                0); // Could be merge_v2_magic_division or merge_v3_division_mod
    EXPECT_TRUE(output.find("low_lengths_") != std::string::npos ||
                output.find("up_lengths_") != std::string::npos);
    EXPECT_TRUE(output.back() == '}');
}

TEST_F(PrintCoordinateTransformTest, PrintUnmerge)
{
    // Test printing unmerge transform
    auto unmerge_transform = make_unmerge_transform(make_tuple(number<4>{}, number<8>{}));

    testing::internal::CaptureStdout();
    print(unmerge_transform);
    std::string output = testing::internal::GetCapturedStdout();

    // Verify it contains the unmerge identifier and key fields
    EXPECT_TRUE(output.find("unmerge{") == 0);
    EXPECT_TRUE(output.find("up_lengths_") != std::string::npos);
    EXPECT_TRUE(output.back() == '}');
}

TEST_F(PrintCoordinateTransformTest, PrintFreeze)
{
    // Test printing freeze transform
    auto freeze_transform = make_freeze_transform(number<5>{});

    testing::internal::CaptureStdout();
    print(freeze_transform);
    std::string output = testing::internal::GetCapturedStdout();

    // Verify it contains the freeze identifier and key fields
    EXPECT_TRUE(output.find("freeze{") == 0);
    EXPECT_TRUE(output.find("low_idx_") != std::string::npos);
    EXPECT_TRUE(output.back() == '}');
}

} // namespace ck_tile
