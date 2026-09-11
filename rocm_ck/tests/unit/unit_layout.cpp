// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/layout.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>

using ::rocm_ck::isValidLayoutForRank;
using ::rocm_ck::Layout;
using ::rocm_ck::layoutName;
using ::rocm_ck::layoutStrides;
using ::rocm_ck::leadingDimStride;
using ::testing::ElementsAre;

namespace {

// ============================================================================
// layoutName
// ============================================================================

TEST(Layout, MapsEnumValuesToExpectedStrings)
{
    EXPECT_STREQ(layoutName(Layout::Row), "Row");
    EXPECT_STREQ(layoutName(Layout::Col), "Col");
    EXPECT_STREQ(layoutName(Layout::Auto), "Auto");
}

// ============================================================================
// isValidLayoutForRank
// ============================================================================

TEST(Layout, AllowsRowAndColOnlyForRank2)
{
    EXPECT_FALSE(isValidLayoutForRank(Layout::Row, 1));
    EXPECT_TRUE(isValidLayoutForRank(Layout::Row, 2));
    EXPECT_FALSE(isValidLayoutForRank(Layout::Col, 1));
    EXPECT_TRUE(isValidLayoutForRank(Layout::Col, 2));
}

TEST(Layout, RejectsAutoForAllRanks)
{
    EXPECT_FALSE(isValidLayoutForRank(Layout::Auto, 0));
    EXPECT_FALSE(isValidLayoutForRank(Layout::Auto, 1));
    EXPECT_FALSE(isValidLayoutForRank(Layout::Auto, 2));
}

TEST(Layout, RejectsRowAndColForRankGreaterThan2)
{
    EXPECT_FALSE(isValidLayoutForRank(Layout::Row, 3));
    EXPECT_FALSE(isValidLayoutForRank(Layout::Row, 4));
    EXPECT_FALSE(isValidLayoutForRank(Layout::Row, 6));
    EXPECT_FALSE(isValidLayoutForRank(Layout::Col, 3));
    EXPECT_FALSE(isValidLayoutForRank(Layout::Col, 4));
    EXPECT_FALSE(isValidLayoutForRank(Layout::Col, 6));
}

// ============================================================================
// leadingDimStride
// ============================================================================

TEST(Layout, LeadingDimStrideReturnsFirstForRow)
{
    EXPECT_EQ(leadingDimStride(Layout::Row, std::array{128, 1}), 128);
}

TEST(Layout, LeadingDimStrideReturnsSecondForCol)
{
    EXPECT_EQ(leadingDimStride(Layout::Col, std::array{1, 64}), 64);
}

// ============================================================================
// layoutStrides
// ============================================================================

TEST(Layout, LayoutStridesRowMajor)
{
    EXPECT_THAT(layoutStrides(Layout::Row, 32, 64), ElementsAre(64, 1));
}

TEST(Layout, LayoutStridesColMajor)
{
    EXPECT_THAT(layoutStrides(Layout::Col, 32, 64), ElementsAre(1, 32));
}

} // namespace
