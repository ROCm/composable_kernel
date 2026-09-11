// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/fixed_string.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::FixedString;

namespace {

TEST(FixedString, MatchesSingleCharacter)
{
    EXPECT_TRUE(FixedString<16>("A") == "A");
    EXPECT_FALSE(FixedString<16>("A") == "B");
}

TEST(FixedString, MatchesExactStringOnly)
{
    EXPECT_TRUE(FixedString<16>("bias") == "bias");
    EXPECT_FALSE(FixedString<16>("bias") == "bia");
    EXPECT_FALSE(FixedString<16>("bias") == "biases");
}

TEST(FixedString, AcceptsMaxCapacityMinusOne)
{
    EXPECT_TRUE(FixedString<16>("123456789012345") == "123456789012345");
}

TEST(FixedString, SupportsEmptyString)
{
    EXPECT_EQ(FixedString<16>("").len, 0);
    EXPECT_TRUE(FixedString<16>("") == "");
    EXPECT_FALSE(FixedString<16>("") == "A");
}

TEST(FixedString, EqualStringsCompareEqual)
{
    EXPECT_EQ(FixedString<16>("A"), FixedString<16>("A"));
    EXPECT_NE(FixedString<16>("A"), FixedString<16>("B"));
}

TEST(FixedString, OrderingIsLexicographic)
{
    EXPECT_LT(FixedString<16>("A"), FixedString<16>("B"));
    EXPECT_LT(FixedString<16>("B"), FixedString<16>("Z"));
    EXPECT_GT(FixedString<16>("Z"), FixedString<16>("A"));
}

} // namespace
