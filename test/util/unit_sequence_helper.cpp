// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/utility/sequence_helper.hpp"
#include "ck/utility/tuple_helper.hpp"

using namespace ck;

// Tests for generate_identity_sequences (PR #3588)
TEST(GenerateIdentitySequences, Size5)
{
    auto result = generate_identity_sequences(Number<5>{});
    auto expected =
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{});
    EXPECT_TRUE((is_same<decltype(result), decltype(expected)>::value));
}

TEST(GenerateIdentitySequences, Size1)
{
    auto result   = generate_identity_sequences(Number<1>{});
    auto expected = make_tuple(Sequence<0>{});
    EXPECT_TRUE((is_same<decltype(result), decltype(expected)>::value));
}

TEST(GenerateIdentitySequences, Size0)
{
    auto result   = generate_identity_sequences(Number<0>{});
    auto expected = make_tuple();
    EXPECT_TRUE((is_same<decltype(result), decltype(expected)>::value));
}

TEST(GenerateIdentitySequences, WithNumber)
{
    constexpr auto result = generate_identity_sequences(Number<3>{});
    EXPECT_EQ(result.Size(), 3);
    EXPECT_TRUE((is_same<decltype(result.At(Number<0>{})), const Sequence<0>&>::value));
    EXPECT_TRUE((is_same<decltype(result.At(Number<1>{})), const Sequence<1>&>::value));
    EXPECT_TRUE((is_same<decltype(result.At(Number<2>{})), const Sequence<2>&>::value));
}

// Tests for unpack_and_merge_sequences (PR #3589)
TEST(UnpackAndMergeSequences, MergeMultipleSequences)
{
    auto input    = make_tuple(Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5, 6>{});
    auto result   = unpack_and_merge_sequences(input);
    auto expected = Sequence<1, 2, 3, 4, 5, 6>{};
    EXPECT_TRUE((is_same<decltype(result), decltype(expected)>::value));
}

TEST(UnpackAndMergeSequences, SingleSequence)
{
    auto input    = make_tuple(Sequence<10, 20, 30>{});
    auto result   = unpack_and_merge_sequences(input);
    auto expected = Sequence<10, 20, 30>{};
    EXPECT_TRUE((is_same<decltype(result), decltype(expected)>::value));
}

TEST(UnpackAndMergeSequences, TwoSequences)
{
    auto input    = make_tuple(Sequence<100>{}, Sequence<200, 300>{});
    auto result   = unpack_and_merge_sequences(input);
    auto expected = Sequence<100, 200, 300>{};
    EXPECT_TRUE((is_same<decltype(result), decltype(expected)>::value));
}
