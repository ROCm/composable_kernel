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

// Tests for sequence_find_value (PR #3600)
TEST(SequenceFindValue, FindExistingElement)
{
    constexpr auto result = sequence_find_value<17>(Sequence<5, 11, 17, 23, 29>{});
    EXPECT_EQ(result, 2); // 17 is at index 2
}

TEST(SequenceFindValue, FindFirstElement)
{
    constexpr auto result = sequence_find_value<7>(Sequence<7, 13, 19, 31>{});
    EXPECT_EQ(result, 0);
}

TEST(SequenceFindValue, FindLastElement)
{
    constexpr auto result = sequence_find_value<41>(Sequence<3, 11, 23, 41>{});
    EXPECT_EQ(result, 3);
}

TEST(SequenceFindValue, ElementNotFound)
{
    constexpr auto result = sequence_find_value<50>(Sequence<2, 8, 14, 26>{});
    EXPECT_EQ(result, -1);
}

TEST(SequenceFindValue, EmptySequence)
{
    constexpr auto result = sequence_find_value<1>(Sequence<>{});
    EXPECT_EQ(result, -1);
}

// Tests for find_in_tuple_of_sequences (PR #3600)
TEST(FindInTupleOfSequences, FindInFirstSequence)
{
    constexpr auto tuple_of_seqs =
        make_tuple(Sequence<5, 11>{}, Sequence<17, 23>{}, Sequence<29, 37>{});
    constexpr auto result = find_in_tuple_of_sequences<11>(tuple_of_seqs);
    EXPECT_EQ(result.itran, 0);   // Found in first sequence (index 0)
    EXPECT_EQ(result.idim_up, 1); // At position 1 within that sequence
    EXPECT_TRUE(result.found);
}

TEST(FindInTupleOfSequences, FindInMiddleSequence)
{
    constexpr auto tuple_of_seqs =
        make_tuple(Sequence<2, 4, 6>{}, Sequence<8, 10>{}, Sequence<12>{});
    constexpr auto result = find_in_tuple_of_sequences<10>(tuple_of_seqs);
    EXPECT_EQ(result.itran, 1);   // Found in second sequence (index 1)
    EXPECT_EQ(result.idim_up, 1); // At position 1 within that sequence
    EXPECT_TRUE(result.found);
}

TEST(FindInTupleOfSequences, FindInLastSequence)
{
    constexpr auto tuple_of_seqs = make_tuple(Sequence<3>{}, Sequence<7>{}, Sequence<13, 19, 31>{});
    constexpr auto result        = find_in_tuple_of_sequences<31>(tuple_of_seqs);
    EXPECT_EQ(result.itran, 2);   // Found in third sequence (index 2)
    EXPECT_EQ(result.idim_up, 2); // At position 2 within that sequence
    EXPECT_TRUE(result.found);
}

TEST(FindInTupleOfSequences, NotFound)
{
    constexpr auto tuple_of_seqs = make_tuple(Sequence<1, 3>{}, Sequence<5, 7, 9>{});
    constexpr auto result        = find_in_tuple_of_sequences<100>(tuple_of_seqs);
    EXPECT_FALSE(result.found);
}

TEST(FindInTupleOfSequences, EmptyTuple)
{
    constexpr auto tuple_of_seqs = make_tuple();
    constexpr auto result        = find_in_tuple_of_sequences<1>(tuple_of_seqs);
    EXPECT_FALSE(result.found);
}

TEST(FindInTupleOfSequences, SingleSequence)
{
    constexpr auto tuple_of_seqs = make_tuple(Sequence<41, 43, 47, 53>{});
    constexpr auto result        = find_in_tuple_of_sequences<47>(tuple_of_seqs);
    EXPECT_EQ(result.itran, 0);
    EXPECT_EQ(result.idim_up, 2);
    EXPECT_TRUE(result.found);
}
