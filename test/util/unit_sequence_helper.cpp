// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/utility/sequence.hpp"
#include "ck/utility/sequence_helper.hpp"
#include "ck/utility/tuple_helper.hpp"

using namespace ck;

// Test generate_identity_sequences
TEST(GenerateIdentitySequences, Size5)
{
    constexpr auto result = generate_identity_sequences<5>();
    // Should produce Tuple<Sequence<0>, Sequence<1>, Sequence<2>, Sequence<3>, Sequence<4>>
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result[Number<0>{}])>, Sequence<0>>::value));
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result[Number<1>{}])>, Sequence<1>>::value));
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result[Number<2>{}])>, Sequence<2>>::value));
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result[Number<3>{}])>, Sequence<3>>::value));
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result[Number<4>{}])>, Sequence<4>>::value));
}

TEST(GenerateIdentitySequences, Size1)
{
    constexpr auto result = generate_identity_sequences<1>();
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result[Number<0>{}])>, Sequence<0>>::value));
    EXPECT_EQ(result.Size(), 1);
}

TEST(GenerateIdentitySequences, Size0)
{
    constexpr auto result = generate_identity_sequences<0>();
    EXPECT_EQ(result.Size(), 0);
}

TEST(GenerateIdentitySequences, WithNumber)
{
    constexpr auto result = generate_identity_sequences(Number<3>{});
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result[Number<0>{}])>, Sequence<0>>::value));
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result[Number<1>{}])>, Sequence<1>>::value));
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result[Number<2>{}])>, Sequence<2>>::value));
    EXPECT_EQ(result.Size(), 3);
}

// Test sequence_find_value
TEST(SequenceFindValue, FindExistingElement)
{
    constexpr auto result = sequence_find_value<3>(Sequence<1, 2, 3, 4, 5>{});
    EXPECT_EQ(result, 2); // 3 is at index 2
}

TEST(SequenceFindValue, FindFirstElement)
{
    constexpr auto result = sequence_find_value<10>(Sequence<10, 20, 30>{});
    EXPECT_EQ(result, 0);
}

TEST(SequenceFindValue, FindLastElement)
{
    constexpr auto result = sequence_find_value<30>(Sequence<10, 20, 30>{});
    EXPECT_EQ(result, 2);
}

TEST(SequenceFindValue, ElementNotFound)
{
    constexpr auto result = sequence_find_value<99>(Sequence<1, 2, 3, 4, 5>{});
    EXPECT_EQ(result, -1);
}

TEST(SequenceFindValue, EmptySequence)
{
    constexpr auto result = sequence_find_value<1>(Sequence<>{});
    EXPECT_EQ(result, -1);
}

// Test find_in_tuple_of_sequences
TEST(FindInTupleOfSequences, FindInFirstSequence)
{
    constexpr auto tuple_of_seqs = make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}, Sequence<4, 5>{});
    constexpr auto result        = find_in_tuple_of_sequences<1>(tuple_of_seqs);
    EXPECT_EQ(result.itran, 0);   // Found in first sequence (index 0)
    EXPECT_EQ(result.idim_up, 1); // At position 1 within that sequence
    EXPECT_TRUE(result.found);
}

TEST(FindInTupleOfSequences, FindInMiddleSequence)
{
    constexpr auto tuple_of_seqs = make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}, Sequence<4, 5>{});
    constexpr auto result        = find_in_tuple_of_sequences<3>(tuple_of_seqs);
    EXPECT_EQ(result.itran, 1);   // Found in second sequence (index 1)
    EXPECT_EQ(result.idim_up, 1); // At position 1 within that sequence
    EXPECT_TRUE(result.found);
}

TEST(FindInTupleOfSequences, FindInLastSequence)
{
    constexpr auto tuple_of_seqs = make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}, Sequence<4, 5>{});
    constexpr auto result        = find_in_tuple_of_sequences<5>(tuple_of_seqs);
    EXPECT_EQ(result.itran, 2);   // Found in third sequence (index 2)
    EXPECT_EQ(result.idim_up, 1); // At position 1 within that sequence
    EXPECT_TRUE(result.found);
}

TEST(FindInTupleOfSequences, NotFound)
{
    constexpr auto tuple_of_seqs = make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}, Sequence<4, 5>{});
    constexpr auto result        = find_in_tuple_of_sequences<99>(tuple_of_seqs);
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
    constexpr auto tuple_of_seqs = make_tuple(Sequence<10, 20, 30>{});
    constexpr auto result        = find_in_tuple_of_sequences<20>(tuple_of_seqs);
    EXPECT_EQ(result.itran, 0);
    EXPECT_EQ(result.idim_up, 1);
    EXPECT_TRUE(result.found);
}

// Test unpack_and_merge_sequences
TEST(UnpackAndMergeSequences, MergeMultipleSequences)
{
    constexpr auto tuple_of_seqs = make_tuple(Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5, 6>{});
    constexpr auto result        = unpack_and_merge_sequences(tuple_of_seqs);
    using Expected               = Sequence<1, 2, 3, 4, 5, 6>;
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result)>, Expected>::value));
}

TEST(UnpackAndMergeSequences, SingleSequence)
{
    constexpr auto tuple_of_seqs = make_tuple(Sequence<1, 2, 3>{});
    constexpr auto result        = unpack_and_merge_sequences(tuple_of_seqs);
    using Expected               = Sequence<1, 2, 3>;
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result)>, Expected>::value));
}
