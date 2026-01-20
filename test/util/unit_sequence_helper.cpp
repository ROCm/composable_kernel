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

// Test find_in_tuple_of_sequences
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

// Test unpack_and_merge_sequences
TEST(UnpackAndMergeSequences, MergeMultipleSequences)
{
    constexpr auto tuple_of_seqs =
        make_tuple(Sequence<7, 11>{}, Sequence<13>{}, Sequence<17, 19, 23>{});
    constexpr auto result = unpack_and_merge_sequences(tuple_of_seqs);
    using Expected        = Sequence<7, 11, 13, 17, 19, 23>;
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result)>, Expected>::value));
}

TEST(UnpackAndMergeSequences, SingleSequence)
{
    constexpr auto tuple_of_seqs = make_tuple(Sequence<29, 31, 37, 41>{});
    constexpr auto result        = unpack_and_merge_sequences(tuple_of_seqs);
    using Expected               = Sequence<29, 31, 37, 41>;
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result)>, Expected>::value));
}

TEST(UnpackAndMergeSequences, TwoSequences)
{
    constexpr auto tuple_of_seqs = make_tuple(Sequence<2, 3, 5>{}, Sequence<7, 11>{});
    constexpr auto result        = unpack_and_merge_sequences(tuple_of_seqs);
    using Expected               = Sequence<2, 3, 5, 7, 11>;
    EXPECT_TRUE((is_same<remove_cvref_t<decltype(result)>, Expected>::value));
}
