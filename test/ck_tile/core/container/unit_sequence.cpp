// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/core/container/sequence.hpp"

using namespace ck_tile;

// Test basic Sequence construction and properties
TEST(Sequence, BasicConstruction)
{
    using Seq = sequence<1, 2, 3, 4, 5>;
    EXPECT_EQ(Seq::size(), 5);
}

TEST(Sequence, EmptySequence)
{
    using Seq = sequence<>;
    EXPECT_EQ(Seq::size(), 0);
}

// Test at() method
TEST(Sequence, AtRuntime)
{
    using Seq = sequence<10, 20, 30, 40>;
    EXPECT_EQ(Seq::at(0), 10);
    EXPECT_EQ(Seq::at(1), 20);
    EXPECT_EQ(Seq::at(2), 30);
    EXPECT_EQ(Seq::at(3), 40);
}

TEST(Sequence, AtCompileTime)
{
    using Seq = sequence<10, 20, 30, 40>;
    EXPECT_EQ(Seq::at(number<0>{}), 10);
    EXPECT_EQ(Seq::at(number<1>{}), 20);
    EXPECT_EQ(Seq::at(number<2>{}), 30);
    EXPECT_EQ(Seq::at(number<3>{}), 40);
}

TEST(Sequence, OperatorBracket)
{
    constexpr auto seq = sequence<5, 10, 15>{};
    EXPECT_EQ(seq[number<0>{}], 5);
    EXPECT_EQ(seq[number<1>{}], 10);
    EXPECT_EQ(seq[number<2>{}], 15);
}

// Test Front() and Back()
TEST(Sequence, FrontBack)
{
    using Seq = sequence<100, 200, 300>;
    EXPECT_EQ(Seq::front(), 100);
    EXPECT_EQ(Seq::back(), 300);
}

TEST(Sequence, FrontBackSingleElement)
{
    using Seq = sequence<42>;
    EXPECT_EQ(Seq::front(), 42);
    EXPECT_EQ(Seq::back(), 42);
}

// Test PushFront and PushBack
TEST(Sequence, PushFront)
{
    using Seq      = sequence<2, 3, 4>;
    using Result   = decltype(Seq::push_front(sequence<1>{}));
    using Expected = sequence<1, 2, 3, 4>;
    EXPECT_TRUE((std::is_same_v<Result, Expected>));
}

TEST(Sequence, PushFrontnumbers)
{
    using Seq      = sequence<3, 4>;
    using Result   = decltype(Seq::push_front(number<1>{}, number<2>{}));
    using Expected = sequence<1, 2, 3, 4>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(Sequence, PushBack)
{
    using Seq      = sequence<1, 2, 3>;
    using Result   = decltype(Seq::push_back(sequence<4, 5>{}));
    using Expected = sequence<1, 2, 3, 4, 5>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(Sequence, PushBacknumbers)
{
    using Seq      = sequence<1, 2>;
    using Result   = decltype(Seq::push_back(number<3>{}, number<4>{}));
    using Expected = sequence<1, 2, 3, 4>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test PopFront and PopBack
TEST(Sequence, PopFront)
{
    using Seq      = sequence<1, 2, 3, 4>;
    using Result   = decltype(Seq::pop_front());
    using Expected = sequence<2, 3, 4>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(Sequence, PopBack)
{
    using Seq      = sequence<1, 2, 3, 4>;
    using Result   = decltype(Seq::pop_back());
    using Expected = sequence<1, 2, 3>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test Extract
TEST(Sequence, ExtractBynumbers)
{
    using Seq      = sequence<10, 20, 30, 40, 50>;
    using Result   = decltype(Seq::extract(number<0>{}, number<2>{}, number<4>{}));
    using Expected = sequence<10, 30, 50>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(Sequence, ExtractBySequence)
{
    using Seq      = sequence<10, 20, 30, 40, 50>;
    using Result   = decltype(Seq::extract(sequence<1, 3>{}));
    using Expected = sequence<20, 40>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test Modify
TEST(Sequence, Modify)
{
    using Seq      = sequence<1, 2, 3, 4>;
    using Result   = decltype(Seq::modify(number<2>{}, number<99>{}));
    using Expected = sequence<1, 2, 99, 4>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test Transform
TEST(Sequence, Transform)
{
    using Seq      = sequence<1, 2, 3, 4>;
    auto double_it = [](auto x) { return 2 * x; };
    using Result   = decltype(Seq::transform(double_it));
    using Expected = sequence<2, 4, 6, 8>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test Reverse
TEST(Sequence, Reverse)
{
    using Seq      = sequence<1, 2, 3, 4, 5>;
    using Result   = decltype(Seq::reverse());
    using Expected = sequence<5, 4, 3, 2, 1>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(Sequence, ReverseSingleElement)
{
    using Seq      = sequence<42>;
    using Result   = decltype(Seq::reverse());
    using Expected = sequence<42>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test ReorderGivenNew2Old
TEST(Sequence, ReorderGivenNew2Old)
{
    using Seq      = sequence<10, 20, 30, 40>;
    using Result   = decltype(Seq::reorder_new_to_old(sequence<3, 1, 2, 0>{}));
    using Expected = sequence<40, 20, 30, 10>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test ReorderGivenOld2New
TEST(Sequence, ReorderGivenOld2New)
{
    using Seq      = sequence<10, 20, 30, 40>;
    using Result   = decltype(Seq::reorder_old_to_new(sequence<3, 1, 2, 0>{}));
    using Expected = sequence<40, 20, 30, 10>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test arithmetic_sequence_gen
TEST(SequenceGen, ArithmeticSequence)
{
    using Result   = typename arithmetic_sequence_gen<0, 5, 1>::type;
    using Expected = sequence<0, 1, 2, 3, 4>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceGen, ArithmeticSequenceWithIncrement)
{
    using Result   = typename arithmetic_sequence_gen<0, 10, 2>::type;
    using Expected = sequence<0, 2, 4, 6, 8>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceGen, ArithmeticSequenceNegativeIncrement)
{
    using Result   = typename arithmetic_sequence_gen<10, 5, -1>::type;
    using Expected = sequence<10, 9, 8, 7, 6>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceGen, ArithmeticSequenceEmpty)
{
    using Result   = typename arithmetic_sequence_gen<5, 5, 1>::type;
    using Expected = sequence<>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test uniform_sequence_gen
TEST(SequenceGen, UniformSequence)
{
    using Result   = typename uniform_sequence_gen<5, 42>::type;
    using Expected = sequence<42, 42, 42, 42, 42>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceGen, UniformSequenceZeroSize)
{
    using Result   = typename uniform_sequence_gen<0, 42>::type;
    using Expected = sequence<>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test make_index_sequence
TEST(SequenceGen, MakeIndexSequence)
{
    using Result   = make_index_sequence<5>;
    using Expected = sequence<0, 1, 2, 3, 4>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceGen, MakeIndexSequenceZero)
{
    using Result   = make_index_sequence<0>;
    using Expected = sequence<>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test sequence_merge
TEST(SequenceMerge, MergeTwoSequences)
{
    using Seq1     = sequence<1, 2, 3>;
    using Seq2     = sequence<4, 5, 6>;
    using Result   = typename sequence_merge<Seq1, Seq2>::type;
    using Expected = sequence<1, 2, 3, 4, 5, 6>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceMerge, MergeMultipleSequences)
{
    using Seq1     = sequence<1, 2>;
    using Seq2     = sequence<3, 4>;
    using Seq3     = sequence<5, 6>;
    using Result   = typename sequence_merge<Seq1, Seq2, Seq3>::type;
    using Expected = sequence<1, 2, 3, 4, 5, 6>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceMerge, MergeSingleSequence)
{
    using Seq      = sequence<1, 2, 3>;
    using Result   = typename sequence_merge<Seq>::type;
    using Expected = sequence<1, 2, 3>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test sequence_split
TEST(SequenceSplit, SplitInMiddle)
{
    using Seq           = sequence<1, 2, 3, 4, 5, 6>;
    using Split         = sequence_split<Seq, 3>;
    using ExpectedLeft  = sequence<1, 2, 3>;
    using ExpectedRight = sequence<4, 5, 6>;
    EXPECT_TRUE((std::is_same<typename Split::left_type, ExpectedLeft>::value));
    EXPECT_TRUE((std::is_same<typename Split::right_type, ExpectedRight>::value));
}

TEST(SequenceSplit, SplitAtBeginning)
{
    using Seq           = sequence<1, 2, 3, 4>;
    using Split         = sequence_split<Seq, 0>;
    using ExpectedLeft  = sequence<>;
    using ExpectedRight = sequence<1, 2, 3, 4>;
    EXPECT_TRUE((std::is_same<typename Split::left_type, ExpectedLeft>::value));
    EXPECT_TRUE((std::is_same<typename Split::right_type, ExpectedRight>::value));
}

TEST(SequenceSplit, SplitAtEnd)
{
    using Seq           = sequence<1, 2, 3, 4>;
    using Split         = sequence_split<Seq, 4>;
    using ExpectedLeft  = sequence<1, 2, 3, 4>;
    using ExpectedRight = sequence<>;
    EXPECT_TRUE((std::is_same<typename Split::left_type, ExpectedLeft>::value));
    EXPECT_TRUE((std::is_same<typename Split::right_type, ExpectedRight>::value));
}

// Test sequence_sort
TEST(SequenceSort, SortAscending)
{
    using Seq      = sequence<5, 2, 8, 1, 9>;
    using Result   = typename sequence_sort<Seq, less<index_t>>::type;
    using Expected = sequence<1, 2, 5, 8, 9>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// TEST(SequenceSort, SortDescending)
// {
//     // Create a greater-than comparator
//     struct greater
//     {
//         __host__ constexpr bool operator()(index_t x, index_t y) const { return x > y; }
//     };
//     using Seq      = sequence<5, 2, 8, 1, 9>;
//     using Result   = typename sequence_sort<Seq, greater>::type;
//     using Expected = sequence<9, 8, 5, 2, 1>;
//     static_assert((std::is_same<Result, Expected>::value));
//     EXPECT_TRUE((std::is_same<Result, Expected>::value));
// }

TEST(SequenceSort, SortAlreadySorted)
{
    using Seq      = sequence<1, 2, 3, 4, 5>;
    using Result   = typename sequence_sort<Seq, less<index_t>>::type;
    using Expected = sequence<1, 2, 3, 4, 5>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceSort, SortWithDuplicates)
{
    using Seq      = sequence<3, 1, 4, 1, 5, 9, 2, 6, 5>;
    using Result   = typename sequence_sort<Seq, less<index_t>>::type;
    using Expected = sequence<1, 1, 2, 3, 4, 5, 5, 6, 9>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceSort, SortEmptySequence)
{
    using Seq      = sequence<>;
    using Result   = typename sequence_sort<Seq, less<index_t>>::type;
    using Expected = sequence<>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceSort, SortSingleElement)
{
    using Seq      = sequence<42>;
    using Result   = typename sequence_sort<Seq, less<index_t>>::type;
    using Expected = sequence<42>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test sequence_sort sorted2unsorted_map (index tracking)
TEST(SequenceSort, SortedMapUnsorted)
{
    using Seq  = sequence<5, 2, 8, 1, 9>;
    using Sort = sequence_sort<Seq, less<index_t>>;
    using Map  = typename Sort::sorted2unsorted_map;
    // sorted = <1,2,5,8,9>, original indices = <3,1,0,2,4>
    using Expected = sequence<3, 1, 0, 2, 4>;
    EXPECT_TRUE((std::is_same<Map, Expected>::value));
}

TEST(SequenceSort, SortedMapAlreadySorted)
{
    using Seq  = sequence<1, 2, 3, 4, 5>;
    using Sort = sequence_sort<Seq, less<index_t>>;
    using Map  = typename Sort::sorted2unsorted_map;
    // Already sorted: map should be identity
    using Expected = sequence<0, 1, 2, 3, 4>;
    EXPECT_TRUE((std::is_same<Map, Expected>::value));
}

TEST(SequenceSort, SortedMapRoundTrip)
{
    // Verify: sorted_values[i] == original[sorted2unsorted_map[i]]
    using Seq  = sequence<5, 2, 8, 1, 9>;
    using Sort = sequence_sort<Seq, less<index_t>>;
    // sorted = <1,2,5,8,9>, map = <3,1,0,2,4>
    EXPECT_EQ(Seq::at(Sort::sorted2unsorted_map::at(0)), Sort::type::at(0));
    EXPECT_EQ(Seq::at(Sort::sorted2unsorted_map::at(1)), Sort::type::at(1));
    EXPECT_EQ(Seq::at(Sort::sorted2unsorted_map::at(2)), Sort::type::at(2));
    EXPECT_EQ(Seq::at(Sort::sorted2unsorted_map::at(3)), Sort::type::at(3));
    EXPECT_EQ(Seq::at(Sort::sorted2unsorted_map::at(4)), Sort::type::at(4));
}

TEST(SequenceSort, SortedMapWithDuplicates)
{
    using Seq    = sequence<3, 1, 3, 1>;
    using Sort   = sequence_sort<Seq, less<index_t>>;
    using Sorted = typename Sort::type;
    using Map    = typename Sort::sorted2unsorted_map;
    // sorted = <1,1,3,3>
    using ExpectedSorted = sequence<1, 1, 3, 3>;
    EXPECT_TRUE((std::is_same<Sorted, ExpectedSorted>::value));
    // Verify round-trip: original[map[i]] == sorted[i] for all i
    // (don't assert specific index order for duplicates - sort stability may vary)
    EXPECT_EQ(Seq::at(Map::at(0)), Sorted::at(0));
    EXPECT_EQ(Seq::at(Map::at(1)), Sorted::at(1));
    EXPECT_EQ(Seq::at(Map::at(2)), Sorted::at(2));
    EXPECT_EQ(Seq::at(Map::at(3)), Sorted::at(3));
}

TEST(SequenceSort, SortedMapReverseSorted)
{
    using Seq       = sequence<5, 4, 3, 2, 1>;
    using Sort      = sequence_sort<Seq, less<index_t>>;
    using Sorted    = typename Sort::type;
    using Map       = typename Sort::sorted2unsorted_map;
    using ExpSorted = sequence<1, 2, 3, 4, 5>;
    using ExpMap    = sequence<4, 3, 2, 1, 0>;
    EXPECT_TRUE((std::is_same<Sorted, ExpSorted>::value));
    EXPECT_TRUE((std::is_same<Map, ExpMap>::value));
}

TEST(SequenceSort, SortedMapEmpty)
{
    using Sort = sequence_sort<sequence<>, less<index_t>>;
    using Map  = typename Sort::sorted2unsorted_map;
    EXPECT_TRUE((std::is_same<Map, sequence<>>::value));
}

TEST(SequenceSort, SortedMapSingleElement)
{
    using Sort = sequence_sort<sequence<42>, less<index_t>>;
    using Map  = typename Sort::sorted2unsorted_map;
    EXPECT_TRUE((std::is_same<Map, sequence<0>>::value));
}

// Test sequence_unique_sort sorted2unsorted_map
TEST(SequenceUniqueSort, UniqueSortMap)
{
    using Seq    = sequence<3, 1, 4, 1, 5, 9, 2, 6, 5>;
    using Result = sequence_unique_sort<Seq, less<index_t>, equal<index_t>>;
    using Map    = typename Result::sorted2unsorted_map;
    // sorted unique = <1,2,3,4,5,6,9>
    // The map should reference the first occurrence of each unique value in the original
    // Verify round-trip: for each i, original[map[i]] == sorted_unique[i]
    using Values = typename Result::type;
    EXPECT_EQ(Seq::at(Map::at(0)), Values::at(0)); // 1
    EXPECT_EQ(Seq::at(Map::at(1)), Values::at(1)); // 2
    EXPECT_EQ(Seq::at(Map::at(2)), Values::at(2)); // 3
    EXPECT_EQ(Seq::at(Map::at(3)), Values::at(3)); // 4
    EXPECT_EQ(Seq::at(Map::at(4)), Values::at(4)); // 5
    EXPECT_EQ(Seq::at(Map::at(5)), Values::at(5)); // 6
    EXPECT_EQ(Seq::at(Map::at(6)), Values::at(6)); // 9
}

// Test sequence_unique_sort
TEST(SequenceUniqueSort, UniqueSort)
{
    using Seq      = sequence<3, 1, 4, 1, 5, 9, 2, 6, 5>;
    using Result   = typename sequence_unique_sort<Seq, less<index_t>, equal<index_t>>::type;
    using Expected = sequence<1, 2, 3, 4, 5, 6, 9>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceUniqueSort, UniqueSortNoDuplicates)
{
    using Seq      = sequence<5, 2, 8, 1, 9>;
    using Result   = typename sequence_unique_sort<Seq, less<index_t>, equal<index_t>>::type;
    using Expected = sequence<1, 2, 5, 8, 9>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceUniqueSort, UniqueSortAllSame)
{
    using Seq      = sequence<5, 5, 5, 5>;
    using Result   = typename sequence_unique_sort<Seq, less<index_t>, equal<index_t>>::type;
    using Expected = sequence<5>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test is_valid_sequence_map
TEST(SequenceMap, ValidMap)
{
    using Map = sequence<0, 1, 2, 3>;
    EXPECT_TRUE((is_valid_sequence_map<Map>::value));
}

TEST(SequenceMap, ValidMapPermuted)
{
    using Map = sequence<2, 0, 3, 1>;
    EXPECT_TRUE((is_valid_sequence_map<Map>::value));
}

TEST(SequenceMap, InvalidMapDuplicate)
{
    using Map = sequence<0, 1, 1, 3>;
    EXPECT_FALSE((is_valid_sequence_map<Map>::value));
}

TEST(SequenceMap, InvalidMapMissing)
{
    using Map = sequence<0, 1, 3, 4>;
    EXPECT_FALSE((is_valid_sequence_map<Map>::value));
}

TEST(SequenceMap, InvalidMapNegative)
{
    using Map = sequence<0, -1, 2>;
    EXPECT_FALSE((is_valid_sequence_map<Map>::value));
}

TEST(SequenceMap, ValidMapSingleElement)
{
    EXPECT_TRUE((is_valid_sequence_map<sequence<0>>::value));
}

TEST(SequenceMap, InvalidMapSingleElement)
{
    EXPECT_FALSE((is_valid_sequence_map<sequence<1>>::value));
}

TEST(SequenceMap, ValidMapEmpty) { EXPECT_TRUE((is_valid_sequence_map<sequence<>>::value)); }

// Test sequence_map_inverse
// Note: sequence_map_inverse inverts a mapping where Map[i] = j means old position i maps to new
// position j The inverse gives us new position i came from old position inverse[i]
TEST(SequenceMapInverse, InverseMap)
{
    // Map = <2, 0, 3, 1> means: old[0]->new[2], old[1]->new[0], old[2]->new[3], old[3]->new[1]
    // Inverse should be: new[0]<-old[1], new[1]<-old[3], new[2]<-old[0], new[3]<-old[2]
    using Map    = sequence<2, 0, 3, 1>;
    using Result = typename sequence_map_inverse<Map>::type;
    // Verify by checking that Map[Result[i]] == i for all i
    EXPECT_EQ((Map::at(number<Result::at(number<0>{})>{}) == 0), true);
    EXPECT_EQ((Map::at(number<Result::at(number<1>{})>{}) == 1), true);
    EXPECT_EQ((Map::at(number<Result::at(number<2>{})>{}) == 2), true);
    EXPECT_EQ((Map::at(number<Result::at(number<3>{})>{}) == 3), true);
}

TEST(SequenceMapInverse, InverseIdentityMap)
{
    using Map    = sequence<0, 1, 2, 3>;
    using Result = typename sequence_map_inverse<Map>::type;
    // Verify by checking that Map[Result[i]] == i for all i (same as the other test)
    EXPECT_EQ((Map::at(number<Result::at(number<0>{})>{}) == 0), true);
    EXPECT_EQ((Map::at(number<Result::at(number<1>{})>{}) == 1), true);
    EXPECT_EQ((Map::at(number<Result::at(number<2>{})>{}) == 2), true);
    EXPECT_EQ((Map::at(number<Result::at(number<3>{})>{}) == 3), true);
}

// Test sequence operators
TEST(SequenceOperators, Equality)
{
    constexpr auto seq1 = sequence<1, 2, 3>{};
    constexpr auto seq2 = sequence<1, 2, 3>{};
    constexpr auto seq3 = sequence<1, 2, 4>{};
    EXPECT_TRUE(seq1 == seq2);
    EXPECT_FALSE(seq1 == seq3);
}

TEST(SequenceOperators, Addition)
{
    using Seq1     = sequence<1, 2, 3>;
    using Seq2     = sequence<4, 5, 6>;
    using Result   = decltype(Seq1{} + Seq2{});
    using Expected = sequence<5, 7, 9>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceOperators, Subtraction)
{
    using Seq1     = sequence<10, 20, 30>;
    using Seq2     = sequence<1, 2, 3>;
    using Result   = decltype(Seq1{} - Seq2{});
    using Expected = sequence<9, 18, 27>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceOperators, Multiplication)
{
    using Seq1     = sequence<2, 3, 4>;
    using Seq2     = sequence<5, 6, 7>;
    using Result   = decltype(Seq1{} * Seq2{});
    using Expected = sequence<10, 18, 28>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceOperators, Division)
{
    using Seq1     = sequence<10, 20, 30>;
    using Seq2     = sequence<2, 4, 5>;
    using Result   = decltype(Seq1{} / Seq2{});
    using Expected = sequence<5, 5, 6>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceOperators, Modulo)
{
    using Seq1     = sequence<10, 20, 30>;
    using Seq2     = sequence<3, 7, 8>;
    using Result   = decltype(Seq1{} % Seq2{});
    using Expected = sequence<1, 6, 6>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceOperators, AdditionWithnumber)
{
    using Seq      = sequence<1, 2, 3>;
    using Result   = decltype(Seq{} + number<10>{});
    using Expected = sequence<11, 12, 13>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceOperators, SubtractionWithnumber)
{
    using Seq      = sequence<10, 20, 30>;
    using Result   = decltype(Seq{} - number<5>{});
    using Expected = sequence<5, 15, 25>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceOperators, MultiplicationWithnumber)
{
    using Seq      = sequence<2, 3, 4>;
    using Result   = decltype(Seq{} * number<3>{});
    using Expected = sequence<6, 9, 12>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceOperators, DivisionWithnumber)
{
    using Seq      = sequence<10, 20, 30>;
    using Result   = decltype(Seq{} / number<5>{});
    using Expected = sequence<2, 4, 6>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceOperators, numberAddition)
{
    using Seq      = sequence<1, 2, 3>;
    using Result   = decltype(number<10>{} + Seq{});
    using Expected = sequence<11, 12, 13>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceOperators, numberMultiplication)
{
    using Seq      = sequence<2, 3, 4>;
    using Result   = decltype(number<3>{} * Seq{});
    using Expected = sequence<6, 9, 12>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test helper functions
TEST(SequenceHelpers, MergeSequences)
{
    using Seq1     = sequence<1, 2>;
    using Seq2     = sequence<3, 4>;
    using Seq3     = sequence<5, 6>;
    using Result   = decltype(merge_sequences(Seq1{}, Seq2{}, Seq3{}));
    using Expected = sequence<1, 2, 3, 4, 5, 6>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceHelpers, TransformSequencesSingle)
{
    auto double_it = [](auto x) { return 2 * x; };
    using Seq      = sequence<1, 2, 3>;
    using Result   = decltype(transform_sequences(double_it, Seq{}));
    using Expected = sequence<2, 4, 6>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceHelpers, TransformSequencesTwo)
{
    auto add       = [](auto x, auto y) { return x + y; };
    using Seq1     = sequence<1, 2, 3>;
    using Seq2     = sequence<4, 5, 6>;
    using Result   = decltype(transform_sequences(add, Seq1{}, Seq2{}));
    using Expected = sequence<5, 7, 9>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceHelpers, TransformSequencesThree)
{
    auto add3      = [](auto x, auto y, auto z) { return x + y + z; };
    using Seq1     = sequence<1, 2, 3>;
    using Seq2     = sequence<4, 5, 6>;
    using Seq3     = sequence<7, 8, 9>;
    using Result   = decltype(transform_sequences(add3, Seq1{}, Seq2{}, Seq3{}));
    using Expected = sequence<12, 15, 18>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceHelpers, ReduceOnSequence)
{
    auto add              = [](auto x, auto y) { return x + y; };
    constexpr auto seq    = sequence<1, 2, 3, 4, 5>{};
    constexpr auto result = reduce_on_sequence(seq, add, number<0>{});
    EXPECT_EQ(result, 15);
}

TEST(SequenceHelpers, SequenceAnyOf)
{
    auto is_even        = [](auto x) { return x % 2 == 0; };
    constexpr auto seq1 = sequence<1, 3, 5, 7>{};
    constexpr auto seq2 = sequence<1, 3, 4, 7>{};
    EXPECT_FALSE(sequence_any_of(seq1, is_even));
    EXPECT_TRUE(sequence_any_of(seq2, is_even));
}

TEST(SequenceHelpers, SequenceAllOf)
{
    auto is_positive    = [](auto x) { return x > 0; };
    constexpr auto seq1 = sequence<1, 2, 3, 4>{};
    constexpr auto seq2 = sequence<1, -2, 3, 4>{};
    EXPECT_TRUE(sequence_all_of(seq1, is_positive));
    EXPECT_FALSE(sequence_all_of(seq2, is_positive));
}

// Test scan operations
TEST(SequenceScan, ReverseInclusiveScan)
{
    using Seq      = sequence<1, 2, 3, 4>;
    using Result   = decltype(reverse_inclusive_scan_sequence(Seq{}, plus<index_t>{}, number<0>{}));
    using Expected = sequence<10, 9, 7, 4>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceScan, ReverseExclusiveScan)
{
    using Seq      = sequence<1, 2, 3, 4>;
    using Result   = decltype(reverse_exclusive_scan_sequence(Seq{}, plus<index_t>{}, number<0>{}));
    using Expected = sequence<9, 7, 4, 0>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceScan, InclusiveScan)
{
    using Seq      = sequence<1, 2, 3, 4>;
    using Result   = decltype(inclusive_scan_sequence(Seq{}, plus<index_t>{}, number<0>{}));
    using Expected = sequence<1, 3, 6, 10>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test pick and modify operations
TEST(SequencePickModify, PickElementsByIds)
{
    using Seq      = sequence<10, 20, 30, 40, 50>;
    using Ids      = sequence<0, 2, 4>;
    using Result   = decltype(pick_sequence_elements_by_ids(Seq{}, Ids{}));
    using Expected = sequence<10, 30, 50>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequencePickModify, PickElementsByMask)
{
    using Seq      = sequence<10, 20, 30, 40, 50>;
    using Mask     = sequence<1, 0, 1, 0, 1>;
    using Result   = decltype(pick_sequence_elements_by_mask(Seq{}, Mask{}));
    using Expected = sequence<10, 30, 50>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequencePickModify, ModifyElementsByIds)
{
    using Seq      = sequence<10, 20, 30, 40, 50>;
    using Values   = sequence<99, 88>;
    using Ids      = sequence<1, 3>;
    using Result   = decltype(modify_sequence_elements_by_ids(Seq{}, Values{}, Ids{}));
    using Expected = sequence<10, 99, 30, 88, 50>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

// Test sequence_reduce
TEST(SequenceReduce, ReduceTwoSequences)
{
    using Seq1     = sequence<1, 2, 3>;
    using Seq2     = sequence<4, 5, 6>;
    using Result   = typename sequence_reduce<plus<index_t>, Seq1, Seq2>::type;
    using Expected = sequence<5, 7, 9>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}

TEST(SequenceReduce, ReduceMultipleSequences)
{
    using Seq1     = sequence<1, 2>;
    using Seq2     = sequence<3, 4>;
    using Seq3     = sequence<5, 6>;
    using Result   = typename sequence_reduce<plus<index_t>, Seq1, Seq2, Seq3>::type;
    using Expected = sequence<9, 12>;
    EXPECT_TRUE((std::is_same<Result, Expected>::value));
}
