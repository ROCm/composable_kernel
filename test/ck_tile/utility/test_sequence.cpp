// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/numeric/math.hpp"

using namespace ck_tile;

// ============================================================================
// sequence::modify tests
// ============================================================================

TEST(CkTileSequence, ModifyFirstElement)
{
    constexpr auto result = sequence<1, 2, 3, 4>{}.modify(number<0>{}, number<99>{});
    EXPECT_EQ(result.at(0), 99);
    EXPECT_EQ(result.at(1), 2);
    EXPECT_EQ(result.at(2), 3);
    EXPECT_EQ(result.at(3), 4);
}

TEST(CkTileSequence, ModifyLastElement)
{
    constexpr auto result = sequence<1, 2, 3, 4>{}.modify(number<3>{}, number<99>{});
    EXPECT_EQ(result.at(0), 1);
    EXPECT_EQ(result.at(3), 99);
}

TEST(CkTileSequence, ModifyMiddleElement)
{
    constexpr auto result = sequence<5, 5, 5>{}.modify(number<1>{}, number<0>{});
    EXPECT_EQ(result.at(0), 5);
    EXPECT_EQ(result.at(1), 0);
    EXPECT_EQ(result.at(2), 5);
}

TEST(CkTileSequence, ModifySingleElement)
{
    constexpr auto result = sequence<42>{}.modify(number<0>{}, number<99>{});
    EXPECT_EQ(result.at(0), 99);
}

TEST(CkTileSequence, ModifyNegativeValue)
{
    constexpr auto result = sequence<1, 2, 3>{}.modify(number<1>{}, number<-1>{});
    EXPECT_EQ(result.at(0), 1);
    EXPECT_EQ(result.at(1), -1);
    EXPECT_EQ(result.at(2), 3);
}

// ============================================================================
// sequence_gen tests
// ============================================================================

TEST(CkTileSequence, SequenceGenZero)
{
    using Result = typename sequence_gen<0, identity>::type;
    EXPECT_EQ(Result::size(), 0);
}

TEST(CkTileSequence, SequenceGenZeroNonIdentityFunctor)
{
    // N=0 specialization should produce empty sequence regardless of functor.
    // Use sequence_gen<1, F> to exercise the functor (suppresses -Wunused-member-function),
    // then verify that N=0 still produces an empty sequence with the same functor type.
    struct F
    {
        constexpr index_t operator()(index_t) const { return 999; }
    };
    using ResultOne  = typename sequence_gen<1, F>::type;
    using ResultZero = typename sequence_gen<0, F>::type;
    EXPECT_EQ(ResultOne{}.at(0), 999);
    EXPECT_EQ(ResultZero::size(), 0);
}

TEST(CkTileSequence, SequenceGenIdentity)
{
    struct F
    {
        constexpr index_t operator()(index_t i) const { return i; }
    };
    using Result = typename sequence_gen<5, F>::type;
    EXPECT_EQ(Result::size(), 5);
    for(index_t i = 0; i < 5; ++i)
    {
        EXPECT_EQ(Result{}.at(i), i);
    }
}

TEST(CkTileSequence, SequenceGenDouble)
{
    struct F
    {
        constexpr index_t operator()(index_t i) const { return i * 2; }
    };
    using Result = typename sequence_gen<4, F>::type;
    EXPECT_EQ(Result{}.at(0), 0);
    EXPECT_EQ(Result{}.at(1), 2);
    EXPECT_EQ(Result{}.at(2), 4);
    EXPECT_EQ(Result{}.at(3), 6);
}

TEST(CkTileSequence, SequenceGenSingle)
{
    struct F
    {
        constexpr index_t operator()(index_t) const { return 42; }
    };
    using Result = typename sequence_gen<1, F>::type;
    EXPECT_EQ(Result::size(), 1);
    EXPECT_EQ(Result{}.at(0), 42);
}

TEST(CkTileSequence, SequenceGenLarger)
{
    struct F
    {
        constexpr index_t operator()(index_t i) const { return i * i; }
    };
    using Result = typename sequence_gen<8, F>::type;
    EXPECT_EQ(Result{}.at(7), 49);
}

// Defined at namespace scope because template members are not allowed in local classes.
namespace {
struct NumberParamFunctor
{
    template <index_t I>
    constexpr index_t operator()(number<I>) const
    {
        return I + 10;
    }
};
} // anonymous namespace

TEST(CkTileSequence, SequenceGenWithNumberParam)
{
    // Verify functor taking number<I> directly (the documented API contract)
    using Result = typename sequence_gen<4, NumberParamFunctor>::type;
    EXPECT_EQ(Result{}.at(0), 10);
    EXPECT_EQ(Result{}.at(3), 13);
}

// ============================================================================
// uniform_sequence_gen tests
// ============================================================================

TEST(CkTileSequence, UniformSequenceGenZero)
{
    using Result = typename uniform_sequence_gen<0, 7>::type;
    EXPECT_EQ(Result::size(), 0);
}

TEST(CkTileSequence, UniformSequenceGenSingle)
{
    using Result = typename uniform_sequence_gen<1, 99>::type;
    EXPECT_EQ(Result{}.at(0), 99);
}

TEST(CkTileSequence, UniformSequenceGenMultiple)
{
    using Result = typename uniform_sequence_gen<4, 0>::type;
    for(index_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(Result{}.at(i), 0);
    }
}

TEST(CkTileSequence, UniformSequenceGenLarger)
{
    using Result = typename uniform_sequence_gen<8, 3>::type;
    for(index_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(Result{}.at(i), 3);
    }
}

// ============================================================================
// sequence_reverse_inclusive_scan tests - runtime value verification
// ============================================================================

TEST(CkTileSequence, ReverseInclusiveScanProduct)
{
    using Result = typename sequence_reverse_inclusive_scan<sequence<1, 2, 3, 4>,
                                                            multiplies<index_t>,
                                                            1>::type;
    // result[3]=4*1=4, result[2]=3*4=12, result[1]=2*12=24, result[0]=1*24=24
    EXPECT_EQ(Result{}.at(0), 24);
    EXPECT_EQ(Result{}.at(1), 24);
    EXPECT_EQ(Result{}.at(2), 12);
    EXPECT_EQ(Result{}.at(3), 4);
}

TEST(CkTileSequence, ReverseInclusiveScanSum)
{
    using Result =
        typename sequence_reverse_inclusive_scan<sequence<1, 2, 3, 4>, plus<index_t>, 0>::type;
    // result[3]=4, result[2]=7, result[1]=9, result[0]=10
    EXPECT_EQ(Result{}.at(0), 10);
    EXPECT_EQ(Result{}.at(1), 9);
    EXPECT_EQ(Result{}.at(2), 7);
    EXPECT_EQ(Result{}.at(3), 4);
}

TEST(CkTileSequence, ReverseInclusiveScanSingleElement)
{
    using Result = typename sequence_reverse_inclusive_scan<sequence<5>, plus<index_t>, 0>::type;
    EXPECT_EQ(Result{}.at(0), 5);
}

TEST(CkTileSequence, ReverseInclusiveScanEmpty)
{
    using Result = typename sequence_reverse_inclusive_scan<sequence<>, plus<index_t>, 0>::type;
    EXPECT_EQ(Result::size(), 0);
}

// ============================================================================
// sequence_inclusive_scan (forward) tests - runtime value verification
// ============================================================================

TEST(CkTileSequence, ForwardInclusiveScanSum)
{
    using Result = typename sequence_inclusive_scan<sequence<1, 2, 3, 4>, plus<index_t>, 0>::type;
    // result[0]=1, result[1]=3, result[2]=6, result[3]=10
    EXPECT_EQ(Result{}.at(0), 1);
    EXPECT_EQ(Result{}.at(1), 3);
    EXPECT_EQ(Result{}.at(2), 6);
    EXPECT_EQ(Result{}.at(3), 10);
}

TEST(CkTileSequence, ForwardInclusiveScanProduct)
{
    using Result =
        typename sequence_inclusive_scan<sequence<1, 2, 3, 4>, multiplies<index_t>, 1>::type;
    // result[0]=1, result[1]=2, result[2]=6, result[3]=24
    EXPECT_EQ(Result{}.at(0), 1);
    EXPECT_EQ(Result{}.at(1), 2);
    EXPECT_EQ(Result{}.at(2), 6);
    EXPECT_EQ(Result{}.at(3), 24);
}

TEST(CkTileSequence, ForwardInclusiveScanNonTrivialInit)
{
    using Result = typename sequence_inclusive_scan<sequence<1, 2, 3>, plus<index_t>, 10>::type;
    // init=10: result[0]=1+10=11, result[1]=2+11=13, result[2]=3+13=16
    EXPECT_EQ(Result{}.at(0), 11);
    EXPECT_EQ(Result{}.at(1), 13);
    EXPECT_EQ(Result{}.at(2), 16);
}

TEST(CkTileSequence, ReverseInclusiveScanNonTrivialInit)
{
    using Result =
        typename sequence_reverse_inclusive_scan<sequence<1, 2, 3>, plus<index_t>, 10>::type;
    // init=10: result[2]=3+10=13, result[1]=2+13=15, result[0]=1+15=16
    EXPECT_EQ(Result{}.at(0), 16);
    EXPECT_EQ(Result{}.at(1), 15);
    EXPECT_EQ(Result{}.at(2), 13);
}

TEST(CkTileSequence, ForwardInclusiveScanSingleElement)
{
    using Result = typename sequence_inclusive_scan<sequence<5>, plus<index_t>, 0>::type;
    EXPECT_EQ(Result{}.at(0), 5);
}

TEST(CkTileSequence, ForwardInclusiveScanEmpty)
{
    using Result = typename sequence_inclusive_scan<sequence<>, plus<index_t>, 0>::type;
    EXPECT_EQ(Result::size(), 0);
}

// ============================================================================
// sequence_map_inverse tests - runtime round-trip verification
// ============================================================================

TEST(CkTileSequence, MapInverseIdentity)
{
    using Result = typename sequence_map_inverse<sequence<0, 1, 2, 3>>::type;
    for(index_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(Result{}.at(i), i);
    }
}

TEST(CkTileSequence, MapInverseSwap)
{
    using Result = typename sequence_map_inverse<sequence<1, 0>>::type;
    EXPECT_EQ(Result{}.at(0), 1);
    EXPECT_EQ(Result{}.at(1), 0);
}

TEST(CkTileSequence, MapInversePermutation)
{
    using Input  = sequence<2, 0, 1>;
    using Result = typename sequence_map_inverse<Input>::type;
    EXPECT_EQ(Result{}.at(0), 1);
    EXPECT_EQ(Result{}.at(1), 2);
    EXPECT_EQ(Result{}.at(2), 0);

    // Verify round-trip: input[result[i]] == i for all i
    for(index_t i = 0; i < 3; ++i)
    {
        EXPECT_EQ(Input{}.at(Result{}.at(i)), i);
    }
}

TEST(CkTileSequence, MapInverseEmpty)
{
    using Result = typename sequence_map_inverse<sequence<>>::type;
    EXPECT_EQ(Result::size(), 0);
}

TEST(CkTileSequence, MapInverseSingle)
{
    using Result = typename sequence_map_inverse<sequence<0>>::type;
    EXPECT_EQ(Result{}.at(0), 0);
}

TEST(CkTileSequence, MapInverseRotation)
{
    using Input  = sequence<1, 2, 0>;
    using Result = typename sequence_map_inverse<Input>::type;
    for(index_t i = 0; i < 3; ++i)
    {
        EXPECT_EQ(Input{}.at(Result{}.at(i)), i);
    }
}

TEST(CkTileSequence, MapInverse4D)
{
    using Input  = sequence<2, 0, 3, 1>;
    using Result = typename sequence_map_inverse<Input>::type;
    EXPECT_EQ(Result{}.at(0), 1);
    EXPECT_EQ(Result{}.at(1), 3);
    EXPECT_EQ(Result{}.at(2), 0);
    EXPECT_EQ(Result{}.at(3), 2);

    // Verify round-trip: input[result[i]] == i for all i
    for(index_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(Input{}.at(Result{}.at(i)), i);
    }
}

// ============================================================================
// make_index_sequence tests
// ============================================================================

TEST(CkTileSequence, MakeIndexSequenceZero)
{
    using Result = make_index_sequence<0>;
    EXPECT_EQ(Result::size(), 0);
}

TEST(CkTileSequence, MakeIndexSequenceOne)
{
    using Result = make_index_sequence<1>;
    EXPECT_EQ(Result::size(), 1);
    EXPECT_EQ(Result{}.at(0), 0);
}

TEST(CkTileSequence, MakeIndexSequenceSmall)
{
    using Result = make_index_sequence<5>;
    EXPECT_EQ(Result::size(), 5);
    for(index_t i = 0; i < 5; ++i)
    {
        EXPECT_EQ(Result{}.at(i), i);
    }
}

// ============================================================================
// sequence basic accessors tests
// ============================================================================

TEST(CkTileSequence, SizeAndIsStatic)
{
    EXPECT_EQ((sequence<1, 2, 3>::size()), 3);
    EXPECT_EQ((sequence<>::size()), 0);
    EXPECT_TRUE((sequence<1, 2, 3>::is_static()));
}

TEST(CkTileSequence, FrontAndBack)
{
    constexpr auto s = sequence<10, 20, 30>{};
    EXPECT_EQ(s.at(0), 10);
    EXPECT_EQ(s.at(2), 30);
}

TEST(CkTileSequence, SumAndProduct)
{
    EXPECT_EQ((sequence<1, 2, 3, 4>::sum()), 10);
    EXPECT_EQ((sequence<1, 2, 3, 4>::product()), 24);
    EXPECT_EQ((sequence<>::sum()), 0);
    EXPECT_EQ((sequence<>::product()), 1);
    EXPECT_EQ((sequence<5>::sum()), 5);
    EXPECT_EQ((sequence<5>::product()), 5);
}

// ============================================================================
// sequence push/pop tests
// ============================================================================

TEST(CkTileSequence, PushFrontSequence)
{
    constexpr auto result = sequence<3, 4>{}.push_front(sequence<1, 2>{});
    EXPECT_EQ(result.at(0), 1);
    EXPECT_EQ(result.at(1), 2);
    EXPECT_EQ(result.at(2), 3);
    EXPECT_EQ(result.at(3), 4);
}

TEST(CkTileSequence, PushBackSequence)
{
    constexpr auto result = sequence<1, 2>{}.push_back(sequence<3, 4>{});
    EXPECT_EQ(result.at(0), 1);
    EXPECT_EQ(result.at(1), 2);
    EXPECT_EQ(result.at(2), 3);
    EXPECT_EQ(result.at(3), 4);
}

TEST(CkTileSequence, PopFront)
{
    constexpr auto result = sequence_pop_front(sequence<1, 2, 3>{});
    EXPECT_EQ(decltype(result)::size(), 2);
    EXPECT_EQ(result.at(0), 2);
    EXPECT_EQ(result.at(1), 3);
}

TEST(CkTileSequence, PopBack)
{
    constexpr auto result = sequence_pop_back(sequence<1, 2, 3>{});
    EXPECT_EQ(decltype(result)::size(), 2);
    EXPECT_EQ(result.at(0), 1);
    EXPECT_EQ(result.at(1), 2);
}

// ============================================================================
// sequence reverse tests
// ============================================================================

TEST(CkTileSequence, Reverse)
{
    constexpr auto result = sequence<1, 2, 3, 4>{}.reverse();
    EXPECT_EQ(result.at(0), 4);
    EXPECT_EQ(result.at(1), 3);
    EXPECT_EQ(result.at(2), 2);
    EXPECT_EQ(result.at(3), 1);
}

TEST(CkTileSequence, ReverseSingle)
{
    constexpr auto result = sequence<42>{}.reverse();
    EXPECT_EQ(result.at(0), 42);
}

// ============================================================================
// sequence extract tests
// ============================================================================

TEST(CkTileSequence, Extract)
{
    constexpr auto result = sequence<10, 20, 30, 40>{}.extract(sequence<2, 0, 3>{});
    EXPECT_EQ(result.at(0), 30);
    EXPECT_EQ(result.at(1), 10);
    EXPECT_EQ(result.at(2), 40);
}

// ============================================================================
// sequence_merge tests
// ============================================================================

TEST(CkTileSequence, MergeTwoSequences)
{
    constexpr auto result = merge_sequences(sequence<1, 2>{}, sequence<3, 4>{});
    EXPECT_EQ(decltype(result)::size(), 4);
    EXPECT_EQ(result.at(0), 1);
    EXPECT_EQ(result.at(3), 4);
}

TEST(CkTileSequence, MergeWithEmpty)
{
    constexpr auto result = merge_sequences(sequence<1, 2>{}, sequence<>{});
    EXPECT_EQ(decltype(result)::size(), 2);
    EXPECT_EQ(result.at(0), 1);
}

// ============================================================================
// sequence arithmetic operator tests
// ============================================================================

TEST(CkTileSequence, OperatorAdd)
{
    constexpr auto result = sequence<1, 2, 3>{} + sequence<10, 20, 30>{};
    EXPECT_EQ(result.at(0), 11);
    EXPECT_EQ(result.at(1), 22);
    EXPECT_EQ(result.at(2), 33);
}

TEST(CkTileSequence, OperatorSubtract)
{
    constexpr auto result = sequence<10, 20, 30>{} - sequence<1, 2, 3>{};
    EXPECT_EQ(result.at(0), 9);
    EXPECT_EQ(result.at(1), 18);
    EXPECT_EQ(result.at(2), 27);
}

TEST(CkTileSequence, OperatorMultiply)
{
    constexpr auto result = sequence<2, 3, 4>{} * sequence<5, 6, 7>{};
    EXPECT_EQ(result.at(0), 10);
    EXPECT_EQ(result.at(1), 18);
    EXPECT_EQ(result.at(2), 28);
}

TEST(CkTileSequence, OperatorAddScalar)
{
    constexpr auto result = sequence<1, 2, 3>{} + number<10>{};
    EXPECT_EQ(result.at(0), 11);
    EXPECT_EQ(result.at(1), 12);
    EXPECT_EQ(result.at(2), 13);
}

TEST(CkTileSequence, OperatorMultiplyScalar)
{
    constexpr auto result = sequence<1, 2, 3>{} * number<10>{};
    EXPECT_EQ(result.at(0), 10);
    EXPECT_EQ(result.at(1), 20);
    EXPECT_EQ(result.at(2), 30);
}

TEST(CkTileSequence, ScalarOperatorAdd)
{
    constexpr auto result = number<100>{} + sequence<1, 2, 3>{};
    EXPECT_EQ(result.at(0), 101);
    EXPECT_EQ(result.at(1), 102);
    EXPECT_EQ(result.at(2), 103);
}

// ============================================================================
// sequence equality tests
// ============================================================================

TEST(CkTileSequence, EqualityTrue) { EXPECT_TRUE((sequence<1, 2, 3>{} == sequence<1, 2, 3>{})); }

TEST(CkTileSequence, EqualityFalse) { EXPECT_FALSE((sequence<1, 2, 3>{} == sequence<1, 2, 4>{})); }

TEST(CkTileSequence, InequalityTrue) { EXPECT_TRUE((sequence<1, 2, 3>{} != sequence<1, 2, 4>{})); }

TEST(CkTileSequence, EqualityEmpty) { EXPECT_TRUE((sequence<>{} == sequence<>{})); }

// ============================================================================
// sequence transform tests
// ============================================================================

TEST(CkTileSequence, Transform)
{
    struct Double
    {
        constexpr index_t operator()(index_t x) const { return x * 2; }
    };
    constexpr auto result = sequence<1, 2, 3>{}.transform(Double{});
    EXPECT_EQ(result.at(0), 2);
    EXPECT_EQ(result.at(1), 4);
    EXPECT_EQ(result.at(2), 6);
}

// ============================================================================
// exclusive_scan_sequence tests
// ============================================================================

TEST(CkTileSequence, ExclusiveScanSum)
{
    // <2, 3, 4> with Init=0, Add -> <0, 2, 5>
    constexpr auto result =
        exclusive_scan_sequence(sequence<2, 3, 4>{}, plus<index_t>{}, number<0>{});
    EXPECT_EQ(result.at(0), 0);
    EXPECT_EQ(result.at(1), 2);
    EXPECT_EQ(result.at(2), 5);
}

TEST(CkTileSequence, ExclusiveScanProduct)
{
    // <2, 3, 4> with Init=1, Mul -> <1, 2, 6>
    constexpr auto result =
        exclusive_scan_sequence(sequence<2, 3, 4>{}, multiplies<index_t>{}, number<1>{});
    EXPECT_EQ(result.at(0), 1);
    EXPECT_EQ(result.at(1), 2);
    EXPECT_EQ(result.at(2), 6);
}

TEST(CkTileSequence, ExclusiveScanSingle)
{
    constexpr auto result = exclusive_scan_sequence(sequence<5>{}, plus<index_t>{}, number<0>{});
    EXPECT_EQ(decltype(result)::size(), 1);
    EXPECT_EQ(result.at(0), 0);
}

TEST(CkTileSequence, ExclusiveScanEmpty)
{
    constexpr auto result = exclusive_scan_sequence(sequence<>{}, plus<index_t>{}, number<0>{});
    EXPECT_EQ(decltype(result)::size(), 0);
}

TEST(CkTileSequence, ExclusiveScanNonZeroInit)
{
    // <1, 2, 3> with Init=10, Add -> <10, 11, 13>
    constexpr auto result =
        exclusive_scan_sequence(sequence<1, 2, 3>{}, plus<index_t>{}, number<10>{});
    EXPECT_EQ(result.at(0), 10);
    EXPECT_EQ(result.at(1), 11);
    EXPECT_EQ(result.at(2), 13);
}

// ============================================================================
// prefix_sum_sequence tests
// ============================================================================

TEST(CkTileSequence, PrefixSumSequence)
{
    // <2, 3, 4> -> <0, 2, 5, 9> (N+1 elements)
    constexpr auto result = prefix_sum_sequence(sequence<2, 3, 4>{});
    EXPECT_EQ(decltype(result)::size(), 4);
    EXPECT_EQ(result.at(0), 0);
    EXPECT_EQ(result.at(1), 2);
    EXPECT_EQ(result.at(2), 5);
    EXPECT_EQ(result.at(3), 9);
}

TEST(CkTileSequence, PrefixSumSingle)
{
    // <5> -> <0, 5>
    constexpr auto result = prefix_sum_sequence(sequence<5>{});
    EXPECT_EQ(decltype(result)::size(), 2);
    EXPECT_EQ(result.at(0), 0);
    EXPECT_EQ(result.at(1), 5);
}

TEST(CkTileSequence, PrefixSumEmpty)
{
    // <> -> <0>
    constexpr auto result = prefix_sum_sequence(sequence<>{});
    EXPECT_EQ(decltype(result)::size(), 1);
    EXPECT_EQ(result.at(0), 0);
}
