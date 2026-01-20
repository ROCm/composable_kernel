// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/utility/container_helper.hpp"
#include "ck/utility/tuple_helper.hpp"

using namespace ck;

// Test container_concat with tuples
TEST(ContainerConcat, ConcatTwoTuples)
{
    constexpr auto t1     = make_tuple(Number<7>{}, Number<11>{});
    constexpr auto t2     = make_tuple(Number<13>{}, Number<17>{});
    constexpr auto result = container_concat(t1, t2);

    EXPECT_EQ(result.Size(), 4);
    EXPECT_EQ(result[Number<0>{}], 7);
    EXPECT_EQ(result[Number<1>{}], 11);
    EXPECT_EQ(result[Number<2>{}], 13);
    EXPECT_EQ(result[Number<3>{}], 17);
}

TEST(ContainerConcat, ConcatThreeTuples)
{
    constexpr auto t1     = make_tuple(Number<19>{});
    constexpr auto t2     = make_tuple(Number<23>{}, Number<29>{});
    constexpr auto t3     = make_tuple(Number<31>{});
    constexpr auto result = container_concat(t1, t2, t3);

    EXPECT_EQ(result.Size(), 4);
    EXPECT_EQ(result[Number<0>{}], 19);
    EXPECT_EQ(result[Number<1>{}], 23);
    EXPECT_EQ(result[Number<2>{}], 29);
    EXPECT_EQ(result[Number<3>{}], 31);
}

TEST(ContainerConcat, ConcatWithEmptyTuple)
{
    constexpr auto t1     = make_tuple(Number<37>{}, Number<41>{});
    constexpr auto empty  = make_tuple();
    constexpr auto result = container_concat(t1, empty);

    EXPECT_EQ(result.Size(), 2);
    EXPECT_EQ(result[Number<0>{}], 37);
    EXPECT_EQ(result[Number<1>{}], 41);
}

TEST(ContainerConcat, ConcatSingleTuple)
{
    constexpr auto t1     = make_tuple(Number<43>{}, Number<47>{}, Number<53>{});
    constexpr auto result = container_concat(t1);

    EXPECT_EQ(result.Size(), 3);
    EXPECT_EQ(result[Number<0>{}], 43);
    EXPECT_EQ(result[Number<1>{}], 47);
    EXPECT_EQ(result[Number<2>{}], 53);
}

// Test container_concat with arrays
TEST(ContainerConcat, ConcatTwoArrays)
{
    constexpr auto a1     = make_array(59, 61);
    constexpr auto a2     = make_array(67, 71);
    constexpr auto result = container_concat(a1, a2);

    EXPECT_EQ(result.Size(), 4);
    EXPECT_EQ(result[Number<0>{}], 59);
    EXPECT_EQ(result[Number<1>{}], 61);
    EXPECT_EQ(result[Number<2>{}], 67);
    EXPECT_EQ(result[Number<3>{}], 71);
}

// Test make_uniform_tuple
TEST(MakeUniformTuple, Size3)
{
    constexpr auto result = make_uniform_tuple<3>(Number<73>{});

    EXPECT_EQ(result.Size(), 3);
    EXPECT_EQ(result[Number<0>{}], 73);
    EXPECT_EQ(result[Number<1>{}], 73);
    EXPECT_EQ(result[Number<2>{}], 73);
}

TEST(MakeUniformTuple, Size1)
{
    constexpr auto result = make_uniform_tuple<1>(Number<79>{});

    EXPECT_EQ(result.Size(), 1);
    EXPECT_EQ(result[Number<0>{}], 79);
}

TEST(MakeUniformTuple, Size0)
{
    constexpr auto result = make_uniform_tuple<0>(Number<83>{});

    EXPECT_EQ(result.Size(), 0);
}

TEST(MakeUniformTuple, Size5)
{
    constexpr auto result = make_uniform_tuple<5>(Number<89>{});

    EXPECT_EQ(result.Size(), 5);
    EXPECT_EQ(result[Number<0>{}], 89);
    EXPECT_EQ(result[Number<1>{}], 89);
    EXPECT_EQ(result[Number<2>{}], 89);
    EXPECT_EQ(result[Number<3>{}], 89);
    EXPECT_EQ(result[Number<4>{}], 89);
}

// Test make_tuple_functor (used internally by container_concat)
TEST(MakeTupleFunctor, CreatesTuple)
{
    make_tuple_functor functor;
    auto result = functor(Number<97>{}, Number<101>{}, Number<103>{});

    EXPECT_EQ(result.Size(), 3);
    EXPECT_EQ(result[Number<0>{}], 97);
    EXPECT_EQ(result[Number<1>{}], 101);
    EXPECT_EQ(result[Number<2>{}], 103);
}

// Test container_push_front and container_push_back
TEST(ContainerPush, PushFront)
{
    constexpr auto t      = make_tuple(Number<109>{}, Number<113>{});
    constexpr auto result = container_push_front(t, Number<107>{});

    EXPECT_EQ(result.Size(), 3);
    EXPECT_EQ(result[Number<0>{}], 107);
    EXPECT_EQ(result[Number<1>{}], 109);
    EXPECT_EQ(result[Number<2>{}], 113);
}

TEST(ContainerPush, PushBack)
{
    constexpr auto t      = make_tuple(Number<127>{}, Number<131>{});
    constexpr auto result = container_push_back(t, Number<137>{});

    EXPECT_EQ(result.Size(), 3);
    EXPECT_EQ(result[Number<0>{}], 127);
    EXPECT_EQ(result[Number<1>{}], 131);
    EXPECT_EQ(result[Number<2>{}], 137);
}
