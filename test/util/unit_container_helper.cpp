// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/utility/container_helper.hpp"
#include "ck/utility/tuple_helper.hpp"

using namespace ck;

// Test container_concat with tuples
TEST(ContainerConcat, ConcatTwoTuples)
{
    constexpr auto t1     = make_tuple(Number<1>{}, Number<2>{});
    constexpr auto t2     = make_tuple(Number<3>{}, Number<4>{});
    constexpr auto result = container_concat(t1, t2);

    EXPECT_EQ(result.Size(), 4);
    EXPECT_EQ(result[Number<0>{}], 1);
    EXPECT_EQ(result[Number<1>{}], 2);
    EXPECT_EQ(result[Number<2>{}], 3);
    EXPECT_EQ(result[Number<3>{}], 4);
}

TEST(ContainerConcat, ConcatThreeTuples)
{
    constexpr auto t1     = make_tuple(Number<1>{});
    constexpr auto t2     = make_tuple(Number<2>{});
    constexpr auto t3     = make_tuple(Number<3>{});
    constexpr auto result = container_concat(t1, t2, t3);

    EXPECT_EQ(result.Size(), 3);
    EXPECT_EQ(result[Number<0>{}], 1);
    EXPECT_EQ(result[Number<1>{}], 2);
    EXPECT_EQ(result[Number<2>{}], 3);
}

TEST(ContainerConcat, ConcatWithEmptyTuple)
{
    constexpr auto t1     = make_tuple(Number<1>{}, Number<2>{});
    constexpr auto empty  = make_tuple();
    constexpr auto result = container_concat(t1, empty);

    EXPECT_EQ(result.Size(), 2);
    EXPECT_EQ(result[Number<0>{}], 1);
    EXPECT_EQ(result[Number<1>{}], 2);
}

TEST(ContainerConcat, ConcatSingleTuple)
{
    constexpr auto t1     = make_tuple(Number<1>{}, Number<2>{}, Number<3>{});
    constexpr auto result = container_concat(t1);

    EXPECT_EQ(result.Size(), 3);
}

// Test container_concat with arrays
TEST(ContainerConcat, ConcatTwoArrays)
{
    constexpr auto a1     = make_array(1, 2);
    constexpr auto a2     = make_array(3, 4);
    constexpr auto result = container_concat(a1, a2);

    EXPECT_EQ(result.Size(), 4);
    EXPECT_EQ(result[Number<0>{}], 1);
    EXPECT_EQ(result[Number<1>{}], 2);
    EXPECT_EQ(result[Number<2>{}], 3);
    EXPECT_EQ(result[Number<3>{}], 4);
}

// Test make_uniform_tuple
TEST(MakeUniformTuple, Size3)
{
    constexpr auto result = make_uniform_tuple<3>(Number<42>{});

    EXPECT_EQ(result.Size(), 3);
    EXPECT_EQ(result[Number<0>{}], 42);
    EXPECT_EQ(result[Number<1>{}], 42);
    EXPECT_EQ(result[Number<2>{}], 42);
}

TEST(MakeUniformTuple, Size1)
{
    constexpr auto result = make_uniform_tuple<1>(Number<99>{});

    EXPECT_EQ(result.Size(), 1);
    EXPECT_EQ(result[Number<0>{}], 99);
}

TEST(MakeUniformTuple, Size0)
{
    constexpr auto result = make_uniform_tuple<0>(Number<42>{});

    EXPECT_EQ(result.Size(), 0);
}

TEST(MakeUniformTuple, Size5)
{
    constexpr auto result = make_uniform_tuple<5>(Number<7>{});

    EXPECT_EQ(result.Size(), 5);
    EXPECT_EQ(result[Number<0>{}], 7);
    EXPECT_EQ(result[Number<1>{}], 7);
    EXPECT_EQ(result[Number<2>{}], 7);
    EXPECT_EQ(result[Number<3>{}], 7);
    EXPECT_EQ(result[Number<4>{}], 7);
}

// Test make_tuple_functor (used internally by container_concat)
TEST(MakeTupleFunctor, CreatesTuple)
{
    make_tuple_functor functor;
    auto result = functor(Number<1>{}, Number<2>{}, Number<3>{});

    EXPECT_EQ(result.Size(), 3);
    EXPECT_EQ(result[Number<0>{}], 1);
    EXPECT_EQ(result[Number<1>{}], 2);
    EXPECT_EQ(result[Number<2>{}], 3);
}

// Test container_push_front and container_push_back
TEST(ContainerPush, PushFront)
{
    constexpr auto t      = make_tuple(Number<2>{}, Number<3>{});
    constexpr auto result = container_push_front(t, Number<1>{});

    EXPECT_EQ(result.Size(), 3);
    EXPECT_EQ(result[Number<0>{}], 1);
    EXPECT_EQ(result[Number<1>{}], 2);
    EXPECT_EQ(result[Number<2>{}], 3);
}

TEST(ContainerPush, PushBack)
{
    constexpr auto t      = make_tuple(Number<1>{}, Number<2>{});
    constexpr auto result = container_push_back(t, Number<3>{});

    EXPECT_EQ(result.Size(), 3);
    EXPECT_EQ(result[Number<0>{}], 1);
    EXPECT_EQ(result[Number<1>{}], 2);
    EXPECT_EQ(result[Number<2>{}], 3);
}
