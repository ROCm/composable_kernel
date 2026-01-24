// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/utility/array.hpp"

using namespace ck;

// Test basic Array construction and properties
TEST(Array, BasicConstruction)
{
    using Arr = Array<index_t, 5>;
    EXPECT_EQ(Arr::Size(), 5);
}

TEST(Array, InitListConstruction)
{
    using Arr = Array<index_t, 5>;
    Arr value{1, 2, 3, 4, 5};
    EXPECT_EQ(value[0], 1);
    EXPECT_EQ(value[4], 5);
}

// Test At() method
TEST(Array, AtMethod)
{
    Array<int, 3> arr{10, 20, 30};
    EXPECT_EQ(arr.At(0), 10);
    EXPECT_EQ(arr.At(1), 20);
    EXPECT_EQ(arr.At(2), 30);

    // Test non-const At() for modification
    arr.At(1) = 25;
    EXPECT_EQ(arr.At(1), 25);
}

// Test const At() method
TEST(Array, ConstAtMethod)
{
    const Array<int, 3> arr{10, 20, 30};
    EXPECT_EQ(arr.At(0), 10);
    EXPECT_EQ(arr.At(1), 20);
    EXPECT_EQ(arr.At(2), 30);
}

// Test operator[]
TEST(Array, OperatorBracket)
{
    const Array<int, 4> arr{5, 10, 15, 20};
    EXPECT_EQ(arr[0], 5);
    EXPECT_EQ(arr[1], 10);
    EXPECT_EQ(arr[2], 15);
    EXPECT_EQ(arr[3], 20);
}

// Test operator()
TEST(Array, OperatorParenthesis)
{
    Array<int, 3> arr{1, 2, 3};
    EXPECT_EQ(arr(0), 1);
    EXPECT_EQ(arr(1), 2);
    EXPECT_EQ(arr(2), 3);

    // Test modification through operator()
    arr(1) = 99;
    EXPECT_EQ(arr(1), 99);
}

// Test operator= assignment
TEST(Array, Assignment)
{
    Array<int, 3> arr1{1, 2, 3};
    Array<int, 3> arr2{0, 0, 0};

    arr2 = arr1;

    EXPECT_EQ(arr2[0], 1);
    EXPECT_EQ(arr2[1], 2);
    EXPECT_EQ(arr2[2], 3);
}

// Test iterators
TEST(Array, Iterators)
{
    Array<int, 5> arr{1, 2, 3, 4, 5};

    // Test begin() and end()
    int sum = 0;
    for(auto it = arr.begin(); it != arr.end(); ++it)
    {
        sum += *it;
    }
    EXPECT_EQ(sum, 15);

    // Test range-based for loop
    sum = 0;
    for(auto val : arr)
    {
        sum += val;
    }
    EXPECT_EQ(sum, 15);
}

// Test const iterators
TEST(Array, ConstIterators)
{
    const Array<int, 4> arr{10, 20, 30, 40};

    int sum = 0;
    for(auto it = arr.begin(); it != arr.end(); ++it)
    {
        sum += *it;
    }
    EXPECT_EQ(sum, 100);

    // Test const range-based for loop
    sum = 0;
    for(auto val : arr)
    {
        sum += val;
    }
    EXPECT_EQ(sum, 100);
}

// Test make_array() helper function
TEST(Array, MakeArray)
{
    auto arr = make_array(1, 2, 3, 4, 5);

    EXPECT_EQ(arr.Size(), 5);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
    EXPECT_EQ(arr[3], 4);
    EXPECT_EQ(arr[4], 5);
}

// Test make_array() with different types
TEST(Array, MakeArrayFloats)
{
    auto arr = make_array(1.5f, 2.5f, 3.5f);

    EXPECT_EQ(arr.Size(), 3);
    EXPECT_FLOAT_EQ(arr[0], 1.5f);
    EXPECT_FLOAT_EQ(arr[1], 2.5f);
    EXPECT_FLOAT_EQ(arr[2], 3.5f);
}

// Test empty Array<T, 0>
TEST(Array, EmptyArray)
{
    using EmptyArr = Array<int, 0>;
    EXPECT_EQ(EmptyArr::Size(), 0);

    // Test make_array() for empty array
    auto empty = make_array<int>();
    EXPECT_EQ(empty.Size(), 0);
}

// Test Array with different data types
TEST(Array, DifferentTypes)
{
    Array<float, 3> float_arr{1.1f, 2.2f, 3.3f};
    EXPECT_FLOAT_EQ(float_arr[0], 1.1f);
    EXPECT_FLOAT_EQ(float_arr[1], 2.2f);
    EXPECT_FLOAT_EQ(float_arr[2], 3.3f);

    Array<double, 2> double_arr{1.23, 4.56};
    EXPECT_DOUBLE_EQ(double_arr[0], 1.23);
    EXPECT_DOUBLE_EQ(double_arr[1], 4.56);
}

// Test Array modification through iterators
TEST(Array, ModifyThroughIterators)
{
    Array<int, 3> arr{1, 2, 3};

    for(auto it = arr.begin(); it != arr.end(); ++it)
    {
        *it *= 2;
    }

    EXPECT_EQ(arr[0], 2);
    EXPECT_EQ(arr[1], 4);
    EXPECT_EQ(arr[2], 6);
}

// Test single element Array
TEST(Array, SingleElement)
{
    Array<int, 1> arr{42};
    EXPECT_EQ(arr.Size(), 1);
    EXPECT_EQ(arr[0], 42);

    auto single = make_array(100);
    EXPECT_EQ(single.Size(), 1);
    EXPECT_EQ(single[0], 100);
}
