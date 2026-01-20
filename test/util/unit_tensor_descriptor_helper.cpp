// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

using namespace ck;

// Test make_naive_tensor_descriptor (public API)
// Formula for element_space_size: 1 + sum((length[i] - 1) * stride[i])

TEST(MakeNaiveTensorDescriptor, ElementSpaceSize2D)
{
    // 5x4 tensor with row-major strides [4, 1]
    // element_space_size = 1 + (5-1)*4 + (4-1)*1 = 1 + 16 + 3 = 20
    constexpr auto lengths = make_tuple(Number<5>{}, Number<4>{});
    constexpr auto strides = make_tuple(Number<4>{}, Number<1>{});
    constexpr auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 20);
}

TEST(MakeNaiveTensorDescriptor, ElementSpaceSize3D)
{
    // 3x8x11 tensor with strides [88, 11, 1]
    // element_space_size = 1 + (3-1)*88 + (8-1)*11 + (11-1)*1 = 1 + 176 + 77 + 10 = 264
    constexpr auto lengths = make_tuple(Number<3>{}, Number<8>{}, Number<11>{});
    constexpr auto strides = make_tuple(Number<88>{}, Number<11>{}, Number<1>{});
    constexpr auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 264);
}

TEST(MakeNaiveTensorDescriptor, BroadcastDimension)
{
    // 8x5 tensor with broadcast on first dimension (stride 0)
    // element_space_size = 1 + (8-1)*0 + (5-1)*1 = 1 + 0 + 4 = 5
    constexpr auto lengths = make_tuple(Number<8>{}, Number<5>{});
    constexpr auto strides = make_tuple(Number<0>{}, Number<1>{});
    constexpr auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 5);
}

TEST(MakeNaiveTensorDescriptor, WithPaddingArbitrary)
{
    // 11x7x3 tensor with arbitrary strides [2, 97, 23] (prime numbers, no common factors)
    // This tests padding where offsets don't correspond to any packed array
    // element_space_size = 1 + (11-1)*2 + (7-1)*97 + (3-1)*23 = 1 + 20 + 582 + 46 = 649
    constexpr auto lengths = make_tuple(Number<11>{}, Number<7>{}, Number<3>{});
    constexpr auto strides = make_tuple(Number<2>{}, Number<97>{}, Number<23>{});
    constexpr auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 649);
}

TEST(MakeNaiveTensorDescriptor, WithPaddingStrideSlice)
{
    // 2x3x4 tensor with strides [1, 5, 35] - like a slice from a 5x7xN column-major tensor
    // This tests padding where there's space for extra elements
    // element_space_size = 1 + (2-1)*1 + (3-1)*5 + (4-1)*35 = 1 + 1 + 10 + 105 = 117
    constexpr auto lengths = make_tuple(Number<2>{}, Number<3>{}, Number<4>{});
    constexpr auto strides = make_tuple(Number<1>{}, Number<5>{}, Number<35>{});
    constexpr auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 117);
}

TEST(MakeNaiveTensorDescriptor, ColumnMajor)
{
    // 5x4 tensor with column-major strides [1, 5]
    // element_space_size = 1 + (5-1)*1 + (4-1)*5 = 1 + 4 + 15 = 20
    constexpr auto lengths = make_tuple(Number<5>{}, Number<4>{});
    constexpr auto strides = make_tuple(Number<1>{}, Number<5>{});
    constexpr auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 20);
}

// Test with runtime values (index_t instead of Number<>)
TEST(MakeNaiveTensorDescriptorRuntime, Simple2D)
{
    // 5x4 tensor with row-major strides
    const auto lengths = make_tuple(index_t{5}, index_t{4});
    const auto strides = make_tuple(index_t{4}, index_t{1});
    const auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 20);
}

TEST(MakeNaiveTensorDescriptorRuntime, WithPadding)
{
    // 11x7x3 tensor with arbitrary strides (using prime numbers)
    const auto lengths = make_tuple(index_t{11}, index_t{7}, index_t{3});
    const auto strides = make_tuple(index_t{2}, index_t{97}, index_t{23});
    const auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 649);
}
