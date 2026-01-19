// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

using namespace ck;

// Test compute_element_space_size helper directly
// Formula: 1 + sum((length[i] - 1) * stride[i])
TEST(ComputeElementSpaceSize, Simple2D)
{
    // 4x4 tensor with row-major strides [4, 1]
    // element_space_size = 1 + (4-1)*4 + (4-1)*1 = 1 + 12 + 3 = 16
    constexpr auto lengths = make_tuple(Number<4>{}, Number<4>{});
    constexpr auto strides = make_tuple(Number<4>{}, Number<1>{});
    constexpr auto result  = detail::compute_element_space_size(lengths, strides, Sequence<0, 1>{});
    EXPECT_EQ(result, 16);
}

TEST(ComputeElementSpaceSize, Simple3D)
{
    // 2x3x4 tensor with strides [12, 4, 1]
    // element_space_size = 1 + (2-1)*12 + (3-1)*4 + (4-1)*1 = 1 + 12 + 8 + 3 = 24
    constexpr auto lengths = make_tuple(Number<2>{}, Number<3>{}, Number<4>{});
    constexpr auto strides = make_tuple(Number<12>{}, Number<4>{}, Number<1>{});
    constexpr auto result =
        detail::compute_element_space_size(lengths, strides, Sequence<0, 1, 2>{});
    EXPECT_EQ(result, 24);
}

TEST(ComputeElementSpaceSize, WithPadding)
{
    // 4x4 tensor with padded strides [8, 1] (8 elements per row instead of 4)
    // element_space_size = 1 + (4-1)*8 + (4-1)*1 = 1 + 24 + 3 = 28
    constexpr auto lengths = make_tuple(Number<4>{}, Number<4>{});
    constexpr auto strides = make_tuple(Number<8>{}, Number<1>{});
    constexpr auto result  = detail::compute_element_space_size(lengths, strides, Sequence<0, 1>{});
    EXPECT_EQ(result, 28);
}

TEST(ComputeElementSpaceSize, ColumnMajor)
{
    // 4x4 tensor with column-major strides [1, 4]
    // element_space_size = 1 + (4-1)*1 + (4-1)*4 = 1 + 3 + 12 = 16
    constexpr auto lengths = make_tuple(Number<4>{}, Number<4>{});
    constexpr auto strides = make_tuple(Number<1>{}, Number<4>{});
    constexpr auto result  = detail::compute_element_space_size(lengths, strides, Sequence<0, 1>{});
    EXPECT_EQ(result, 16);
}

TEST(ComputeElementSpaceSize, SingleDimension)
{
    // 10 element 1D tensor with stride 1
    // element_space_size = 1 + (10-1)*1 = 10
    constexpr auto lengths = make_tuple(Number<10>{});
    constexpr auto strides = make_tuple(Number<1>{});
    constexpr auto result  = detail::compute_element_space_size(lengths, strides, Sequence<0>{});
    EXPECT_EQ(result, 10);
}

TEST(ComputeElementSpaceSize, SingleElement)
{
    // 1x1 tensor - should have size 1
    // element_space_size = 1 + (1-1)*1 + (1-1)*1 = 1
    constexpr auto lengths = make_tuple(Number<1>{}, Number<1>{});
    constexpr auto strides = make_tuple(Number<1>{}, Number<1>{});
    constexpr auto result  = detail::compute_element_space_size(lengths, strides, Sequence<0, 1>{});
    EXPECT_EQ(result, 1);
}

TEST(ComputeElementSpaceSize, BroadcastDimension)
{
    // 8x5 tensor with broadcast on first dimension (stride 0)
    // element_space_size = 1 + (8-1)*0 + (5-1)*1 = 1 + 0 + 4 = 5
    constexpr auto lengths = make_tuple(Number<8>{}, Number<5>{});
    constexpr auto strides = make_tuple(Number<0>{}, Number<1>{});
    constexpr auto result  = detail::compute_element_space_size(lengths, strides, Sequence<0, 1>{});
    EXPECT_EQ(result, 5);
}

// Test make_naive_tensor_descriptor uses compute_element_space_size correctly
TEST(MakeNaiveTensorDescriptor, ElementSpaceSize2D)
{
    constexpr auto lengths = make_tuple(Number<4>{}, Number<4>{});
    constexpr auto strides = make_tuple(Number<4>{}, Number<1>{});
    constexpr auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 16);
}

TEST(MakeNaiveTensorDescriptor, ElementSpaceSize3D)
{
    constexpr auto lengths = make_tuple(Number<2>{}, Number<3>{}, Number<4>{});
    constexpr auto strides = make_tuple(Number<12>{}, Number<4>{}, Number<1>{});
    constexpr auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 24);
}

TEST(MakeNaiveTensorDescriptor, BroadcastDimension)
{
    // 8x5 tensor with broadcast on first dimension (stride 0)
    // element_space_size = 1 + (8-1)*0 + (5-1)*1 = 5
    constexpr auto lengths = make_tuple(Number<8>{}, Number<5>{});
    constexpr auto strides = make_tuple(Number<0>{}, Number<1>{});
    constexpr auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 5);
}

// Test with runtime values (index_t instead of Number<>)
TEST(ComputeElementSpaceSizeRuntime, Simple2D)
{
    const auto lengths = make_tuple(index_t{4}, index_t{4});
    const auto strides = make_tuple(index_t{4}, index_t{1});
    const auto result  = detail::compute_element_space_size(lengths, strides, Sequence<0, 1>{});
    EXPECT_EQ(result, 16);
}

TEST(ComputeElementSpaceSizeRuntime, WithPadding)
{
    const auto lengths = make_tuple(index_t{4}, index_t{4});
    const auto strides = make_tuple(index_t{8}, index_t{1});
    const auto result  = detail::compute_element_space_size(lengths, strides, Sequence<0, 1>{});
    EXPECT_EQ(result, 28);
}
