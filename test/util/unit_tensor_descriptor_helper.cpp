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
    // 2x3x5 tensor with strides [1, 7, 35] - like a slice from a 7x7xN column-major tensor
    // This tests padding where there's space for extra elements
    // element_space_size = 1 + (2-1)*1 + (3-1)*7 + (5-1)*35 = 1 + 1 + 14 + 140 = 156
    constexpr auto lengths = make_tuple(Number<2>{}, Number<3>{}, Number<5>{});
    constexpr auto strides = make_tuple(Number<1>{}, Number<7>{}, Number<35>{});
    constexpr auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 156);
}

TEST(MakeNaiveTensorDescriptor, ColumnMajor)
{
    // 7x3 tensor with column-major strides [1, 7]
    // element_space_size = 1 + (7-1)*1 + (3-1)*7 = 1 + 6 + 14 = 21
    constexpr auto lengths = make_tuple(Number<7>{}, Number<3>{});
    constexpr auto strides = make_tuple(Number<1>{}, Number<7>{});
    constexpr auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 21);
}

// Test with runtime values (index_t instead of Number<>)
TEST(MakeNaiveTensorDescriptorRuntime, Simple2D)
{
    // 9x4 tensor with row-major strides
    // element_space_size = 1 + (9-1)*4 + (4-1)*1 = 1 + 32 + 3 = 36
    const auto lengths = make_tuple(index_t{9}, index_t{4});
    const auto strides = make_tuple(index_t{4}, index_t{1});
    const auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 36);
}

TEST(MakeNaiveTensorDescriptorRuntime, WithPadding)
{
    // 13x5x2 tensor with arbitrary strides (using prime numbers)
    // element_space_size = 1 + (13-1)*2 + (5-1)*97 + (2-1)*23 = 1 + 24 + 388 + 23 = 436
    const auto lengths = make_tuple(index_t{13}, index_t{5}, index_t{2});
    const auto strides = make_tuple(index_t{2}, index_t{97}, index_t{23});
    const auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 436);
}

// Test 1D tensors with explicit strides
TEST(MakeNaiveTensorDescriptor, ElementSpaceSize1D)
{
    // 13-element 1D tensor with stride 1
    constexpr auto lengths = make_tuple(Number<13>{});
    constexpr auto strides = make_tuple(Number<1>{});
    constexpr auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 13);
}

TEST(MakeNaiveTensorDescriptor, ElementSpaceSize1DStrided)
{
    // 7-element 1D tensor with stride 3 (every 3rd element)
    // element_space_size = 1 + (7-1)*3 = 19
    constexpr auto lengths = make_tuple(Number<7>{});
    constexpr auto strides = make_tuple(Number<3>{});
    constexpr auto desc    = make_naive_tensor_descriptor(lengths, strides);
    EXPECT_EQ(desc.GetElementSpaceSize(), 19);
}

// Test make_naive_tensor_descriptor_packed (contiguous layout)
// element_space_size = product of all lengths
TEST(MakeNaiveTensorDescriptorPacked, Simple1D)
{
    // 17-element packed tensor - element_space_size = 17
    constexpr auto lengths = make_tuple(Number<17>{});
    constexpr auto desc    = make_naive_tensor_descriptor_packed(lengths);
    EXPECT_EQ(desc.GetElementSpaceSize(), 17);
}

TEST(MakeNaiveTensorDescriptorPacked, Simple2D)
{
    // 6x5 packed tensor - element_space_size = 6*5 = 30
    constexpr auto lengths = make_tuple(Number<6>{}, Number<5>{});
    constexpr auto desc    = make_naive_tensor_descriptor_packed(lengths);
    EXPECT_EQ(desc.GetElementSpaceSize(), 30);
}

TEST(MakeNaiveTensorDescriptorPacked, Simple3D)
{
    // 4x5x9 packed tensor - element_space_size = 4*5*9 = 180
    constexpr auto lengths = make_tuple(Number<4>{}, Number<5>{}, Number<9>{});
    constexpr auto desc    = make_naive_tensor_descriptor_packed(lengths);
    EXPECT_EQ(desc.GetElementSpaceSize(), 180);
}

// Test make_naive_tensor_descriptor_aligned (stride alignment for memory access)
// Aligns the second-to-last stride to be a multiple of 'align'
TEST(MakeNaiveTensorDescriptorAligned, Align4)
{
    // 5x3 tensor aligned to 4 elements
    // strides[1] = 1, strides[0] = integer_least_multiple(3, 4) = 4
    // element_space_size = 1 + (5-1)*4 + (3-1)*1 = 1 + 16 + 2 = 19
    constexpr auto lengths = make_tuple(Number<5>{}, Number<3>{});
    constexpr auto desc    = make_naive_tensor_descriptor_aligned(lengths, Number<4>{});
    EXPECT_EQ(desc.GetElementSpaceSize(), 19);
}

TEST(MakeNaiveTensorDescriptorAligned, Align8)
{
    // 3x5x7 tensor aligned to 8 elements
    // strides[2] = 1, strides[1] = integer_least_multiple(7, 8) = 8, strides[0] = 5*8 = 40
    // element_space_size = 1 + (3-1)*40 + (5-1)*8 + (7-1)*1 = 1 + 80 + 32 + 6 = 119
    constexpr auto lengths = make_tuple(Number<3>{}, Number<5>{}, Number<7>{});
    constexpr auto desc    = make_naive_tensor_descriptor_aligned(lengths, Number<8>{});
    EXPECT_EQ(desc.GetElementSpaceSize(), 119);
}
