// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <vector>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"

using namespace ck_tile;

class TestCoordinateTransform : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestCoordinateTransform, Merge2Dto1D)
{
    constexpr auto I0 = number<0>{};
    constexpr auto I1 = number<1>{};
    
    // Test merging [4, 5] -> [20]
    constexpr index_t len0 = 4;
    constexpr index_t len1 = 5;
    const auto low_lengths = make_tuple(len0, len1);
    
    auto merge_transform = make_merge_transform(low_lengths);
    
    // Check upper dimension length
    const auto up_lengths = merge_transform.get_upper_lengths();
    EXPECT_EQ(up_lengths.size(), 1);
    EXPECT_EQ(up_lengths[I0], len0 * len1);  // Should be 20
    
    // Test coordinate mapping: idx_low = [i, j] -> idx_up = [i * len1 + j]
    // "Upper" index is the internal flattened index.
    // "Lower" index is the external multi-dimensional index.
    multi_index<2> idx_low{-1, -1};
    multi_index<1> idx_up;
    
    // Test case 1: [0] -> [0, 0]
    idx_up(I0) = 0;
    merge_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 0);
    EXPECT_EQ(idx_low[I1], 0);
    
    // Test case 2: [7] (1 * 5 + 2 = 7) -> [1, 2]
    idx_up(I0) = 7;
    merge_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 1);
    EXPECT_EQ(idx_low[I1], 2);
    
    // Test case 3: [19] (3 * 5 + 4 = 19) -> [3, 4]
    idx_up(I0) = 19;
    merge_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 3);
    EXPECT_EQ(idx_low[I1], 4);
}

TEST_F(TestCoordinateTransform, Merge3Dto1D)
{
    constexpr auto I0 = number<0>{};
    constexpr auto I1 = number<1>{};
    constexpr auto I2 = number<2>{};
    
    // Test merging [2, 3, 4] -> [24]
    constexpr index_t len0 = 2;
    constexpr index_t len1 = 3;
    constexpr index_t len2 = 4;
    const auto low_lengths = make_tuple(len0, len1, len2);
    
    auto merge_transform = make_merge_transform(low_lengths);
    
    // Check upper dimension length.
    const auto up_lengths = merge_transform.get_upper_lengths();
    EXPECT_EQ(up_lengths.size(), 1);
    EXPECT_EQ(up_lengths[I0], len0 * len1 * len2);  // Should be 24
    
    // Test coordinate mapping: idx_low = [i, j, k] -> idx_up = [i * len1 * len2 + j * len2 + k]
    multi_index<3> idx_low {-1, -1, -1};
    multi_index<1> idx_up;
    
    // Test case 1: [23] -> [1, 2, 3]
    idx_up[I0] = 23;
    merge_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 1);
    EXPECT_EQ(idx_low[I1], 2);
    EXPECT_EQ(idx_low[I2], 3);
    
    // Test case 2: [6] -> [0, 1, 2]
    idx_up[I0] = 6;
    merge_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 0);
    EXPECT_EQ(idx_low[I1], 1);
    EXPECT_EQ(idx_low[I2], 2);
    
    // Test case 3: [0] -> [0, 0, 0]
    idx_up[I0] = 0;
    merge_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 0);
    EXPECT_EQ(idx_low[I1], 0);
    EXPECT_EQ(idx_low[I2], 0);
}

// Test make_pad_transform with left and right padding
TEST_F(TestCoordinateTransform, Pad)
{
    constexpr auto I0 = number<0>{};
    
    // Test padding: low_length=10, left_pad=2, right_pad=3
    // Result: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] (length 15)
    // Valid indices in upper space: 2-11 map to 0-9 in lower space
    constexpr index_t low_length = 10;
    constexpr index_t left_pad = 2;
    constexpr index_t right_pad = 3;
    
    auto pad_transform = make_pad_transform(low_length, left_pad, right_pad);
    
    // Check upper dimension length
    const auto up_lengths = pad_transform.get_upper_lengths();
    EXPECT_EQ(up_lengths.size(), 1);
    EXPECT_EQ(up_lengths[I0], low_length + left_pad + right_pad);  // Should be 15
    
    // Test coordinate mapping: upper -> lower
    multi_index<1> idx_low{-1};
    multi_index<1> idx_up;
    
    // Test case 1: upper[2] -> lower[0] (first valid position)
    idx_up[I0] = 2;
    pad_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 0);
    
    // Test case 2: upper[5] -> lower[3]
    idx_up[I0] = 5;
    pad_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 3);
    
    // Test case 3: upper[11] -> lower[9] (last valid position)
    idx_up[I0] = 11;
    pad_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 9);
    
    // Test validity checks
    idx_up[I0] = 0;  // In left padding region
    EXPECT_FALSE(pad_transform.is_valid_upper_index_mapped_to_valid_lower_index(idx_up));
    
    idx_up[I0] = 1;  // In left padding region
    EXPECT_FALSE(pad_transform.is_valid_upper_index_mapped_to_valid_lower_index(idx_up));
    
    idx_up[I0] = 2;  // First valid position
    EXPECT_TRUE(pad_transform.is_valid_upper_index_mapped_to_valid_lower_index(idx_up));
    
    idx_up[I0] = 11;  // Last valid position
    EXPECT_TRUE(pad_transform.is_valid_upper_index_mapped_to_valid_lower_index(idx_up));
    
    idx_up[I0] = 12;  // In right padding region
    EXPECT_FALSE(pad_transform.is_valid_upper_index_mapped_to_valid_lower_index(idx_up));
    
    idx_up[I0] = 14;  // In right padding region
    EXPECT_FALSE(pad_transform.is_valid_upper_index_mapped_to_valid_lower_index(idx_up));
}

// Test make_embed_transform for convolution im2col pattern
TEST_F(TestCoordinateTransform, Embed2Dto1D)
{
    constexpr auto I0 = number<0>{};
    constexpr auto I1 = number<1>{};
    
    // Simulate im2col for 1D convolution:
    // input_width=8, filter_width=3, stride=2, output_width=3
    // upper dims: [filter_idx, output_idx] -> lower: input_idx = filter_idx * dilation + output_idx * stride
    const auto up_lengths = make_tuple(4, 5);       // [filter_width, output_width]
    const auto coefficients = make_tuple(2, 3);     // [dilation=2, stride=3]
    
    auto embed_transform = make_embed_transform(up_lengths, coefficients);
    
    // Test coordinate mapping: upper -> lower
    multi_index<1> idx_low{-1};
    multi_index<2> idx_up;
    
    // Test sliding window positions
    // Filter position 0, Output position 0: input[0]
    idx_up[I0] = 0; idx_up[I1] = 0;
    embed_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 0);
    
    // Filter position 1, Output position 0: input[2]
    idx_up[I0] = 1; idx_up[I1] = 0;
    embed_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 2);
    
    // Filter position 2, Output position 0: input[4]
    idx_up[I0] = 2; idx_up[I1] = 0;
    embed_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 4);
    
    // Filter position 0, Output position 1: input[3]
    idx_up[I0] = 0; idx_up[I1] = 1;
    embed_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 3);
    
    // Filter position 1, Output position 1: input[5]
    idx_up[I0] = 1; idx_up[I1] = 1;
    embed_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 5);
    
    // Filter position 2, Output position 1: input[7]
    idx_up[I0] = 2; idx_up[I1] = 1;
    embed_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 7);
    
    // Filter position 0, Output position 2: input[6]
    idx_up[I0] = 0; idx_up[I1] = 2;
    embed_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 6);
    
    // Filter position 2, Output position 2: input[10]
    idx_up[I0] = 2; idx_up[I1] = 2;
    embed_transform.calculate_lower_index(idx_low, idx_up);
    EXPECT_EQ(idx_low[I0], 10);
}
