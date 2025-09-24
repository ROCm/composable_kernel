// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"

using namespace ck_tile;

class TestTensorDescriptor : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

template <index_t MPerBlock, index_t NPerBlock, index_t Gm>
constexpr auto make_blocked_tensor_descriptor()
{
    constexpr index_t MWidth = MPerBlock / Gm;
    constexpr index_t NWidth = NPerBlock / Gm;
    
    // Create a 4D tensor descriptor: [Gm_row, Gm_col, MWidth, NWidth]
    constexpr auto logical_lengths = make_tuple(
        number<Gm>{},      // Number of block rows
        number<Gm>{},      // Number of block columns 
        number<MWidth>{},  // Rows within each block
        number<NWidth>{}   // Columns within each block
    );
    
    // The strides correspond to indexing formula:
    // idx = (col * NWidth + n_loc) * MPerBlock + row
    constexpr auto logical_strides = make_tuple(
        number<NWidth * MPerBlock>{},  // col stride: NWidth * MPerBlock
        number<1>{},                   // Block column stride
        number<1>{},                   // row stride: 1 (fastest changing)
        number<MPerBlock>{}            // n_loc stride: MPerBlock
    );
    
    // Create the 4D tensor descriptor
    auto desc_4d = make_naive_tensor_descriptor(logical_lengths, logical_strides);
    
    // Transform to 2D by merging dimensions: [Gm_row*MWidth, Gm_col*NWidth]
    // This gives you the original MPerBlock × NPerBlock view
    // return transform_tensor_descriptor(
    //     desc_4d,
    //     make_tuple(make_merge_transform(make_tuple(number<Gm>{}, number<MWidth>{}))),
    //     make_tuple(sequence<0, 2>{}),  // Merge dims 0,2 from 4D descriptor
    //     make_tuple(sequence<0>{})      // Create new dimension 0
    // );
    return desc_4d;
}

TEST_F(TestTensorDescriptor, RowMajorBlocksWithColumnMajorData)
{
    constexpr index_t MPerBlock = 2;
    constexpr index_t NPerBlock = 8;
    constexpr index_t Gm = 2; // Number of blocks in each dimension
    constexpr index_t MWidth = MPerBlock / Gm;
    constexpr index_t NWidth = NPerBlock / Gm;

    // This data represents data in 2x2 block matric with 1x4 blocks.
    // 0 1 2  3  | 4  5  6  7
    // -----------------------
    // 8 9 10 11 | 12 13 14 15 
    std::vector<int> data {
        0,  4,  1,  5,  2,  6,  3,  7,    // Col 0: interleaved rows
        8,  12, 9,  13, 10, 14, 11, 15    // Col 1: interleaved rows
    };

    constexpr auto desc = make_blocked_tensor_descriptor<MPerBlock, NPerBlock, Gm>();

    //Print using explicit indexing
    std::cout << "Explicit Indexing:" << std::endl;
    for (int col = 0; col < Gm; ++col)
    {
        std::cout << "Col " << col << ": " << std::endl;
        for (int row = 0; row < Gm; ++row)
        {
            std::cout << "Row " << row << ": ";
            for (int n_loc = 0; n_loc < NWidth; ++n_loc)
            {
                int idx = (row * NWidth + n_loc) * MPerBlock + col;
                std::cout << data[idx] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::vector<int> data_explicit_indexing(MPerBlock * NPerBlock, -1);
    for (int col = 0; col < Gm; ++col)
    {
        for (int row = 0; row < Gm; ++row)
        {
            for (int n_loc = 0; n_loc < NWidth; ++n_loc)
            {
                int idx = (row * NWidth + n_loc) * MPerBlock + col;
                data_explicit_indexing[idx] = data[idx];
            }
        }
    }

    // Print using tensor descriptor
    std::cout << "Using Tensor Descriptor:" << std::endl;
    for (int col = 0; col < Gm; ++col)
    {
        std::cout << "Col " << col << ": " << std::endl;
        for (int row = 0; row < Gm; ++row)
        {
            std::cout << "Row " << row << ": ";
            // Column-major ordering within the block
            for (int n = 0; n < NWidth; ++n)
            {
                for (int m = 0; m < MWidth; ++m)
                {
                    const auto block_coord = make_tuple(row, col, m, n);
                    const auto idx = desc.calculate_offset(block_coord);
                    std::cout << data[idx] << " ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::vector<int> data_tensor_desc(MPerBlock * NPerBlock, -1);
    for (int col = 0; col < Gm; ++col)
    {
        for (int row = 0; row < Gm; ++row)
        {
            for (int n = 0; n < NWidth; ++n)
            {
                for (int m = 0; m < MWidth; ++m)
                {
                    const auto block_coord = make_tuple(row, col, m, n);
                    const auto idx = desc.calculate_offset(block_coord);
                    data_tensor_desc[idx] = data[idx];
                }
            }
        }
    }

    // Verify both methods yield the same result
    EXPECT_EQ(data_explicit_indexing, data_tensor_desc);
}
