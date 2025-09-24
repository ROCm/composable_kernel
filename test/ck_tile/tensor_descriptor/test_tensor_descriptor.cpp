// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/tensor/tensor_view.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"

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
    constexpr auto lengths = make_tuple(
        number<Gm>{},      // Number of block rows (r)
        number<Gm>{},      // Number of block columns (c) 
        number<MWidth>{},  // Rows within each block (m)
        number<NWidth>{}   // Columns within each block (n)
    );
    
 
    constexpr auto strides = make_tuple(
        number<Gm * MWidth * NWidth>{}, // Row stride
        number<1>{},                    // Column stride
        number<Gm>{},                   // Row within block stride    
        number<Gm * MWidth>{}           // Column within block stride
    );
    
    // Create the 4D tensor descriptor
    auto desc_4d = make_naive_tensor_descriptor(lengths, strides);
    return desc_4d;
}

void debug_print_explicit(const std::vector<int>& data, index_t MWidth, index_t NWidth, index_t Gm)
{
    std::cout << "Explicit Indexing:" << std::endl;
    for (int c = 0; c < Gm; ++c)
    {
        std::cout << "Col " << c << ": " << std::endl;
        for (int r = 0; r < Gm; ++r)
        {
            for (int m = 0; m < MWidth; ++m)
            {
                std::cout << "Row " << r << " (sub-row " << m << "): ";
                for (int n = 0; n < NWidth; ++n)
                {
                    int idx = c + Gm * m + Gm * MWidth * n + Gm * MWidth * NWidth * r; 
                    std::cout << data[idx] << " ";
                }
                if (MWidth > 1)
                {
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void debug_print_tensor_desc(const std::vector<int>& data, const auto& desc)
{
    const auto lengths = desc.get_lengths();
    std::cout << "Using Tensor Descriptor:" << std::endl;
    for (int c = 0; c < lengths[number<1>{}]; ++c)
    {
        std::cout << "Col " << c << ": " << std::endl;
        for (int r = 0; r < lengths[number<0>{}]; ++r)
        {
            for (int m = 0; m < lengths[number<2>{}]; ++m)
            {
                std::cout << "Row " << r << " (sub-row " << m << "): ";
                for (int n = 0; n < lengths[number<3>{}]; ++n)
                {
                    const auto block_coord = make_tuple(r, c, m, n);
                    const auto idx = desc.calculate_offset(block_coord);
                    std::cout << data[idx] << " ";
                }
                if (lengths[number<2>{}] > 1)
                {
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

TEST_F(TestTensorDescriptor, RowMajorBlocksWithColumnMajorData_1x4_blocks)
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

    debug_print_explicit(data, MWidth, NWidth, Gm);
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

    debug_print_tensor_desc(data, desc);
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

TEST_F(TestTensorDescriptor, RowMajorBlocksWithColumnMajorData_2x4_blocks)
{
    constexpr index_t MPerBlock = 4;
    constexpr index_t NPerBlock = 8;
    constexpr index_t Gm = 2;
    constexpr index_t MWidth = MPerBlock / Gm; 
    constexpr index_t NWidth = NPerBlock / Gm; 

    // This data represents a 4x8 matrix divided into 2x2 blocks of size 2x4 each
    // Block structure:
    // Block(0,0) | Block(0,1)
    // ----------------------
    // Block(1,0) | Block(1,1)
    std::vector<int> data {
        0,  4,  8,   12,  1,  5,  9, 13,    
        2,  6,  10,  14,  3,  7, 11, 15,     
        16, 20, 24, 28,  17, 21, 25, 29,
        18, 22, 26, 30,  19, 23, 27, 31
    };

    constexpr auto desc = make_blocked_tensor_descriptor<MPerBlock, NPerBlock, Gm>();

    debug_print_explicit(data, MWidth, NWidth, Gm);
    debug_print_tensor_desc(data, desc);
}

TEST_F(TestTensorDescriptor, GetSubBlockWithVectorizedAccess)
{
    constexpr index_t MPerBlock = 4;
    constexpr index_t NPerBlock = 8;
    constexpr index_t Gm = 2;
   
    // This data represents a 4x8 matrix divided into 2x2 blocks of size 2x4 each
    // Block structure:
    // Block(0,0) | Block(0,1)
    // ----------------------
    // Block(1,0) | Block(1,1)
    std::vector<int> data_vec {
        0,  4,  8,   12,  1,  5,  9, 13,    
        2,  6,  10,  14,  3,  7, 11, 15,     
        16, 20, 24, 28,  17, 21, 25, 29,
        18, 22, 26, 30,  19, 23, 27, 31
    };

    constexpr auto desc = make_blocked_tensor_descriptor<MPerBlock, NPerBlock, Gm>();

    const auto tensor_view = make_tensor_view(reinterpret_cast<int4*>(data_vec.data()), desc);
    
    const auto base_addr = make_multi_index(number<1>{}, number<1>{}, number<0>{}, number<0>{});
    const auto block_offset = make_tensor_coordinate(desc, base_addr); 

    // First row of sub-block (1,1)
    const auto row1 = tensor_view.get_vectorized_elements<int4>(block_offset, 0);
    EXPECT_EQ(row1.x, 20);
    EXPECT_EQ(row1.y, 21);
    EXPECT_EQ(row1.z, 22);
    EXPECT_EQ(row1.w, 23);

    // Second row of sub-block (1,1)
    const auto row2 = tensor_view.get_vectorized_elements<int4>(block_offset, 1);
    EXPECT_EQ(row2.x, 28);
    EXPECT_EQ(row2.y, 29);
    EXPECT_EQ(row2.z, 30);
    EXPECT_EQ(row2.w, 31);
}
