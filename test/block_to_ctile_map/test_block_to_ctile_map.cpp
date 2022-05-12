#include <ck/config.hpp>
#include <ck/utility/block_to_ctile_map.hpp>
#include "gtest/gtest.h"
#include <iostream>
#include <vector>

using namespace ck;

static auto I0 = Number<0>{};
static auto I1 = Number<1>{};

TEST(BlockToCTileMap, TestBlockToCTileMap_M00_N00_M01_N01_DeviceCTileIndexCheck1)
{
    const index_t M         = 384;
    const index_t N         = 384;
    const index_t MPerBlock = 128;
    const index_t NPerBlock = 128;
    const index_t MBlock    = M / MPerBlock;
    const index_t NBlock    = N / NPerBlock;
    const index_t M01       = 4;
    const index_t N01       = 4;

    auto c_grid_desc_m_n = make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, I1));

    printf("(M, N, MPerBlock, NPerBlock, M01, N01) = (%d, %d, %d, %d, %d, %d)\n",
           M,
           N,
           MPerBlock,
           NPerBlock,
           M01,
           N01);

    BlockToCTileMap_M00_N00_M01_N01<MPerBlock, NPerBlock, decltype(c_grid_desc_m_n), true> tile_map(
        c_grid_desc_m_n, M01, N01);

    EXPECT_TRUE(tile_map.CheckValidity(c_grid_desc_m_n) == true);
    EXPECT_TRUE(tile_map.CalculateGridSize(c_grid_desc_m_n) == 16);

    // clang-format off
    std::vector<std::vector<int>> expected = {
        {0, 0, 1},
        {0, 1, 1},
        {0, 2, 1},
        {0, 3, 0},
        {1, 0, 1},
        {1, 1, 1},
        {1, 2, 1},
        {1, 3, 0},
        {2, 0, 1},
        {2, 1, 1},
        {2, 2, 1},
        {2, 3, 0},
        {3, 0, 0},
        {3, 1, 0},
        {3, 2, 0},
        {3, 3, 0}
    };
    // clang-format on

    for(index_t i = 0; i < tile_map.CalculateGridSize(c_grid_desc_m_n); i++)
    {
        auto m0n0_idx = tile_map.CalculateBottomIndex(make_multi_index(i));
        std::cout << "block_1d_id = " << i << ", m0, n0 = " << m0n0_idx[I0] << ", " << m0n0_idx[I1];
        std::cout << ", valid = " << tile_map.ValidCTileIndex(m0n0_idx, make_tuple(MBlock, NBlock))
                  << std::endl;
        bool equal =
            expected[i] ==
            std::vector<int>{m0n0_idx[I0],
                             m0n0_idx[I1],
                             tile_map.ValidCTileIndex(m0n0_idx, make_tuple(MBlock, NBlock))};
        EXPECT_TRUE(equal);
    }
}

TEST(BlockToCTileMap, TestBlockToCTileMap_M00_N00_M01_N01_DeviceCTileIndexCheck0)
{
    const index_t M         = 384;
    const index_t N         = 384;
    const index_t MPerBlock = 128;
    const index_t NPerBlock = 128;
    // const index_t MBlock    = M / MPerBlock;
    // const index_t NBlock    = N / NPerBlock;
    const index_t M01 = 4;
    const index_t N01 = 4;

    auto c_grid_desc_m_n = make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, I1));

    printf("(M, N, MPerBlock, NPerBlock, M01, N01) = (%d, %d, %d, %d, %d, %d)\n",
           M,
           N,
           MPerBlock,
           NPerBlock,
           M01,
           N01);

    BlockToCTileMap_M00_N00_M01_N01<MPerBlock, NPerBlock, decltype(c_grid_desc_m_n), false>
        tile_map(c_grid_desc_m_n, M01, N01);

    EXPECT_TRUE(tile_map.CheckValidity(c_grid_desc_m_n) == false);
}
