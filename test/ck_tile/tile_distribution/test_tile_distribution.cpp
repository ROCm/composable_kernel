#include <gtest/gtest.h>
#include <vector>

#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"

using namespace ck_tile;

class TestTileDistribution : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestTileDistribution, 4x4_matrix_2x2_blocks)
{
    constexpr index_t MRepeat = 1;
    constexpr index_t NRepeat = 1;
    constexpr index_t MWarpPerBlock = 1;
    constexpr index_t NWarpPerBlock = 1;
    constexpr index_t MThreadPerWarp = 2;
    constexpr index_t NThreadPerWarp = 2;
    constexpr index_t MVectorPerThread = 2;
    constexpr index_t NVectorPerThread = 2;

    // Tile distribution encoding for 4x4 matrix as 2x2 blocks
    constexpr auto matrix_4x4_dstr_encoding = tile_distribution_encoding<
        sequence<>,                                                      // No reduction dims
        tuple<
            sequence<MRepeat, MWarpPerBlock, MThreadPerWarp, MVectorPerThread>, 
            sequence<NRepeat, NWarpPerBlock, NThreadPerWarp, NVectorPerThread>>, 
        tuple<sequence<1, 2>, sequence<1,2>>,                           // 2D thread grid mapping
        tuple<sequence<1, 1>, sequence<2,2>>,                           // Warp arrangement
        sequence<1, 1, 2, 2>,                                           // Dimension order
        sequence<0, 3, 0, 3>>{};                                        // Each thread has 2x2 blocks.

    constexpr auto matrix_4x4_dstr = make_static_tile_distribution(matrix_4x4_dstr_encoding);

    EXPECT_EQ(matrix_4x4_dstr.get_num_of_dimension_x(), 2);
    EXPECT_EQ(matrix_4x4_dstr.get_num_of_dimension_y(), 2);
    EXPECT_EQ(matrix_4x4_dstr.get_num_of_dimension_p(), 1);
    EXPECT_EQ(matrix_4x4_dstr.get_num_of_dimension_r(), 0);

    const auto distributed_spans = matrix_4x4_dstr.get_distributed_spans();
    EXPECT_EQ(distributed_spans.size(), 2);
    EXPECT_EQ(distributed_spans[number<0>{}].impl_.size(), 1); // M dimension
    EXPECT_EQ(distributed_spans[number<1>{}].impl_.size(), 1); // N dimension
    EXPECT_EQ(distributed_spans[number<0>{}].impl_[0], 4); // M dimension
    EXPECT_EQ(distributed_spans[number<1>{}].impl_[0], 4); // N dimension
}
