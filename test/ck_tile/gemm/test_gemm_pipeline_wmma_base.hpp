#pragma once

#include "test_gemm_pipeline_util.hpp"

template <typename Tuple, typename Derived>
class TestCkTileGemmPipelineWmmaBase : public TestCkTileGemmPipeline<Tuple, Derived>
{
    public:
    template <typename ADataType,
              typename BDataType,
              typename AccDataType,
              ck_tile::index_t M_Warp_Tile,
              ck_tile::index_t N_Warp_Tile,
              ck_tile::index_t K_Warp_Tile>
    bool check_data_type_impl()
    {
        return ck_tile::check_wmma_supported<ADataType,
                                             BDataType,
                                             AccDataType,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile>();
    }
};
