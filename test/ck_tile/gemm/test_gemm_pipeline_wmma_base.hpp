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
        if(ck_tile::is_gfx12_supported())
        {
            return ck_tile::has_wmma_traits_v<ck_tile::gfx12_t,
                                              ADataType,
                                              BDataType,
                                              AccDataType,
                                              M_Warp_Tile,
                                              N_Warp_Tile,
                                              K_Warp_Tile>;
        }
        else if(ck_tile::is_gfx11_supported())
        {
            return ck_tile::has_wmma_traits_v<ck_tile::gfx11_t,
                                              ADataType,
                                              BDataType,
                                              AccDataType,
                                              M_Warp_Tile,
                                              N_Warp_Tile,
                                              K_Warp_Tile>;
        }
        else
        {
            return false;
        }
    }
};
