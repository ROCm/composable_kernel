// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "test_gemm_streamk_common_includes.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"

template <typename Tuple>
class TestCkTileStreamKWmma : public TestCkTileStreamK<Tuple>
{
    protected:
    void SetUp() override
    {
        using Base = TestCkTileStreamK<Tuple>;

        constexpr bool has_gfx11_wmma = ck_tile::has_wmma_traits_v<ck_tile::gfx11_t,
                                                                   typename Base::ADataType,
                                                                   typename Base::BDataType,
                                                                   typename Base::AccDataType,
                                                                   Base::M_Warp_Tile,
                                                                   Base::N_Warp_Tile,
                                                                   Base::K_Warp_Tile>;

        constexpr bool has_gfx12_wmma = ck_tile::has_wmma_traits_v<ck_tile::gfx12_t,
                                                                   typename Base::ADataType,
                                                                   typename Base::BDataType,
                                                                   typename Base::AccDataType,
                                                                   Base::M_Warp_Tile,
                                                                   Base::N_Warp_Tile,
                                                                   Base::K_Warp_Tile>;

        if constexpr(!has_gfx11_wmma && !has_gfx12_wmma)
        {
            GTEST_SKIP() << "Unsupported WMMA data type/tile combination";
        }
    }
};
