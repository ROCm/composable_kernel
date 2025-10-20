// Copyright © Advanced Micro Devices, Inc. or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"
#include "test_gemm_pipeline_util.hpp"

template <typename Tuple, typename Derived>
class TestCkTileGemmPipelineWmmaBase : public TestCkTileGemmPipeline<Tuple, Derived>
{
    public:
    static constexpr bool check_data_type()
    {
        using Base     = TestCkTileGemmPipeline<Tuple, Derived>;
        using DeviceIp = ck_tile::remove_cvref_t<decltype(ck_tile::get_device_arch())>;
        return ck_tile::has_wmma_traits_v<DeviceIp,
                                          typename Base::ADataType,
                                          typename Base::BDataType,
                                          typename Base::AccDataType,
                                          ck_tile::constant<Base::M_Warp_Tile>::value,
                                          ck_tile::constant<Base::N_Warp_Tile>::value,
                                          ck_tile::constant<Base::K_Warp_Tile>::value>;
    }
};
