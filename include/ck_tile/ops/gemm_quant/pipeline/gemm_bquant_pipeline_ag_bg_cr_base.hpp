// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"

namespace ck_tile {

template <typename Problem, typename Policy>
struct GemmBQuantPipelineAgBgCrImplBase : public GemmPipelineAgBgCrImplBase<Problem, Policy>
{
    using Base           = GemmPipelineAgBgCrImplBase<Problem, Policy>;
    using ADataType      = typename Base::ADataType;
    using ALayout        = typename Base::ALayout;
    using BDataType      = typename Base::BDataType;
    using BLayout        = typename Base::BLayout;
    using BlockGemmShape = typename Base::BlockGemmShape;

    using BQLayout = remove_cvref_t<typename Problem::BQLayout>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t QuantGroupSize = Problem::kQuantGroupSize;
    static constexpr index_t KPerBlockBQ    = KPerBlock / QuantGroupSize;

    static_assert(KPerBlock % QuantGroupSize == 0,
                  "KPerBlock must be a multiple of QuantGroupSize");

    // Create DRAM tile window for BQ
    template <typename BQDramBlockWindowTmp>
    CK_TILE_DEVICE constexpr auto
    GetBQDramLoadWindow(const BQDramBlockWindowTmp& bq_dram_block_window_tmp) const
    {
        static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>);

        using YPerTile = number<NPerBlock>;
        using XPerTile = number<KPerBlockBQ>;

        auto bq_copy_dram_window =
            make_tile_window(bq_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(YPerTile(), XPerTile()),
                             bq_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBQDramTileDistribution<Problem>());
        return bq_copy_dram_window;
    }
};

} // namespace ck_tile
