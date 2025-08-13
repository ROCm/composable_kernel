// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"

namespace ck_tile {

template <typename Problem, typename Policy>
struct GemmMXPipelineAgBgCrImplBase : public GemmPipelineAgBgCrImplBase<Problem, Policy>
{
    using Base           = GemmPipelineAgBgCrImplBase<Problem, Policy>;
    using ADataType      = typename Base::ADataType;
    using ALayout        = typename Base::ALayout;
    using BDataType      = typename Base::BDataType;
    using BLayout        = typename Base::BLayout;
    using BlockGemmShape = typename Base::BlockGemmShape;

    using AScaleLayout = remove_cvref_t<typename Problem::AScaleLayout>;
    using BScaleLayout = remove_cvref_t<typename Problem::BScaleLayout>;

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t BlockScaleSize = Problem::kBlockScaleSize;

    static_assert(KPerBlock * % BlockScaleSize == 0,
                  "KPerBlock must be a multiple of BlockScaleSize");

    static constexpr auto MXdlPack = 2;
    static constexpr auto NXdlPack = 2;
    static constexpr auto KXdlPack = 2;

    // Create DRAM tile window for A scale
    template <typename AScaleDramBlockWindowTmp>
    CK_TILE_DEVICE constexpr auto
    GetAScaleDramLoadWindow(const AScaleDramBlockWindowTmp& a_scale_dram_block_window_tmp) const
    {
        static_assert(
            std::is_same_v<typename Problem::AScaleLayout, tensor_layout::gemm::RowMajor>);
        using YPerTile = number<MPerBlock / MXdlPack>;
        using XPerTile = number<KPerBlock * APackedSize / (BlockScaleSize * KXdlPack)>;

        auto a_copy_draw_window =
            make_tile_window(a_scale_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(YPerTile(), XPerTile()),
                             a_scale_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeAScaleDramTileDistribution<Problem>());
        return a_copy_draw_window;
    }

    // Create DRAM tile window for B scale
    template <typename BScaleDramBlockWindowTmp>
    CK_TILE_DEVICE constexpr auto
    GetBScaleDramLoadWindow(const BScaleDramBlockWindowTmp& b_scale_dram_block_window_tmp) const
    {
        static_assert(
            std::is_same_v<typename Problem::BScaleLayout, tensor_layout::gemm::ColumnMajor>);
        using YPerTile = number<NPerBlock / NXdlPack>;
        using XPerTile = number<KPerBlock * BPackedSize / (BlockScaleSize * KXdlPack)>;

        auto b_copy_draw_window =
            make_tile_window(b_scale_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(YPerTile(), XPerTile()),
                             b_scale_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBScaleDramTileDistribution<Problem>());
    }
};

} // namespace ck_tile
