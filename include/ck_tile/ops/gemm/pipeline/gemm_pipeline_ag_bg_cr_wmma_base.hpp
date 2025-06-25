// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"

// this is used for double lds buffer and mixed with wmma policy such as transpose load and async
// load
namespace ck_tile {

template <typename Problem, typename Policy>
struct GemmPipelineAgBgCrWmmaImplBase : GemmPipelineAgBgCrImplBase<Problem, Policy>
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using ALayout        = remove_cvref_t<typename Problem::ALayout>;
    using BLayout        = remove_cvref_t<typename Problem::BLayout>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using I0 = number<0>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    template <typename ADramBlockWindowTmp, typename ALdsTensorView, typename ALdsLoadTileDistr>
    CK_TILE_DEVICE constexpr auto GetAWindows(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                              const ALdsTensorView& a_lds_block_view,
                                              const ALdsLoadTileDistr&) const
    {
        constexpr bool is_col_major = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;

        using YPerTile = std::conditional_t<is_col_major, number<KPerBlock>, number<MPerBlock>>;
        using XPerTile = std::conditional_t<is_col_major, number<MPerBlock>, number<KPerBlock>>;

        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(YPerTile{}, XPerTile{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        // A LDS tile window for store
        auto a_copy_lds_window = make_tile_window(
            a_lds_block_view, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});

        auto a_lds_gemm_window =
            make_tile_window(a_lds_block_view,
                             make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                             {0, 0},
                             ALdsLoadTileDistr{});

        return make_tuple(std::move(a_copy_dram_window),
                          std::move(a_copy_lds_window),
                          std::move(a_lds_gemm_window));
    }

    template <typename BDramBlockWindowTmp, typename BLdsTensorView, typename BLdsLoadTileDistr>
    CK_TILE_DEVICE constexpr auto GetBWindows(const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                              const BLdsTensorView& b_lds_block_view,
                                              const BLdsLoadTileDistr&) const
    {
        constexpr bool is_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

        using YPerTile = std::conditional_t<is_row_major, number<KPerBlock>, number<NPerBlock>>;
        using XPerTile = std::conditional_t<is_row_major, number<NPerBlock>, number<KPerBlock>>;

        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(YPerTile{}, XPerTile{}),
                             b_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // TODO: Do we really need those two tile windows???
        // They're exactly same...
        // B LDS tile window for store
        auto b_copy_lds_window = make_tile_window(
            b_lds_block_view, make_tuple(number<NPerBlock>{}, number<KPerBlock>{}), {0, 0});

        auto b_lds_gemm_window =
            make_tile_window(b_lds_block_view,
                             make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                             {0, 0},
                             BLdsLoadTileDistr{});

        return make_tuple(std::move(b_copy_dram_window),
                          std::move(b_copy_lds_window),
                          std::move(b_lds_gemm_window));
    }

    template <typename ADramBlockWindowTmp, typename ALdsTensorView>
    CK_TILE_DEVICE auto GetAMultiLdsWindows(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                            const ALdsTensorView& a_lds_block_view) const
    {
        constexpr bool is_col_major = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        using YPerTile = std::conditional_t<is_col_major, number<KPerBlock>, number<MPerBlock>>;
        using XPerTile = std::conditional_t<is_col_major, number<MPerBlock>, number<KPerBlock>>;
        using LdsTileWindow =
            std::conditional_t<is_col_major,
                               decltype(make_tuple(number<KPerBlock>{}, number<MPerBlock>{})),
                               decltype(make_tuple(number<MPerBlock>{}, number<KPerBlock>{}))>;
        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(YPerTile{}, XPerTile{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        //   LDS tiles window for store
        auto a_copy_lds_window = generate_tuple(
            [&](auto i_buf) {
                return make_tile_window(
                    a_lds_block_view, LdsTileWindow{}, {LdsTileWindow{}[I0{}] * i_buf, 0});
            },
            number<Policy::NumLdsNumA>{});

        auto a_lds_gemm_window = generate_tuple(
            [&](auto i_buf) {
                return make_tile_window(
                    a_lds_block_view, LdsTileWindow{}, {LdsTileWindow{}[I0{}] * i_buf, 0});
            },
            number<Policy::NumLdsNumA>{});

        return make_tuple(std::move(a_copy_dram_window),
                          std::move(a_copy_lds_window),
                          std::move(a_lds_gemm_window));
    }

    template <typename BDramBlockWindowTmp, typename BLdsTensorView>
    CK_TILE_DEVICE auto GetBMultiLdsWindows(const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                            const BLdsTensorView& b_lds_block_view) const
    {
        constexpr bool is_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
        using YPerTile = std::conditional_t<is_row_major, number<KPerBlock>, number<NPerBlock>>;
        using XPerTile = std::conditional_t<is_row_major, number<NPerBlock>, number<KPerBlock>>;
        using LdsTileWindow =
            std::conditional_t<is_row_major,
                               decltype(make_tuple(number<KPerBlock>{}, number<NPerBlock>{})),
                               decltype(make_tuple(number<NPerBlock>{}, number<KPerBlock>{}))>;
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(YPerTile{}, XPerTile{}),
                             b_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // TODO: Do we really need those two tile windows???
        // They're exactly same...
        // B LDS tile window for store
        auto b_copy_lds_window = generate_tuple(
            [&](auto i_buf) {
                return make_tile_window(
                    b_lds_block_view, LdsTileWindow{}, {LdsTileWindow{}[I0{}] * i_buf, 0});
            },
            number<Policy::NumLdsNumB>{});

        auto b_lds_gemm_window = generate_tuple(
            [&](auto i_buf) {
                return make_tile_window(
                    b_lds_block_view, LdsTileWindow{}, {LdsTileWindow{}[I0{}] * i_buf, 0});
            },
            number<Policy::NumLdsNumB>{});

        return make_tuple(std::move(b_copy_dram_window),
                          std::move(b_copy_lds_window),
                          std::move(b_lds_gemm_window));
    }
};

} // namespace ck_tile
