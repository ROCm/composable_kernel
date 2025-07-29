// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

template <typename Problem, typename Policy>
struct GemmPipelineAgBgCrImplBase
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using ALayout        = remove_cvref_t<typename Problem::ALayout>;
    using BLayout        = remove_cvref_t<typename Problem::BLayout>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;
#if defined(__gfx950__)
    static constexpr bool is_a_load_tr = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
    static constexpr bool is_b_load_tr = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
#else
    static constexpr bool is_a_load_tr = false;
    static constexpr bool is_b_load_tr = false;
#endif

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    template <typename DstBlockTile, typename SrcTileWindow, typename DramTileWindowStep>
    CK_TILE_DEVICE void GlobalPrefetch(DstBlockTile& dst_block_tile,
                                       SrcTileWindow& dram_tile_window,
                                       const DramTileWindowStep& dram_tile_window_step) const
    {
        load_tile(dst_block_tile, dram_tile_window);
        move_tile_window(dram_tile_window, dram_tile_window_step);
    }

    template <typename DstBlockWindow, typename SrcTileWindow, typename DramTileWindowStep>
    CK_TILE_DEVICE void GlobalPrefetchAsync(DstBlockWindow& dst_block_window,
                                            SrcTileWindow& dram_tile_window,
                                            const DramTileWindowStep& dram_tile_window_step) const
    {
        async_load_tile(dst_block_window, dram_tile_window);
        move_tile_window(dram_tile_window, dram_tile_window_step);
    }

    template <typename DstTileWindow, typename SrcBlockTile, typename ElementFunction>
    CK_TILE_DEVICE void LocalPrefill(DstTileWindow& lds_tile_window,
                                     const SrcBlockTile& src_block_tile,
                                     const ElementFunction& element_func) const
    {
        const auto block_tile = tile_elementwise_in(element_func, src_block_tile);
        store_tile(lds_tile_window, block_tile);
    }

    template <typename DstBlockTile, typename SrcTileWindow, bool LoadTranspose = false>
    CK_TILE_DEVICE void LocalPrefetch(DstBlockTile& dst_block_tile,
                                      const SrcTileWindow& lds_tile_window,
                                      bool_constant<LoadTranspose> = {}) const
    {
        if constexpr(LoadTranspose)
            dst_block_tile = load_tile_transpose(lds_tile_window);
        else
            load_tile(dst_block_tile, lds_tile_window);
    }

    CK_TILE_DEVICE auto GetABLdsTensorViews(void* p_smem) const
    {
        // A tile in LDS
        ADataType* __restrict__ p_a_lds = static_cast<ADataType*>(p_smem);
        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        // TODO: LDS alignment should come from Policy!
        constexpr index_t a_lds_block_space_size_aligned =
            Problem::SkipALds
                ? 0
                : integer_least_multiple(
                      sizeof(ADataType) * a_lds_block_desc.get_element_space_size(), 16);

        // B tile in LDS
        BDataType* __restrict__ p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));
        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
        auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

        return make_tuple(std::move(a_lds_block), std::move(b_lds_block));
    }

    template <typename ADramBlockWindow, typename ALdsTensorView, typename ALoadTileDistr>
    CK_TILE_DEVICE constexpr auto GetAWindows(const ADramBlockWindow& a_dram_block_window,
                                              const ALdsTensorView& a_lds_block_view,
                                              const ALoadTileDistr&,
                                              const array<index_t, 2>& offset = {0, 0}) const
    {
        constexpr bool is_col_major = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;

        using YPerTile = std::conditional_t<is_col_major, number<KPerBlock>, number<MPerBlock>>;
        using XPerTile = std::conditional_t<is_col_major, number<MPerBlock>, number<KPerBlock>>;

        // A DRAM tile window for load
        decltype(auto) a_copy_dram_window = [&] {
            if constexpr(Problem::SkipALds == false)
            {
                return make_tile_window(a_dram_block_window.get_bottom_tensor_view(),
                                        make_tuple(YPerTile{}, XPerTile{}),
                                        a_dram_block_window.get_window_origin() + offset,
                                        Policy::template MakeADramTileDistribution<Problem>());
            }
            else
            {
                return make_tile_window(a_dram_block_window.get_bottom_tensor_view(),
                                        make_tuple(YPerTile{}, XPerTile{}),
                                        a_dram_block_window.get_window_origin() + offset,
                                        ALoadTileDistr{});
            }
        }();

        // A LDS tile window for store
        auto a_lds_shape = []() {
            if constexpr(is_a_load_tr)
                return make_tuple(number<KPerBlock>{}, number<MPerBlock>{});
            else
                return make_tuple(number<MPerBlock>{}, number<KPerBlock>{});
        }();
        auto a_copy_lds_window = make_tile_window(a_lds_block_view, a_lds_shape, {0, 0});

        auto a_lds_load_tile_distr = []() {
            if constexpr(is_a_load_tr)
                return make_static_tile_distribution(
                    typename InputTileDistributionTraits<
                        typename ALoadTileDistr::DstrEncode,
                        typename Problem::ADataType>::TransposedDstrEncode{});
            else
                return ALoadTileDistr{};
        }();
        auto a_lds_gemm_window =
            make_tile_window(a_lds_block_view, a_lds_shape, {0, 0}, a_lds_load_tile_distr);

        return make_tuple(std::move(a_copy_dram_window),
                          std::move(a_copy_lds_window),
                          std::move(a_lds_gemm_window));
    }

    template <typename BDramBlockWindow, typename BLdsTensorView, typename BLoadTileDistr>
    CK_TILE_DEVICE constexpr auto GetBWindows(const BDramBlockWindow& b_dram_block_window,
                                              const BLdsTensorView& b_lds_block_view,
                                              const BLoadTileDistr&,
                                              const array<index_t, 2>& offset = {0, 0}) const
    {
        constexpr bool is_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

        using YPerTile = std::conditional_t<is_row_major, number<KPerBlock>, number<NPerBlock>>;
        using XPerTile = std::conditional_t<is_row_major, number<NPerBlock>, number<KPerBlock>>;

        decltype(auto) b_copy_dram_window = [&] {
            if constexpr(Problem::SkipBLds == false)
            {
                return make_tile_window(b_dram_block_window.get_bottom_tensor_view(),
                                        make_tuple(YPerTile{}, XPerTile{}),
                                        b_dram_block_window.get_window_origin() + offset,
                                        Policy::template MakeBDramTileDistribution<Problem>());
            }
            else
            {
                return make_tile_window(b_dram_block_window.get_bottom_tensor_view(),
                                        make_tuple(YPerTile{}, XPerTile{}),
                                        b_dram_block_window.get_window_origin() + offset,
                                        BLoadTileDistr{});
            }
        }();

        // TODO: Do we really need those two tile windows???
        // They're exactly same...
        // B LDS tile window for store
        auto b_lds_shape = []() {
            if constexpr(is_b_load_tr)
                return make_tuple(number<KPerBlock>{}, number<NPerBlock>{});
            else
                return make_tuple(number<NPerBlock>{}, number<KPerBlock>{});
        }();
        auto b_copy_lds_window = make_tile_window(b_lds_block_view, b_lds_shape, {0, 0});

        auto b_lds_load_tile_distr = []() {
            if constexpr(is_b_load_tr)
                return make_static_tile_distribution(
                    typename InputTileDistributionTraits<
                        typename BLoadTileDistr::DstrEncode,
                        typename Problem::BDataType>::TransposedDstrEncode{});
            else
                return BLoadTileDistr{};
        }();
        auto b_lds_gemm_window =
            make_tile_window(b_lds_block_view, b_lds_shape, {0, 0}, b_lds_load_tile_distr);

        return make_tuple(std::move(b_copy_dram_window),
                          std::move(b_copy_lds_window),
                          std::move(b_lds_gemm_window));
    }
};

} // namespace ck_tile
