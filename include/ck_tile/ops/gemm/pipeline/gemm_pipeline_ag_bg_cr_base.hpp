// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

template <typename Policy, typename Problem, typename = void>
struct has_get_pipeline_subtile_params : std::false_type
{
};

template <typename Policy, typename Problem>
struct has_get_pipeline_subtile_params<
    Policy,
    Problem,
    std::void_t<decltype(Policy::template GetPipelineSubTileNum<Problem>())>> : std::true_type
{
};

template <typename Problem, typename Policy>
struct GemmPipelineAgBgCrImplBase
{
    using AsDataType     = problem_as_data_type_t<Problem>;
    using BsDataType     = problem_bs_data_type_t<Problem>;
    using AsLayout       = problem_as_layout_t<Problem>;
    using BsLayout       = problem_bs_layout_t<Problem>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using ADataType   = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType>>;
    using ALayout     = remove_cvref_t<std::tuple_element_t<number<0>{}, AsLayout>>;
    using BInDataType = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataType>>;

    template <typename T>
    using has_bcastpolicy_type = decltype(T::BCastPolicy);

    static constexpr bool IsBCastPolicyBeforeLDSWrite = [] {
        if constexpr(is_detected<has_bcastpolicy_type, Problem>{})
        {
            return Problem::BCastPolicy == CastPolicy::BeforeLDSWrite;
        }
        else
        {
            return false;
        }
    }();

    using BDataType = std::conditional_t<IsBCastPolicyBeforeLDSWrite, ADataType, BInDataType>;

    using BLayout = remove_cvref_t<std::tuple_element_t<number<0>{}, BsLayout>>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    // Delegate to Policy's single definition to avoid duplication
    static constexpr bool is_a_load_tr = Policy::template is_a_load_tr<Problem>;
    static constexpr bool is_b_load_tr = Policy::template is_b_load_tr<Problem>;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    template <index_t UnaryOpSize = 8,
              typename DstBlockTile,
              typename SrcTileWindow,
              typename DramTileWindowStep>
    CK_TILE_DEVICE void GlobalPrefetch(DstBlockTile& dst_block_tile,
                                       SrcTileWindow& dram_tile_window,
                                       const DramTileWindowStep& dram_tile_window_step) const
    {
        load_and_convert_tile<UnaryOpSize>(dst_block_tile, dram_tile_window);
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

    template <typename TDMConfig_,
              typename DstBlockWindow,
              typename SrcTileWindow,
              typename DramTileWindowStep>
    CK_TILE_DEVICE void GlobalPrefetchTDM(const TDMConfig_& tdm_config,
                                          DstBlockWindow& dst_block_window,
                                          SrcTileWindow& dram_tile_window,
                                          const DramTileWindowStep& dram_tile_window_step) const
    {
        load_tile_tdm(tdm_config, dst_block_window, dram_tile_window);
        move_tile_window(dram_tile_window, dram_tile_window_step);
    }

    // overload without dram_tile_window_step
    template <typename TDMConfig_, typename DstBlockWindow, typename SrcTileWindow>
    CK_TILE_DEVICE void GlobalPrefetchTDM(const TDMConfig_& tdm_config,
                                          DstBlockWindow& dst_block_window,
                                          SrcTileWindow& dram_tile_window) const
    {
        load_tile_tdm(tdm_config, dst_block_window, dram_tile_window);
    }

    template <typename DstTileWindow, typename SrcBlockTile, typename ElementFunction>
    CK_TILE_DEVICE void LocalPrefill(DstTileWindow& lds_tile_window,
                                     const SrcBlockTile& src_block_tile,
                                     const ElementFunction& element_func) const
    {
        const auto block_tile_tmp = tile_elementwise_in(element_func, src_block_tile);
        store_tile(lds_tile_window, block_tile_tmp);
    }

    template <typename DstTileWindow, typename SrcBlockTile>
    CK_TILE_DEVICE void LocalPrefill(DstTileWindow& lds_tile_window,
                                     const SrcBlockTile& src_block_tile) const
    {
        store_tile(lds_tile_window, src_block_tile);
    }

    template <typename DstBlockTile, typename SrcTileWindow, bool LoadTranspose = false>
    CK_TILE_DEVICE void LocalPrefetch(DstBlockTile& dst_block_tile,
                                      const SrcTileWindow& lds_tile_window,
                                      bool_constant<LoadTranspose> = {}) const
    {
        if constexpr(LoadTranspose)
            load_tile_transpose(dst_block_tile, lds_tile_window);
        else
            load_tile(dst_block_tile, lds_tile_window);
    }

    CK_TILE_DEVICE auto GetABLdsTensorViews(void* p_smem) const
    {
        using ALdsType = typename Policy::template ALdsDataType_<Problem>;
        using BLdsType = typename Policy::template BLdsDataType_<Problem>;
        // A tile in LDS
        ALdsType* __restrict__ p_a_lds  = static_cast<ALdsType*>(p_smem);
        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        constexpr index_t APackedSize =
            ck_tile::numeric_traits<remove_cvref_t<ALdsType>>::PackedSize;

        // TODO: LDS alignment should come from Policy!
        constexpr index_t a_lds_block_space_size_aligned = integer_least_multiple(
            sizeof(ALdsType) * a_lds_block_desc.get_element_space_size() / APackedSize, 16);

        // B tile in LDS
        BLdsType* __restrict__ p_b_lds = static_cast<BLdsType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));
        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
        auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

        return make_tuple(std::move(a_lds_block), std::move(b_lds_block));
    }

    // this is used in gfx1250 to avoid lds partition conflict
    template <index_t num_lds_buffers>
    CK_TILE_DEVICE auto GetABLdsTensorViews(void* p_smem) const
    {
        using ALdsType                  = typename Policy::template ALdsDataType_<Problem>;
        using BLdsType                  = typename Policy::template BLdsDataType_<Problem>;
        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();

        constexpr index_t APackedSize =
            ck_tile::numeric_traits<remove_cvref_t<ALdsType>>::PackedSize;
        constexpr index_t BPackedSize =
            ck_tile::numeric_traits<remove_cvref_t<BLdsType>>::PackedSize;

        constexpr index_t a_lds_block_space_size_aligned = integer_least_multiple(
            sizeof(ALdsType) * a_lds_block_desc.get_element_space_size() / APackedSize, 16);
        constexpr index_t b_lds_block_space_size_aligned = integer_least_multiple(
            sizeof(BLdsType) * b_lds_block_desc.get_element_space_size() / BPackedSize, 16);

        constexpr index_t all_a_buffers_size = a_lds_block_space_size_aligned * num_lds_buffers;

        // num_lds_buffers a_lds_block: [A_0][A_1]
        auto a_lds_blocks = generate_tuple(
            [&](auto i) {
                ALdsType* __restrict__ p_a_lds = static_cast<ALdsType*>(static_cast<void*>(
                    static_cast<char*>(p_smem) + a_lds_block_space_size_aligned * i.value));
                return make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);
            },
            number<num_lds_buffers>{});

        // num_lds_buffers b_lds_block: [B_0][B_1]
        auto b_lds_blocks = generate_tuple(
            [&](auto i) {
                BLdsType* __restrict__ p_b_lds = static_cast<BLdsType*>(
                    static_cast<void*>(static_cast<char*>(p_smem) + all_a_buffers_size +
                                       b_lds_block_space_size_aligned * i.value));
                return make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);
            },
            number<num_lds_buffers>{});

        return make_tuple(std::move(a_lds_blocks), std::move(b_lds_blocks));
    }

    template <typename DramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, DramBlockWindowTmp>::value, bool>* =
                  nullptr>
    CK_TILE_DEVICE constexpr auto CopyADramWindow(const DramBlockWindowTmp& dram_block_window_tmp,
                                                  const array<index_t, 2>& offset = {0, 0}) const
    {
        constexpr bool is_col_major = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;

        using YPerTile = std::conditional_t<is_col_major, number<KPerBlock>, number<MPerBlock>>;
        using XPerTile = std::conditional_t<is_col_major, number<MPerBlock>, number<KPerBlock>>;
        // A DRAM tile window for load
        auto a_copy_dram_window = generate_tuple(
            [&](auto idx) {
                return make_tile_window(
                    dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                    make_tuple(YPerTile{}, XPerTile{}),
                    dram_block_window_tmp[number<idx>{}].get_window_origin() + offset,
                    Policy::template MakeADramTileDistribution<Problem>());
            },
            number<DramBlockWindowTmp::size()>{});
        return std::move(a_copy_dram_window);
    }

    template <typename DramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, DramBlockWindowTmp>::value, bool>* =
                  nullptr>
    CK_TILE_DEVICE constexpr auto CopyADramWindow(const DramBlockWindowTmp& dram_block_window_tmp,
                                                  const array<index_t, 2>& offset = {0, 0}) const
    {
        constexpr bool is_col_major = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;

        using YPerTile = std::conditional_t<is_col_major, number<KPerBlock>, number<MPerBlock>>;
        using XPerTile = std::conditional_t<is_col_major, number<MPerBlock>, number<KPerBlock>>;
        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(YPerTile{}, XPerTile{}),
                             dram_block_window_tmp.get_window_origin() + offset,
                             Policy::template MakeADramTileDistribution<Problem>());

        return std::move(a_copy_dram_window);
    }

    template <typename DramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, DramBlockWindowTmp>::value, bool>* =
                  nullptr>
    CK_TILE_DEVICE constexpr auto CopyBDramWindow(const DramBlockWindowTmp& dram_block_window_tmp,
                                                  const array<index_t, 2>& offset = {0, 0}) const
    {
        constexpr bool is_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

        using YPerTile = std::conditional_t<is_row_major, number<KPerBlock>, number<NPerBlock>>;
        using XPerTile = std::conditional_t<is_row_major, number<NPerBlock>, number<KPerBlock>>;
        // A DRAM tile window for load
        auto a_copy_dram_window = generate_tuple(
            [&](auto idx) {
                return make_tile_window(
                    dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                    make_tuple(YPerTile{}, XPerTile{}),
                    dram_block_window_tmp[number<idx>{}].get_window_origin() + offset,
                    Policy::template MakeBDramTileDistribution<Problem>());
            },
            number<DramBlockWindowTmp::size()>{});
        return std::move(a_copy_dram_window);
    }

    template <typename DramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, DramBlockWindowTmp>::value, bool>* =
                  nullptr>
    CK_TILE_DEVICE constexpr auto CopyBDramWindow(const DramBlockWindowTmp& dram_block_window_tmp,
                                                  const array<index_t, 2>& offset = {0, 0}) const
    {
        constexpr bool is_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

        using YPerTile = std::conditional_t<is_row_major, number<KPerBlock>, number<NPerBlock>>;
        using XPerTile = std::conditional_t<is_row_major, number<NPerBlock>, number<KPerBlock>>;
        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(YPerTile{}, XPerTile{}),
                             dram_block_window_tmp.get_window_origin() + offset,
                             Policy::template MakeBDramTileDistribution<Problem>());

        return std::move(a_copy_dram_window);
    }

    template <typename ALdsTensorView, typename ALdsLoadTileDistr>
    CK_TILE_DEVICE constexpr auto MakeALdsWindows(const ALdsTensorView& a_lds_block_view,
                                                  const ALdsLoadTileDistr&) const
    {
        auto a_lds_shape = []() {
            if constexpr(is_a_load_tr)
                return make_tuple(number<KPerBlock>{}, number<MPerBlock>{});
            else
                return make_tuple(number<MPerBlock>{}, number<KPerBlock>{});
        }();

        auto a_copy_lds_window = make_tile_window(a_lds_block_view, a_lds_shape, {0, 0});

        auto a_lds_load_tile_distr = []() {
            if constexpr(is_a_load_tr)
            {
                return make_static_tile_distribution(
                    typename InputTileDistributionTraits<
                        typename ALdsLoadTileDistr::DstrEncode,
                        typename ALdsTensorView::DataType>::TransposedDstrEncode{});
            }
            else
            {
                return ALdsLoadTileDistr{};
            }
        }();

        constexpr index_t KSubTileNum = []() {
            if constexpr(has_get_pipeline_subtile_params<Policy, Problem>::value)
                return Policy::template GetPipelineSubTileNum<Problem>().value;
            else
                return 1;
        }();

        auto a_lds_gemm_shape = []() {
            if constexpr(is_a_load_tr)
                return make_tuple(number<KPerBlock / KSubTileNum>{}, number<MPerBlock>{});
            else
                return make_tuple(number<MPerBlock>{}, number<KPerBlock / KSubTileNum>{});
        }();

        auto a_lds_gemm_window =
            make_tile_window(a_lds_block_view, a_lds_gemm_shape, {0, 0}, a_lds_load_tile_distr);

        return make_tuple(std::move(a_copy_lds_window), std::move(a_lds_gemm_window));
    }

    template <
        typename ADramBlockWindowTmp,
        typename ALdsTensorView,
        typename ALdsLoadTileDistr,
        typename std::enable_if_t<!is_detected<is_tuple, ALdsTensorView>::value, bool>* = nullptr>
    CK_TILE_DEVICE constexpr auto GetAWindows(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                              const ALdsTensorView& a_lds_block_view,
                                              const ALdsLoadTileDistr& a_lds_load_tile_distr,
                                              const array<index_t, 2>& offset = {0, 0}) const
    {
        // A DRAM tile window for load
        auto a_copy_dram_window = CopyADramWindow(a_dram_block_window_tmp, offset);

        // Create LDS windows
        auto [a_copy_lds_window, a_lds_gemm_window] =
            MakeALdsWindows(a_lds_block_view, a_lds_load_tile_distr);

        return make_tuple(std::move(a_copy_dram_window),
                          std::move(a_copy_lds_window),
                          std::move(a_lds_gemm_window));
    }

    // Unified GetAWindows that supports 1, 2, or 3 LDS buffers
    template <typename ADramBlockWindowTmp,
              typename ALdsTensorViewsTuple,
              typename ALdsLoadTileDistr,
              typename std::enable_if_t<is_detected<is_tuple, ALdsTensorViewsTuple>::value, bool>* =
                  nullptr>
    CK_TILE_DEVICE constexpr auto GetAWindows(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                              const ALdsTensorViewsTuple& a_lds_block_views_tuple,
                                              const ALdsLoadTileDistr& a_lds_load_tile_distr,
                                              const array<index_t, 2>& offset = {0, 0}) const
    {
        // A DRAM tile window for load
        auto a_copy_dram_window = CopyADramWindow(a_dram_block_window_tmp, offset);

        // Create LDS windows for each buffer
        constexpr index_t num_buffers = ALdsTensorViewsTuple::size();
        auto a_lds_windows            = generate_tuple(
            [&](auto i) {
                return MakeALdsWindows(a_lds_block_views_tuple[i], a_lds_load_tile_distr);
            },
            number<num_buffers>{});

        // Return: (dram_window, lds_windows_tuple)
        // lds_windows_tuple[i] = (copy_lds_window_i, lds_gemm_window_i)
        return make_tuple(std::move(a_copy_dram_window), std::move(a_lds_windows));
    }

    template <typename BLdsTensorView, typename BLdsLoadTileDistr>
    CK_TILE_DEVICE constexpr auto MakeBLdsWindows(const BLdsTensorView& b_lds_block_view,
                                                  const BLdsLoadTileDistr&) const
    {
        auto b_lds_shape = []() {
            if constexpr(is_b_load_tr)
                return make_tuple(number<KPerBlock>{}, number<NPerBlock>{});
            else
                return make_tuple(number<NPerBlock>{}, number<KPerBlock>{});
        }();

        auto b_copy_lds_window = make_tile_window(b_lds_block_view, b_lds_shape, {0, 0});

        auto b_lds_load_tile_distr = []() {
            if constexpr(is_b_load_tr)
            {
                return make_static_tile_distribution(
                    typename InputTileDistributionTraits<
                        typename BLdsLoadTileDistr::DstrEncode,
                        typename BLdsTensorView::DataType>::TransposedDstrEncode{});
            }
            else
            {
                return BLdsLoadTileDistr{};
            }
        }();

        constexpr index_t KSubTileNum = []() {
            if constexpr(has_get_pipeline_subtile_params<Policy, Problem>::value)
                return Policy::template GetPipelineSubTileNum<Problem>().value;
            else
                return 1;
        }();

        auto b_lds_gemm_shape = []() {
            if constexpr(is_b_load_tr)
                return make_tuple(number<KPerBlock / KSubTileNum>{}, number<NPerBlock>{});
            else
                return make_tuple(number<NPerBlock>{}, number<KPerBlock / KSubTileNum>{});
        }();

        auto b_lds_gemm_window =
            make_tile_window(b_lds_block_view, b_lds_gemm_shape, {0, 0}, b_lds_load_tile_distr);

        return make_tuple(std::move(b_copy_lds_window), std::move(b_lds_gemm_window));
    }

    template <
        typename BDramBlockWindowTmp,
        typename BLdsTensorView,
        typename BLdsLoadTileDistr,
        typename std::enable_if_t<!is_detected<is_tuple, BLdsTensorView>::value, bool>* = nullptr>
    CK_TILE_DEVICE constexpr auto GetBWindows(const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                              const BLdsTensorView& b_lds_block_view,
                                              const BLdsLoadTileDistr& b_lds_load_tile_distr,
                                              const array<index_t, 2>& offset = {0, 0}) const
    {
        // A DRAM tile window for load
        auto b_copy_dram_window = CopyBDramWindow(b_dram_block_window_tmp, offset);

        // Create LDS windows
        auto [b_copy_lds_window, b_lds_gemm_window] =
            MakeBLdsWindows(b_lds_block_view, b_lds_load_tile_distr);

        return make_tuple(std::move(b_copy_dram_window),
                          std::move(b_copy_lds_window),
                          std::move(b_lds_gemm_window));
    }

    // Unified GetBWindows that supports 1, 2, or 3 LDS buffers
    template <typename BDramBlockWindowTmp,
              typename BLdsTensorViewsTuple,
              typename BLdsLoadTileDistr,
              typename std::enable_if_t<is_detected<is_tuple, BLdsTensorViewsTuple>::value, bool>* =
                  nullptr>
    CK_TILE_DEVICE constexpr auto GetBWindows(const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                              const BLdsTensorViewsTuple& b_lds_block_views_tuple,
                                              const BLdsLoadTileDistr& b_lds_load_tile_distr,
                                              const array<index_t, 2>& offset = {0, 0}) const
    {
        // B DRAM tile window for load
        auto b_copy_dram_window = CopyBDramWindow(b_dram_block_window_tmp, offset);

        // Create LDS windows for each buffer
        constexpr index_t num_buffers = BLdsTensorViewsTuple::size();
        auto b_lds_windows            = generate_tuple(
            [&](auto i) {
                return MakeBLdsWindows(b_lds_block_views_tuple[i], b_lds_load_tile_distr);
            },
            number<num_buffers>{});

        // Return: (dram_window, lds_windows_tuple)
        // lds_windows_tuple[i] = (copy_lds_window_i, lds_gemm_window_i)
        return make_tuple(std::move(b_copy_dram_window), std::move(b_lds_windows));
    }
};

} // namespace ck_tile
