// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

// this epilogue just store out a M*N matrix, row major

template <typename AccDataType_,
          typename ODataType_,
          bool kPadM_,
          bool kPadN_,
          bool UseRawStore_                      = true,
          memory_operation_enum MemoryOperation_ = memory_operation_enum::set>
struct Default2DEpilogueProblem
{
    using AccDataType                                      = remove_cvref_t<AccDataType_>;
    using ODataType                                        = remove_cvref_t<ODataType_>;
    static constexpr bool kPadM                            = kPadM_;
    static constexpr bool kPadN                            = kPadN_;
    static constexpr bool UseRawStore                      = UseRawStore_;
    static constexpr memory_operation_enum MemoryOperation = MemoryOperation_;
};

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename CLayout_,
          bool kPadM_,
          bool kPadN_,
          index_t kM_, 
          index_t kN_, 
          index_t MWave_, 
          index_t NWave_,
          index_t kMPerXdl_,
          index_t kNPerXdl_,
          index_t kKPerXdl_,
          bool isCTransposed_,
          bool UseRawStore_                      = true,
          memory_operation_enum MemoryOperation_ = memory_operation_enum::set, 
          index_t kNumWaveGroups_ = 1>
struct DefaultGemm2DEpilogueProblem : public Default2DEpilogueProblem<AccDataType_,
                                                                      ODataType_,
                                                                      kPadM_,
                                                                      kPadN_,
                                                                      UseRawStore_,
                                                                      MemoryOperation_>
{
    using ADataType                        = remove_cvref_t<ADataType_>;
    using BDataType                        = remove_cvref_t<BDataType_>;
    using CLayout                          = remove_cvref_t<CLayout_>;
    static constexpr index_t kMPerXdl      = kMPerXdl_;
    static constexpr index_t kNPerXdl      = kNPerXdl_;
    static constexpr index_t kKPerXdl      = kKPerXdl_;
    static constexpr index_t isCTransposed = isCTransposed_;

    static constexpr index_t kNumWaveGroups = kNumWaveGroups_;

    static constexpr index_t MWave          = MWave_;
    static constexpr index_t NWave          = NWave_;
    static constexpr index_t kMPerBlock     = kM_;
    static constexpr index_t kNPerBlock     = kN_;
    static constexpr index_t kBlockSize     = MWave_ * NWave_ * get_warp_size();

    using ODataType                        = remove_cvref_t<ODataType_>;
};

template <typename Problem_, typename Policy_ = void>
struct Default2DEpilogue
{
    using Problem                     = remove_cvref_t<Problem_>;
    using AccDataType                 = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType                   = remove_cvref_t<typename Problem::ODataType>;
    static constexpr bool kPadM       = Problem::kPadM;
    static constexpr bool kPadN       = Problem::kPadN;
    static constexpr bool UseRawStore = Problem::UseRawStore;
    static constexpr memory_operation_enum MemoryOperation = Problem::MemoryOperation;


    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    // TODO: this function assume store out vector size is the same as OAccTile last dimension size
    //       how do we fix this ?
    template <typename ODramWindowTmp, typename OAccTile>
    CK_TILE_DEVICE auto
    operator()([[maybe_unused]]ODramWindowTmp& o_dram_window_tmp, [[maybe_unused]]const OAccTile& o_acc_tile, void* = nullptr) const
    {

        //o_acc_tile --> static distributed tensor
        // o_dram_window_tmp --> tile window, tensor, padded_tensor, tensor_view.

        // Create tensor view, and then use make_tile_window to create a tile window
        // that matches the distribution of o_acc_tile.
        // Then load from the tile window and store to o_dram_window_tmp
        /*
        auto casted_tile = cast_tile<ODataType>(o_acc_tile);
        using TileEncodingPattern = 
            TileDistributionEncodingPattern2D<Problem::kBlockSize * Problem::kNumWaveGroups, 
                                              Problem::kMPerBlock, 
                                              Problem::kNPerBlock, 
                                              4, 
                                              tile_distribution_pattern::thread_raked, 
                                              Problem::kNumWaveGroups>;
        constexpr auto dram_tile_distribution = TileEncodingPattern::Make2DStaticTileDistribution();

        auto tensor_descriptor = make_naive_tensor_descriptor(make_tuple(number<Problem::kMPerBlock>{}, number<Problem::kNPerBlock>{}), 
                                                                make_tuple(number<Problem::kNPerBlock>{}, number<1>{}));

        auto tile_view = make_tensor_view(
                                            static_cast<ODataType*>(casted_tile.get_thread_buffer().get()), 
                                            tensor_descriptor);
        auto in_window = make_tile_window(tile_view, 
                                          make_tuple(number<Problem::kMPerBlock>{}, number<Problem::kNPerBlock>{}), 
                                          {0, 0}, 
                                        dram_tile_distribution);

        auto c_out_tensor = load_tile(in_window);
        update_tile(o_dram_window_tmp, c_out_tensor);
        */
        //update_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));

        if constexpr(UseRawStore && (kPadM || kPadN))
        {
            if constexpr(MemoryOperation == memory_operation_enum::set)
            {
                store_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            else
            {
                update_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            buffer_store_fence();
        }
        else
        {
            if constexpr(MemoryOperation == memory_operation_enum::set)
            {
                store_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            else
            {
                update_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
        }        
    }

    template <typename ODramWindowTmp, typename OAccTile, typename DsDramWindows>
    CK_TILE_DEVICE auto operator()(ODramWindowTmp& o_dram_window_tmp,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& /* unused */,
                                   void* = nullptr) const
    {
        return operator()<ODramWindowTmp, OAccTile>(o_dram_window_tmp, o_acc_tile);
    }
};

template <typename Problem_, typename Policy_ = void>
struct DefaultGemm2DEpilogue : public Default2DEpilogue<Problem_, Policy_>
{
    using Problem     = remove_cvref_t<Problem_>;
    using ADataType   = remove_cvref_t<typename Problem::ADataType>;
    using BDataType   = remove_cvref_t<typename Problem::BDataType>;
    using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType   = remove_cvref_t<typename Problem::ODataType>;
    // Used for weight-only quantization kernel, B would be dequantized to the same data type as A
    using BTypeToUse =
        std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
    using DsDataType                       = ck_tile::tuple<>;
    using DsLayout                         = ck_tile::tuple<>;
    using CLayout                          = remove_cvref_t<typename Problem::CLayout>;
    static constexpr index_t kMPerXdl      = Problem::kMPerXdl;
    static constexpr index_t kNPerXdl      = Problem::kNPerXdl;
    static constexpr index_t kKPerXdl      = Problem::kKPerXdl;
    static constexpr index_t isCTransposed = Problem::isCTransposed;

    using WG = WarpGemmDispatcher<ADataType,
                                  BTypeToUse,
                                  AccDataType,
                                  kMPerXdl,
                                  kNPerXdl,
                                  kKPerXdl,
                                  isCTransposed>;

    using CWarpDstr = typename WG::CWarpDstr;

    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeC()
    {
        // N is contiguous dimension
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if constexpr(isCTransposed)
            {
                // In this case each thread has multiple consecutive elements in
                // N dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
            else
            {
                // In this case each thread has just a single item in Ndim
                return (WG::WarpGemmAttribute::Impl::kCNLane *
                        WG::WarpGemmAttribute::Impl::kBNBlock) /
                       WG::kN;
            }
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>)
        {
            if constexpr(isCTransposed)
            {
                // In this case each thread has just a single item in Mdim
                return (WG::WarpGemmAttribute::Impl::kCNLane *
                        WG::WarpGemmAttribute::Impl::kAMBlock) /
                       WG::kN;
            }
            else
            {
                // In this case each thread has multiple consecutive elements in
                // M dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
        }
        else
        {
            static_assert(false, "Unsupported CLayout!");
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeD() { return 1; }
};

} // namespace ck_tile
