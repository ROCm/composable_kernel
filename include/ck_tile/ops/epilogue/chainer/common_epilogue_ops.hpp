// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/epilogue/chainer/cshuffle_epilogue_chainer_ops.hpp"

#include <optional>

namespace ck_tile {

/// @file common_epilogue_ops.hpp
/// @brief Reusable epilogue operations for chainer composition
///
/// @par Overview
///     This file provides epilogue operations that can be composed
///     into epilogue graphs.

template <typename SFC,
          typename CWarpDstr,
          index_t NumMXdlPerWavePerShuffle,
          index_t NumNXdlPerWavePerShuffle,
          index_t MPerIterationShuffle,
          index_t NPerIterationShuffle>
struct SliceEpilogue
{

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows, typename ContextType>
    CK_TILE_DEVICE auto operator()([[maybe_unused]] ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   [[maybe_unused]] const DsDramWindows& ds_dram_windows,
                                   [[maybe_unused]] void* p_smem,
                                   auto iAccess,
                                   ContextType& context)
    {
        block_sync_lds();

        // Calculate which tile slice to extract based on access index
        constexpr auto idx_y_start = SFC::get_index(iAccess);
        constexpr auto mIter       = number<idx_y_start.at(number<0>{}) / (MPerIterationShuffle)>{};
        constexpr auto nIter       = number<idx_y_start.at(number<1>{}) / (NPerIterationShuffle)>{};

        // Get warp distribution parameters
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // Extract the slice from accumulator tile and store in context LDS tile
        context.lds_tile.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
            merge_sequences(
                sequence<mIter * NumMXdlPerWavePerShuffle, nIter * NumNXdlPerWavePerShuffle>{},
                c_warp_y_index_zeros),
            merge_sequences(sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{},
                            c_warp_y_lengths));
    }
};

template <typename SFC>
struct ScaleEpilogue
{

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ContextType,
              typename ScaleM,
              typename ScaleN>
    CK_TILE_DEVICE auto operator()([[maybe_unused]] ODramWindow& out_dram_window,
                                   [[maybe_unused]] const OAccTile& o_acc_tile,
                                   [[maybe_unused]] const DsDramWindows& ds_dram_windows,
                                   [[maybe_unused]] void* p_smem,
                                   auto iAccess,
                                   ContextType& context,
                                   const ScaleM& scale_m_tensor,
                                   const ScaleN& scale_n_tensor)
    {
        // Calculate offset for this iteration
        constexpr auto step     = SFC::get_index(iAccess);
        constexpr auto m_offset = step.at(number<0>{});
        constexpr auto n_offset = step.at(number<1>{});

        // Create windows with correct offset directly
        auto scale_m_window = make_tile_window(
            scale_m_tensor, {m_offset, n_offset}, context.lds_tile.get_tile_distribution());
        auto scale_n_window = make_tile_window(
            scale_n_tensor, {m_offset, n_offset}, context.lds_tile.get_tile_distribution());

        // Load and apply scaling
        const auto scale_m_tile = load_tile(scale_m_window);
        const auto scale_n_tile = load_tile(scale_n_window);

        tile_elementwise_inout(element_wise::MultiDMultiply{},
                               context.lds_tile,
                               context.lds_tile,
                               scale_m_tile,
                               scale_n_tile);
    }
};

template <typename ODataType>
struct CastLdsEpilogue
{

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows, typename ContextType>
    CK_TILE_DEVICE auto operator()([[maybe_unused]] ODramWindow& out_dram_window,
                                   [[maybe_unused]] const OAccTile& o_acc_tile,
                                   [[maybe_unused]] const DsDramWindows& ds_dram_windows,
                                   [[maybe_unused]] void* p_smem,
                                   [[maybe_unused]] auto iAccess,
                                   ContextType& context)
    {
        // Cast LDS tile to output data type and store to LDS
        const auto c_warptile_in_tensor_casted = cast_tile<ODataType>(context.lds_tile);
        store_tile(context.in_lds_window, c_warptile_in_tensor_casted);
    }
};

template <typename TileEncodingPattern>
struct PrepCTensorEpilogue
{

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows, typename ContextType>
    CK_TILE_DEVICE auto operator()([[maybe_unused]] ODramWindow& out_dram_window,
                                   [[maybe_unused]] const OAccTile& o_acc_tile,
                                   [[maybe_unused]] const DsDramWindows& ds_dram_windows,
                                   [[maybe_unused]] void* p_smem,
                                   [[maybe_unused]] auto iAccess,
                                   ContextType& context)
    {
        // Create distribution and synchronize before loading from LDS
        constexpr auto dram_tile_distribution =
            TileEncodingPattern::make_2d_static_tile_distribution();
        block_sync_lds();

        // Load C tensor from LDS into context
        context.c_out_tensor =
            load_tile(make_tile_window(context.out_lds_window, dram_tile_distribution));
    }
};

template <typename CDElementwise, index_t NumDTensor>
struct ApplyDEpilogue
{

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows, typename ContextType>
    CK_TILE_DEVICE auto operator()([[maybe_unused]] ODramWindow& out_dram_window,
                                   [[maybe_unused]] const OAccTile& o_acc_tile,
                                   [[maybe_unused]] const DsDramWindows& ds_dram_windows,
                                   [[maybe_unused]] void* p_smem,
                                   [[maybe_unused]] auto iAccess,
                                   ContextType& context)
    {
        // Load all D tensors
        const auto ds_tensor = generate_tuple(
            [&](auto idx) { return load_tile(context.d_dram_windows[idx]); }, number<NumDTensor>{});

        // Concatenate C and D tensors for element-wise operation
        const auto c_ds_tiles = concat_tuple_of_reference(
            tie(context.c_out_tensor, context.c_out_tensor),
            generate_tie([&](auto idx) -> const auto& { return ds_tensor[idx]; },
                         number<NumDTensor>{}));

        // Apply element-wise operation (e.g., C = C + D0 + D1 + ...)
        tile_elementwise_inout_unpack(CDElementwise{}, c_ds_tiles);
    }
};

template <memory_operation_enum MemoryOperation>
struct StoreToDramEpilogue
{

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows, typename ContextType>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   [[maybe_unused]] const OAccTile& o_acc_tile,
                                   [[maybe_unused]] const DsDramWindows& ds_dram_windows,
                                   [[maybe_unused]] void* p_smem,
                                   [[maybe_unused]] auto iAccess,
                                   ContextType& context)
    {
        // Store final tensor based on memory operation type
        if constexpr(MemoryOperation == memory_operation_enum::set)
        {
            store_tile(out_dram_window, context.c_out_tensor);
        }
        else
        {
            update_tile(out_dram_window, context.c_out_tensor);
        }
    }
};

template <typename SFC, index_t NumDTensor>
struct MoveWindowsEpilogue
{

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows, typename ContextType>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   [[maybe_unused]] const OAccTile& o_acc_tile,
                                   [[maybe_unused]] const DsDramWindows& ds_dram_windows,
                                   [[maybe_unused]] void* p_smem,
                                   auto iAccess,
                                   ContextType& context)
    {
        // Move windows only if not the last access iteration
        constexpr index_t num_access = SFC::get_num_of_access();
        if constexpr(iAccess != num_access - 1)
        {
            constexpr auto step = SFC::get_forward_step(iAccess);

            // Move output window
            move_tile_window(out_dram_window, {step.at(number<0>{}), step.at(number<1>{})});

            // Move all D tensor windows
            static_for<0, NumDTensor, 1>{}([&](auto idx) {
                move_tile_window(context.d_dram_windows[idx],
                                 {step.at(number<0>{}), step.at(number<1>{})});
            });
        }
    }
};

} // namespace ck_tile
