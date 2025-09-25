// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename EpilogueType, typename... Args>
struct EpilogueNode
{
    using Epilogue = EpilogueType;
    ck_tile::tuple<Args...> args;

    constexpr EpilogueNode(Args... a) : args(a...) {}

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows, typename Context>
    CK_TILE_DEVICE void execute(ODramWindow& out_dram_window,
                               const OAccTile& o_acc_tile,
                               const DsDramWindows& ds_dram_windows,
                               void* p_smem,
                               Context& context) const
    {
        ck_tile::apply([&](auto&&... epilogue_args) {
            EpilogueType{}(out_dram_window, o_acc_tile, ds_dram_windows, p_smem, context,
                          std::forward<decltype(epilogue_args)>(epilogue_args)...);
        }, args);
    }

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows, typename Context, index_t I>
    CK_TILE_DEVICE void execute(ODramWindow& out_dram_window,
                                              const OAccTile& o_acc_tile,
                                              const DsDramWindows& ds_dram_windows,
                                              void* p_smem,
                                              Context& context,
                                              number<I> iAccess) const
    {
        ck_tile::apply([&](auto&&... epilogue_args) {
            EpilogueType{}(out_dram_window, o_acc_tile, ds_dram_windows, p_smem, iAccess, context,
                          std::forward<decltype(epilogue_args)>(epilogue_args)...);
        }, args);
    }
};


template <typename EpilogueType>
struct EpilogueNode<EpilogueType>
{
    using Epilogue = EpilogueType;
    ck_tile::tuple<> args;
    
    constexpr EpilogueNode() = default;
    
    template <typename ODramWindow, typename OAccTile, typename DsDramWindows, typename Context>
    CK_TILE_DEVICE void execute(ODramWindow& out_dram_window,
                               const OAccTile& o_acc_tile,
                               const DsDramWindows& ds_dram_windows,
                               void* p_smem,
                               Context& context) const
    {
        EpilogueType{}(out_dram_window, o_acc_tile, ds_dram_windows, p_smem, context);
    }
    
    template <typename ODramWindow, typename OAccTile, typename DsDramWindows, typename Context, index_t I>
    CK_TILE_DEVICE void execute(ODramWindow& out_dram_window,
                                              const OAccTile& o_acc_tile,
                                              const DsDramWindows& ds_dram_windows,
                                              void* p_smem,
                                              Context& context,
                                              number<I> iAccess) const
    {
        EpilogueType{}(out_dram_window, o_acc_tile, ds_dram_windows, p_smem, iAccess, context);
    }
};

// Loop construct for executing epilogue sequences
template <index_t Count, typename... EpilogueTypes>
struct EpilogueLoop
{
    ck_tile::tuple<EpilogueTypes...> epilogues;
    
    constexpr EpilogueLoop() = default;
    constexpr EpilogueLoop(EpilogueTypes... eps) : epilogues(eps...) {}
    
    template <typename ODramWindow, typename OAccTile, typename DsDramWindows, typename Context>
    CK_TILE_DEVICE void execute(ODramWindow& out_dram_window,
                               const OAccTile& o_acc_tile,
                               const DsDramWindows& ds_dram_windows,
                               void* p_smem,
                               Context& context) const
    {
        static_for<0, Count, 1>{}([&](auto iAccess) {
            static_for<0, sizeof...(EpilogueTypes), 1>{}([&](auto I) {
                epilogues.template get<I.value>().execute(
                    out_dram_window, o_acc_tile, ds_dram_windows, p_smem, context, iAccess);
            });
        });
    }
};

// Convenience functions
template <typename EpilogueType, typename... Args>
constexpr auto make_node(Args... args)
{
    return EpilogueNode<EpilogueType, Args...>{args...};
}

template <index_t Count, typename... EpilogueTypes>
constexpr auto make_loop(EpilogueTypes... epilogues)
{
    return EpilogueLoop<Count, EpilogueTypes...>{epilogues...};
}

} // namespace ck_tile


