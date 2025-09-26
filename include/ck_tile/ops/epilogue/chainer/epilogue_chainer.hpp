// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/chainer/epilogue_graph.hpp"

namespace ck_tile {
template <typename InitEpilogue, typename MainSequence>
class EpilogueChainer
{
    public:
    using SelectEpilogue                  = InitEpilogue;
    using Problem                         = typename InitEpilogue::Problem;
    using ODataType                       = typename InitEpilogue::ODataType;
    using DsDataType                      = typename InitEpilogue::DsDataType;
    using DsLayout                        = typename InitEpilogue::DsLayout;
    using AccDataType                     = typename InitEpilogue::AccDataType;
    static constexpr auto MemoryOperation = InitEpilogue::MemoryOperation;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return InitEpilogue::GetSmemSize();
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC()
    {
        return InitEpilogue::GetVectorSizeC();
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeD(number<I> idx)
    {
        return InitEpilogue::GetVectorSizeD(idx);
    }

    CK_TILE_DEVICE static constexpr auto MakeLdsDistributionEncode()
    {
        return InitEpilogue::MakeLdsDistributionEncode();
    }

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows>
    CK_TILE_DEVICE void operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem,
                                   const MainSequence& sequence) const
    {
        // Initialize context
        auto context = InitEpilogue{}(out_dram_window, o_acc_tile, ds_dram_windows, p_smem);

        // Execute the sequence
        sequence.execute(out_dram_window, o_acc_tile, ds_dram_windows, p_smem, context);
    }
};

} // namespace ck_tile
