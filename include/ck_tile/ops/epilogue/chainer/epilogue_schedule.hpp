// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/epilogue/chainer/epilogue_graph.hpp"
#include "ck_tile/ops/epilogue/chainer/cshuffle_chained_epilogues.hpp"

namespace ck_tile {

/// @brief Pre-built CShuffle epilogue sequences
/// Provides common epilogue patterns
template <typename Problem>
struct CshuffleEpilogueSchedule
{
    using Init                         = CShuffleEpilogueStageBase<Problem>;
    static constexpr index_t NumAccess = Init::SFC::get_num_of_access();

    /// @brief Standard epilogue without scaling
    /// Sequence: Slice -> Cast -> PrepC -> ApplyD -> Store -> MoveWindows
    CK_TILE_DEVICE static auto make_base_schedule()
    {
        return make_loop<NumAccess>(make_node<SliceEpilogue<Problem>>(),
                                    make_node<CastLdsEpilogue<Problem>>(),
                                    make_node<PrepCTensorEpilogue<Problem>>(),
                                    make_node<ApplyDEpilogue<Problem>>(),
                                    make_node<StoreToDramEpilogue<Problem>>(),
                                    make_node<MoveWindowsEpilogue<Problem>>());
    }

    /// @brief Epilogue with row/column scaling
    /// Sequence: Slice -> Scale -> Cast -> PrepC -> ApplyD -> Store -> MoveWindows
    /// @param scale_m_window M scaling window
    /// @param scale_n_window N scaling window
    template <typename ScaleMWindow, typename ScaleNWindow>
    CK_TILE_DEVICE static auto make_scale_schedule(const ScaleMWindow& scale_m_window,
                                                   const ScaleNWindow& scale_n_window)
    {
        return make_loop<NumAccess>(
            make_node<SliceEpilogue<Problem>>(),
            make_node<ScaleEpilogue<Problem>>(scale_m_window, scale_n_window),
            make_node<CastLdsEpilogue<Problem>>(),
            make_node<PrepCTensorEpilogue<Problem>>(),
            make_node<ApplyDEpilogue<Problem>>(),
            make_node<StoreToDramEpilogue<Problem>>(),
            make_node<MoveWindowsEpilogue<Problem>>());
    }
};

} // namespace ck_tile
