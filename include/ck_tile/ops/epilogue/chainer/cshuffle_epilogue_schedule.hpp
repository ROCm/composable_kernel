// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/epilogue/chainer/common_epilogue_ops.hpp"
#include "ck_tile/ops/epilogue/chainer/epilogue_chainer.hpp"
#include "ck_tile/ops/epilogue/chainer/cshuffle_epilogue_chainer_ops.hpp"

namespace ck_tile {

/// @brief Schedule type tags for epilogue selection
///
/// @par Purpose
///     Each tag corresponds to a pre-built schedule, these are used to select a schedule
struct DefaultScheduleTag
{
}; ///< Standard epilogue schedule: Slice → Cast → PrepC → ApplyD → Store → Move
struct ScaleScheduleTag
{
}; ///< Scaling epilogue schedule: Slice → Scale → Cast → PrepC → ApplyD → Store → Move

/// @brief CShuffle epilogue scheduler providing pre-built schedules
///
/// @par Overview
///     CshuffleEpilogueSchedule acts as the scheduler component for EpilogueChainer.
///     It provides context creation and pre-built schedules. The scheduler
///     uses tags to select/create appropriate epilogue schedule.
///
/// @tparam Problem The epilogue problem configuration
/// @tparam ScheduleTag Tag selecting the epilogue schedule
template <typename Problem, typename ScheduleTag = DefaultScheduleTag>
struct CshuffleEpilogueSchedule
{
    using ProblemType = Problem;
    using BaseOp      = CShuffleEpilogueChainBaseOp<Problem>;
    using Tag         = ScheduleTag;

    static constexpr index_t NumAccess = BaseOp::SFC::get_num_of_access();

    /// @brief Create context for epilogue operations
    template <typename ODramWindow, typename OAccTile, typename DsDramWindows>
    CK_TILE_DEVICE static auto create_context(ODramWindow& out_dram_window,
                                              const OAccTile& o_acc_tile,
                                              const DsDramWindows& ds_dram_windows,
                                              void* p_smem)
    {
        return BaseOp{}(out_dram_window, o_acc_tile, ds_dram_windows, p_smem);
    }

    /// @brief make schedule based on compile-time tag selection
    template <typename... Args>
    CK_TILE_DEVICE static auto make_schedule(Args&&... args)
    {
        if constexpr(std::is_same_v<ScheduleTag, DefaultScheduleTag>)
        {
            // Standard epilogue
            // Schedule: Slice -> Cast -> PrepC -> ApplyD -> Store -> MoveWindows
            static_assert(sizeof...(args) == 0, "DefaultSchedule expects no arguments");
            return make_graph<NumAccess>(
                make_node<SliceEpilogue<typename BaseOp::SFC,
                                        typename BaseOp::CWarpDstr,
                                        BaseOp::NumMXdlPerWavePerShuffle,
                                        BaseOp::NumNXdlPerWavePerShuffle,
                                        BaseOp::MPerIterationShuffle,
                                        BaseOp::NPerIterationShuffle>>(),
                make_node<CastLdsEpilogue<typename BaseOp::ODataType>>(),
                make_node<PrepCTensorEpilogue<typename BaseOp::TileEncodingPattern>>(),
                make_node<ApplyDEpilogue<typename Problem::CDElementwise, Problem::NumDTensor>>(),
                make_node<StoreToDramEpilogue<Problem::MemoryOperation>>(),
                make_node<MoveWindowsEpilogue<typename BaseOp::SFC, Problem::NumDTensor>>());
        }
        else if constexpr(std::is_same_v<ScheduleTag, ScaleScheduleTag>)
        {
            // Scaling schedule
            // Schedule: Slice -> Scale -> Cast -> PrepC -> ApplyD -> Store -> MoveWindows
            static_assert(sizeof...(args) == 2, "ScaleSchedule requires exactly 2 scale arguments");
            return make_graph<NumAccess>(
                make_node<SliceEpilogue<typename BaseOp::SFC,
                                        typename BaseOp::CWarpDstr,
                                        BaseOp::NumMXdlPerWavePerShuffle,
                                        BaseOp::NumNXdlPerWavePerShuffle,
                                        BaseOp::MPerIterationShuffle,
                                        BaseOp::NPerIterationShuffle>>(),
                make_node<ScaleEpilogue<typename BaseOp::SFC>>(std::forward<Args>(args)...),
                make_node<CastLdsEpilogue<typename BaseOp::ODataType>>(),
                make_node<PrepCTensorEpilogue<typename BaseOp::TileEncodingPattern>>(),
                make_node<ApplyDEpilogue<typename Problem::CDElementwise, Problem::NumDTensor>>(),
                make_node<StoreToDramEpilogue<Problem::MemoryOperation>>(),
                make_node<MoveWindowsEpilogue<typename BaseOp::SFC, Problem::NumDTensor>>());
        }
        else
        {
            static_assert(sizeof(ScheduleTag) == 0, "Unknown schedule tag");
        }
    }
};

} // namespace ck_tile
