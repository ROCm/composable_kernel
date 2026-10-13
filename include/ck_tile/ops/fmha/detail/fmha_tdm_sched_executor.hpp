// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_schedule_types.hpp"

namespace ck_tile {

// Execute the selected row placement directly. All callbacks perform real
// instructions; there are no semantic event tokens or fallback operations.
template <typename TileSchedule, typename Mode>
struct FmhaTdmSchedPlacementExecutor
{
    using Placement = typename TileSchedule::LocalPlacement;
    using Geometry  = typename Mode::Geometry;

    template <index_t Stage, typename WmmaEmitter, typename ReadEmitter, typename FenceEmitter>
    CK_TILE_HOST_DEVICE static constexpr void
    ExecuteQkStage(WmmaEmitter& emit_wmma, ReadEmitter& emit_read, FenceEmitter& emit_fence)
    {
        static_assert(Stage >= 0 && Stage < Geometry::kQkStages);
        static_for<0, Geometry::kQkWmmasPerStage, 1>{}([&](auto wmma) {
            Placement::template RunQkRow<Mode, Stage, decltype(wmma)::value>(
                emit_wmma, emit_read, emit_fence);
        });
    }

    template <index_t Stage,
              typename WmmaEmitter,
              typename ReadEmitter,
              typename FenceEmitter,
              typename RescaleEmitter>
    CK_TILE_HOST_DEVICE static constexpr void ExecutePvStage(WmmaEmitter& emit_wmma,
                                                             ReadEmitter& emit_read,
                                                             FenceEmitter& emit_fence,
                                                             RescaleEmitter& emit_rescale)
    {
        static_assert(Stage >= 0 && Stage < Geometry::kPvStages);
        static_for<0, Geometry::kPvWmmasPerStage, 1>{}([&](auto wmma) {
            Placement::template RunPvRow<Mode, Stage, decltype(wmma)::value>(
                emit_wmma, emit_read, emit_fence, emit_rescale);
        });
    }
};

} // namespace ck_tile
