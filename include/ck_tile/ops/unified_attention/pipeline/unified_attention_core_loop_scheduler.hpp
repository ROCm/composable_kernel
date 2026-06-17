// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// UA-owned counterpart of `CoreLoopScheduler` in
// `ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp`. Emits the
// per-phase `__builtin_amdgcn_sched_group_barrier` instruction-mix hints for
// the unified-attention ping-pong loop. The mask sits in the K-side memory
// phase. The gemm1 + fmha_alu_D_upd phase reserves 2 VALU slots per PV-MFMA:
// the conditional online-softmax rescale (see CONDITIONAL_RESCALE) skips the
// 128-VGPR o_acc rescale on most KV tiles, so a tighter reservation lets the
// MFMAs pack the common skip path.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename PipelineProblem, bool kIsMasking>
struct UAcoreLoopScheduler;

template <typename PipelineProblem>
struct UAcoreLoopScheduler<PipelineProblem, /*kIsMasking=*/true>
{
    template <ck_tile::index_t WaveGroup, ck_tile::index_t Phase>
    CK_TILE_DEVICE static constexpr void schedule(ck_tile::number<WaveGroup>,
                                                  ck_tile::number<Phase>)
    {
        using namespace ck_tile;

        if constexpr(WaveGroup == 0)
        {
            if constexpr(Phase == 0)
            {
                // gemm0 + fmha_alu1
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x200, 2, 0); // TRANS
                    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                });
            }
            else if constexpr(Phase == 1)
            {
                // K_mem_load + V_lds_load + fmha_mask
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
            else if constexpr(Phase == 2)
            {
                // gemm1 + fmha_alu_D_upd
                __builtin_amdgcn_sched_group_barrier(0x002, 4, 0); // VALU
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                });
            }
            else if constexpr(Phase == 3)
            {
                // V_mem_load + K_lds_load
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
        }
        else
        {
            if constexpr(Phase == 0)
            {
                // V_mem_load + K_lds_load
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
            else if constexpr(Phase == 1)
            {
                // gemm0 + fmha_alu1
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x200, 2, 0); // TRANS
                    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                });
            }
            else if constexpr(Phase == 2)
            {
                // K_mem_load + V_lds_load + fmha_mask
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
            else if constexpr(Phase == 3)
            {
                // gemm1 + fmha_alu_D_upd
                __builtin_amdgcn_sched_group_barrier(0x002, 4, 0); // VALU
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                });
            }
        }
    }
};

template <typename PipelineProblem>
struct UAcoreLoopScheduler<PipelineProblem, /*kIsMasking=*/false>
{
    template <ck_tile::index_t WaveGroup, ck_tile::index_t Phase>
    CK_TILE_DEVICE static constexpr void schedule(ck_tile::number<WaveGroup>,
                                                  ck_tile::number<Phase>)
    {
        using namespace ck_tile;

        if constexpr(WaveGroup == 0)
        {
            if constexpr(Phase == 0)
            {
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x200, 2, 0); // TRANS
                    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                });
            }
            else if constexpr(Phase == 1)
            {
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
            else if constexpr(Phase == 2)
            {
                __builtin_amdgcn_sched_group_barrier(0x002, 4, 0); // VALU
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                });
            }
            else if constexpr(Phase == 3)
            {
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
        }
        else
        {
            if constexpr(Phase == 0)
            {
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
            else if constexpr(Phase == 1)
            {
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x200, 2, 0); // TRANS
                    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                });
            }
            else if constexpr(Phase == 2)
            {
                __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
            }
            else if constexpr(Phase == 3)
            {
                __builtin_amdgcn_sched_group_barrier(0x002, 4, 0); // VALU
                static_for<0, 8, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                });
            }
        }
    }
};

} // namespace ck_tile
