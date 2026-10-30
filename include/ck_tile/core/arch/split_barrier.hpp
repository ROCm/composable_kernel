// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

// Split workgroup barrier: arrive now, wait later.
//
// gfx12 splits the workgroup barrier into S_BARRIER_SIGNAL (arrive, does not block) and
// S_BARRIER_WAIT (block until every wave has arrived). Work placed between the two overlaps the
// barrier, which is how a double-buffered K loop hides its synchronisation behind WMMAs:
//
//     prologue:  fill buffer 0;            workgroup_barrier_arrive<0 on fill issuers>();
//     loop top:  workgroup_barrier_wait(); refill the other buffer; read this buffer;
//                compute part 1;           workgroup_barrier_arrive<...>();
//                compute part 2           (overlaps the barrier)
//     after:     workgroup_barrier_wait();
//
// A bare signal/wait pair is not enough, for the reasons named_barrier.hpp documents in full:
//   - Without a release fence before the signal, the signal can land while this wave's own LDS
//     accesses are still in flight, and nothing stops LLVM from moving LDS accesses across the
//     barrier (loop-invariant ds_loads have been seen hoisted out of the loop). The fence also
//     emits the s_wait_dscnt 0 drain; no explicit wait is needed next to it.
//   - Without an acquire fence after the wait, later LDS reads may be moved above it.
//   - The fences do NOT emit s_wait_tensorcnt: the memory model does not track TENSORcnt. A wave
//     that issued TENSOR_LOAD_TO_LDS fills must drain them before it publishes; pass
//     TensorCnt >= 0 to have workgroup_barrier_arrive do that first.
//   - Fences alone do not hold the overlap window. They order memory accesses only, and a WMMA
//     is register-only, so the scheduler may still slide the WMMAs meant to run between arrive
//     and wait to the other side of the signal (compile probe on gfx1250: 4 of 4 window WMMAs
//     with the pins below, 0 of 4 with the fences but without the pins, under CK's compile
//     flags). sched_barrier(0) on both sides of each barrier instruction keeps the code where
//     the caller put it.
//
// The fences are "workgroup" scope on purpose: the barrier never leaves one workgroup's LDS, and
// "agent" scope would add a vector L0 writeback and invalidate on every step.
//
// Rules for callers:
//   - Every wave of the workgroup calls arrive and wait the same number of times, in
//     alternation: arrive, wait, arrive, wait. Do not call any other workgroup barrier
//     (block_sync_lds, __syncthreads, ...) between an arrive and its wait.
//   - LDS accesses issued between an arrive and its wait are not covered by that barrier: only
//     what a wave did before its arrive is visible to the other waves after their wait.
//
// Targets without the split barrier: arrive only does the optional TENSORcnt wait, and wait is
// block_sync_lds(). Such a build has one full barrier per arrive/wait pair, at the wait.
//
// Not to be confused with ck_tile::workgroup_barrier (workgroup_barrier.hpp), a global-memory
// barrier between workgroups.

namespace ck_tile {

/// @brief Arrive on the workgroup barrier without blocking (split barrier, gfx12).
/// @tparam TensorCnt -1: no TENSORcnt wait. >= 0: first wait until at most TensorCnt of this
///         wave's TENSOR_LOAD_TO_LDS fills are outstanding (use 0 on the waves that issued the
///         fill this barrier publishes).
template <index_t TensorCnt = -1>
CK_TILE_DEVICE void workgroup_barrier_arrive()
{
    static_assert(TensorCnt >= -1, "TensorCnt is -1 (no wait) or a TENSORcnt threshold >= 0");
#if defined(__gfx12__)
    __builtin_amdgcn_sched_barrier(0);
    if constexpr(TensorCnt >= 0)
    {
        s_wait_tensorcnt<TensorCnt>();
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_sched_barrier(0);
#else
    if constexpr(TensorCnt >= 0)
    {
        s_wait_tensorcnt<TensorCnt>();
    }
#endif
}

/// @brief Wait until every wave of the workgroup has called workgroup_barrier_arrive().
CK_TILE_DEVICE void workgroup_barrier_wait()
{
#if defined(__gfx12__)
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    __builtin_amdgcn_sched_barrier(0);
#else
    block_sync_lds();
#endif
}

} // namespace ck_tile
