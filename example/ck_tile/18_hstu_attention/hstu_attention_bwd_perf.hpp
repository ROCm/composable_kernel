// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include "ck_tile/host/timer.hpp"

// HSTU attention backward — M8 MI (measurement infrastructure) helper.
//
// `time_op` wraps a single kernel/op launch on `stream`. It is the ONLY timing
// primitive the bwd dispatches use so per-kernel attribution (PRE / memset / MAIN
// / POST) is uniform.
//
//   measure == false  ->  run `fn()` EXACTLY ONCE and return 0. This is the normal
//                          (non -perf) path: byte-identical host behavior to a bare
//                          launch, and crucially it never touches the DEVICE code
//                          (the kernel template / kargs are unchanged), so the
//                          generated gfx950 device symbols are byte-identical to a
//                          build without MI (co_symbols zero-regression gate).
//   measure == true   ->  `cold` warmup launches, then time `nrep` launches with a
//                          hipEvent gpu_timer and return the mean per-launch ms.
//
// Repeating MAIN without re-zeroing dq_acc is fine for a *timing* measurement: the
// per-launch kernel runtime is what we report, and the harness has already pulled
// the validated dQ/dK/dV to host before any -perf re-run (see harness).
namespace hstu_bwd_perf {

template <typename Fn>
inline float time_op(bool measure, hipStream_t stream, Fn&& fn, int cold = 3, int nrep = 10)
{
    if(!measure)
    {
        fn();
        return 0.f;
    }
    for(int i = 0; i < cold; ++i)
        fn();
    ck_tile::gpu_timer timer{};
    timer.start(stream);
    for(int i = 0; i < nrep; ++i)
        fn();
    timer.stop(stream);
    return timer.duration() / static_cast<float>(nrep);
}

} // namespace hstu_bwd_perf
