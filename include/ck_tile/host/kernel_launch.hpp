// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/timer.hpp"
#include <hip/hip_runtime.h>
#include <cstddef>

namespace ck_tile {
template <int MaxThreadPerBlock, int MinBlockPerCu, typename Kernel, typename... Args>
#if CK_TILE_USE_LAUNCH_BOUNDS
__launch_bounds__(MaxThreadPerBlock, MinBlockPerCu)
#endif
    __global__ void kentry(Args... args)
{
    Kernel{}(args...);
}

template <int MaxThreadPerBlock = CK_TILE_MAX_THREAD_PER_BLOCK,
          int MinBlockPerCu     = CK_TILE_MIN_BLOCK_PER_CU,
          typename KernelImpl,
          typename... Args>
CK_TILE_HOST auto
make_kernel(KernelImpl /*f*/, dim3 grid_dim, dim3 block_dim, std::size_t lds_byte, Args... args)
{
    const auto kernel = kentry<MaxThreadPerBlock, MinBlockPerCu, KernelImpl, Args...>;

    return [=](const stream_config& s) {
        kernel<<<grid_dim, block_dim, lds_byte, s.stream_id_>>>(args...);
    };
}

template <typename Timer, typename... Callables>
CK_TILE_HOST float launch_with_timer_(const stream_config& s, Callables... callables)
{
    // TODO: assume the s.
    Timer timer{};

    // warmup
    for(int i = 0; i < s.cold_niters_; i++)
    {
        (callables(s), ...);
    }
    hip_check_error(hipGetLastError());

    timer.start(s.stream_id_);
    // repeat
    for(int i = 0; i < s.nrepeat_; i++)
    {
        (callables(s), ...);
    }
    hip_check_error(hipGetLastError());
    timer.stop(s.stream_id_);

    float ms = timer.duration();
    return ms / s.nrepeat_;
}

/*
 * launch_kernel()
 * this is the function to launch arbitrary kernels with potential timer(selected by stream_config)
 *
 * the callables should have signature as "operator()(const stream_config& s){ ... }" to call
 * ck_tile::launch_kernel(s,
 *                       ck_tile::make_kernel<ThreadPerBlock0, BlockPerCu0>(ck_kernel_0{}, grids0,
 *blocks0, 0, kargs0), ck_tile::make_kernel<ThreadPerBlock1, BlockPerCu1>(ck_kernel_1{}, grids1,
 *blocks1, 0, kargs1),
 *                       ...);
 **/

template <typename... Callables>
CK_TILE_HOST float launch_kernel(const stream_config& s, Callables... callables)
{
    // clang-format off
    if(!s.time_kernel_) {
        (callables(s),...); hip_check_error(hipGetLastError());
        return 0;
    }
    if(s.is_gpu_timer_) {
        gpu_timer timer {};

        // warmup
        for(int i = 0; i < s.cold_niters_; i++) { (callables(s),...); } hip_check_error(hipGetLastError());

        timer.start(s.stream_id_);
        for(int i = 0; i < s.nrepeat_; i++) { (callables(s),...); } hip_check_error(hipGetLastError());
        timer.stop(s.stream_id_);

        return timer.duration() / s.nrepeat_;
    }
    else {
        cpu_timer timer {};

        // warmup
        for(int i = 0; i < s.cold_niters_; i++) { (callables(s),...); } hip_check_error(hipGetLastError());

        timer.start(s.stream_id_);
        for(int i = 0; i < s.nrepeat_; i++) { (callables(s),...); } hip_check_error(hipGetLastError());
        timer.stop(s.stream_id_);

        return timer.duration() / s.nrepeat_;
    }
    // clang-format on
}

} // namespace ck_tile
