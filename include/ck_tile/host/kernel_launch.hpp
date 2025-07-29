// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/ignore.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/host/timer.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>

namespace ck_tile {

#define LOW_CU_PROCESSORS 80
#define HIGH_CU_PROCESSORS 228
#define OPTIMAL_LATENCY_LOW_CU_PROCESSORS 0.005
#define OPTIMAL_LATENCY_HIGH_CU_PROCESSORS 0.0015
#define OPTIMAL_LATENCY_SAFE_MARGIN 0.01

template <int MaxThreadPerBlock, int MinBlockPerCu, typename Kernel, typename... Args>
#if CK_TILE_USE_LAUNCH_BOUNDS
__launch_bounds__(MaxThreadPerBlock, MinBlockPerCu)
#endif
    __global__ void kentry(Args... args)
{
#if defined(__HIP_DEVICE_COMPILE__)
    Kernel{}(args...);
#else
    (..., (ignore = args, 0));
#endif
}

//
// return a anonymous functor(lambda) to be called later
// the KernelImpl should be a class without non-static data member, or let's say
// can be instantiate with "KernelImpl{}"
//
// the "static __device__ operator()(some_arg)" is the entry point of KernelImpl
//
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

template <typename... Callables>
CK_TILE_HOST void launch_and_check(const stream_config& sc, Callables&&... callables)
{
    // abort the sequence in case of intermediate error
    if(!((static_cast<void>(callables(sc)), hipPeekAtLastError() == hipSuccess) && ...))
    {
        HIP_CHECK_ERROR(hipGetLastError());
    }
}

// clang-format off
/*
 * launch_kernel()
 *
 * this is the function to launch arbitrary number of kernels with optional timer(selected by stream_config)
 * the callables should have signature as "operator()(const stream_config& s){ ... }" to call
 * 
 * the simplest way is pass in a lambda function, with "[=](const stream_config& s){ call_your_kernel_here() }"
 * as signature, for the callable (pay attention to the capture list)
 * 
 * e.g.
 *  ck_tile::launch_kernel(s,
 *                      [=](const stream_config& s){ hipMemset(ptr, 0, size) },
 *                      [=](const stream_config& s){ some_kernel<<<grids, blocks>>>(arg); }
 *                      );
 * 
 * if you use ck_tile kernel, or similiar to this style (structure with "static __device__ operator()(...){}")
 * you can pass your kernel to ck_tile::make_kernel(), which will create a anonymous functor for you,
 * then pass it to ck_tile::launch_kernel()
 * 
 * e.g.
 *  ck_tile::launch_kernel(s,
 *                      ck_tile::make_kernel<T0, B0>(kernel_0{}, grids0, blocks0, 0, kargs0),
 *                      ck_tile::make_kernel<T0, B1>(kernel_1{}, grids1, blocks1, 0, kargs1),
 *                       ...);
 **/
// clang-format on
template <typename... Callables>
CK_TILE_HOST float launch_kernel(const stream_config& s, Callables&&... callables)
{
    static_assert(sizeof...(callables) > 0, "At least one callable is required!");

    if(!s.time_kernel_)
    {
        launch_and_check(s, std::forward<Callables>(callables)...);
        return 0;
    }

    auto time_launches = [&](auto timer) {
        // Warmup
        for(int i = 0; i < s.cold_niters_; i++)
        {
            launch_and_check(s, std::forward<Callables>(callables)...);
        }

        timer.start(s.stream_id_);
        for(int i = 0; i < s.nrepeat_; i++)
        {
            launch_and_check(s, std::forward<Callables>(callables)...);
        }
        timer.stop(s.stream_id_);

        return timer.duration() / s.nrepeat_;
    };

    if(s.is_gpu_timer_)
    {
        return time_launches(gpu_timer{});
    }
    else
    {
        return time_launches(cpu_timer{});
    }
}

template <typename PreprocessFunc, typename... Callables>
CK_TILE_HOST float launch_kernel_preprocess(const stream_config& s,
                                            PreprocessFunc preprocess,
                                            Callables&&... callables)
{
    static_assert(sizeof...(callables) > 0, "At least one callable is required!");

    if(!s.time_kernel_)
    {
        preprocess();
        launch_and_check(s, std::forward<Callables>(callables)...);
        return 0;
    }

    auto time_launches = [&](auto timer) {
        // Warmup
        for(int i = 0; i < s.cold_niters_; i++)
        {
            launch_and_check(s, std::forward<Callables>(callables)...);
        }

        timer.start(s.stream_id_);
        for(int i = 0; i < s.nrepeat_; i++)
        {
            preprocess();
            launch_and_check(s, std::forward<Callables>(callables)...);
        }
        timer.stop(s.stream_id_);

        hipDeviceProp_t deviceProps;
        HIP_CHECK_ERROR(hipGetDeviceProperties(&deviceProps, 0));

        float preprocess_offset = (deviceProps.multiProcessorCount >= HIGH_CU_PROCESSORS)
                                      ? OPTIMAL_LATENCY_HIGH_CU_PROCESSORS
                                  : (deviceProps.multiProcessorCount == LOW_CU_PROCESSORS)
                                      ? OPTIMAL_LATENCY_LOW_CU_PROCESSORS
                                      : OPTIMAL_LATENCY_SAFE_MARGIN;
        return (timer.duration() - preprocess_offset * s.nrepeat_) / s.nrepeat_;
    };

    if(s.is_gpu_timer_)
    {
        return time_launches(gpu_timer{});
    }
    else
    {
        return time_launches(cpu_timer{});
    }
}
} // namespace ck_tile
