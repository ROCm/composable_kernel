// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <numeric>
#include <functional>
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

template <class it>
typename std::iterator_traits<it>::value_type median(it begin, it end)
{
    if(begin == end)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    auto n  = std::distance(begin, end);
    auto n2 = n / 2;
    std::nth_element(begin, begin + n2, end);
    return (n % 2) ? begin[n2] : (*std::max_element(begin, begin + n2) + begin[n2]) / 2.0;
}

inline void remove_outliers(std::vector<float>& v)
{
    // 1.5x IQR method to detect and remove outliers
    auto n2 = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n2, v.end());
    auto q1  = median(v.begin(), v.begin() + n2);
    auto q3  = median(v.begin() + ((v.size() % 2) ? n2 + 1 : n2), v.end());
    auto iqr = q3 - q1;
    auto lb  = q1 - 1.5 * iqr;
    auto ub  = q3 + 1.5 * iqr;
    v.erase(std::remove_if(v.begin(), v.end(), [&](float f) { return f < lb || f > ub; }), v.end());
}

template <typename TimerType, typename CallablesFunc>
CK_TILE_HOST double timing_loop_impl(TimerType timer,
                                     const stream_config& s,
                                     CallablesFunc&& callables_func,
                                     std::function<void()> preprocess = nullptr)
{
    for(int i = 0; i < s.cold_niters_; i++)
    {
        callables_func();
    }

    float per_iter_time = 0.f;
    std::vector<float> times;
    int i = 0;
    while(i < s.nrepeat_ || per_iter_time < s.bench_time_ms_)
    {
        if(preprocess)
            preprocess();

        timer.start(s.stream_id_, i);
        callables_func();
        timer.stop(s.stream_id_, i);

        if(i > 0)
        {
            per_iter_time = timer.duration(i - 1);
            times.push_back(per_iter_time);
            per_iter_time = timer.is_exceed(i - 1);
        }
        i++;
    }

    if(!i)
        return 0.;

    per_iter_time = timer.duration(i - 1);
    times.push_back(per_iter_time);
    remove_outliers(times);
    return std::accumulate(times.begin(), times.end(), 0.) / times.size();
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

    auto callables_func = [&]() { launch_and_check(s, std::forward<Callables>(callables)...); };

    if(s.is_gpu_timer_)
    {
        return timing_loop_impl(gpu_timer_new{s.stream_id_}, s, callables_func);
    }
    else
    {
        return timing_loop_impl(cpu_timer{}, s, callables_func);
    }
}

template <typename PreprocessFunc, typename... Callables>
CK_TILE_HOST float
launch_kernel_time_mask(const stream_config& s, PreprocessFunc preprocess, Callables&&... callables)
{
    static_assert(sizeof...(callables) > 0, "At least one callable is required!");

    if(!s.time_kernel_)
    {
        preprocess();
        launch_and_check(s, std::forward<Callables>(callables)...);
        return 0;
    }

    auto callables_func = [&]() { launch_and_check(s, std::forward<Callables>(callables)...); };

    if(s.is_gpu_timer_)
    {
        return timing_loop_impl(gpu_timer_new{s.stream_id_}, s, callables_func, preprocess);
    }
    else
    {
        return timing_loop_impl(cpu_timer{}, s, callables_func, preprocess);
    }
}
} // namespace ck_tile
