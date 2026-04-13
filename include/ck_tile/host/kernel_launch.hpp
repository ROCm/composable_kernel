// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <numeric>
#include <functional>
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/ignore.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/host/timer.hpp"
#include "ck_tile/host/flush_icache.hpp"
#include "ck_tile/host/rotating_buffers.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

namespace ck_tile {

template <typename T, typename = void>
inline constexpr bool kattr_no_packed_fp32_ops_v = false;
template <typename T>
inline constexpr bool
    kattr_no_packed_fp32_ops_v<T, std::void_t<decltype(T::kattr_no_packed_fp32_ops)>> =
        T::kattr_no_packed_fp32_ops;

// TODO: rename to something more specific (e.g. kernel_attr_no_packed_fp32) since
// kernel_attr<bool> only controls the no-packed-fp32-ops flag, not a general attribute bag.
template <bool no_packed_fp32_ops>
struct kernel_attr
{
    // The kernel function attribute "no-packed-fp32-ops": Disable the use of packed FP32
    // instructions so that they can be co-executed with matrix operations
    static constexpr bool kattr_no_packed_fp32_ops = no_packed_fp32_ops;
};

// Compose an architecture tag with kernel attributes.
// Inherits ArchTag for symbol mangling and adds attribute flags.
// kernel_attr_for<gfx950_t>              -> gfx950_t  (identity)
// kernel_attr_for<gfx950_t, kernel_attr<true>> -> unique type with attribute
namespace detail {
template <typename ArchTag, typename... Attrs>
struct kernel_attr_for_impl : ArchTag, Attrs...
{
};

template <typename ArchTag, typename... Attrs>
struct kernel_attr_for_helper
{
    using type = kernel_attr_for_impl<ArchTag, Attrs...>;
};

template <typename ArchTag>
struct kernel_attr_for_helper<ArchTag>
{
    using type = ArchTag;
};
} // namespace detail

template <typename ArchTag, typename... Attrs>
using kernel_attr_for = typename detail::kernel_attr_for_helper<ArchTag, Attrs...>::type;

#if CK_TILE_USE_LAUNCH_BOUNDS
#define KENTRY_LAUNCH_BOUNDS __launch_bounds__(Kernel::kBlockSize, MinBlockPerCu)
#else
#define KENTRY_LAUNCH_BOUNDS
#endif
#if defined(__HIP_DEVICE_COMPILE__)
#define KENTRY_BODY Kernel{}(args...)
#define KENTRY_ATTR_NO_PACKED_FP32_OPS __attribute__((target("no-packed-fp32-ops")))
#else
#define KENTRY_BODY (..., (ignore = args, 0))
#define KENTRY_ATTR_NO_PACKED_FP32_OPS
#endif

template <int MinBlockPerCu, typename Kernel, typename... Args>
KENTRY_LAUNCH_BOUNDS __global__ void kentry(Args... args)
{
    KENTRY_BODY;
}
template <typename Attr, int MinBlockPerCu, typename Kernel, typename... Args>
KENTRY_LAUNCH_BOUNDS __global__ //
    std::enable_if_t<!kattr_no_packed_fp32_ops_v<Attr>>
    kentry(Args... args)
{
    KENTRY_BODY;
}
template <typename Attr, int MinBlockPerCu, typename Kernel, typename... Args>
KENTRY_LAUNCH_BOUNDS KENTRY_ATTR_NO_PACKED_FP32_OPS __global__ //
    std::enable_if_t<kattr_no_packed_fp32_ops_v<Attr>>
    kentry(Args... args)
{
    KENTRY_BODY;
}

#undef KENTRY_LAUNCH_BOUNDS
#undef KENTRY_BODY
#undef KENTRY_ATTR_NO_PACKED_FP32_OPS

//
// return a anonymous functor(lambda) to be called later
// the KernelImpl should be a class without non-static data member, or let's say
// can be instantiate with "KernelImpl{}"
//
// the "static __device__ operator()(some_arg)" is the entry point of KernelImpl
//
// Attr can be used to support linking multiple object files that have the same kernel compiled for
// different architectures. In this case each object file has to use a different tag (gfx9_t,
// gfx12_t etc.), so the kernel will have different symbols for each architecture. It can also be
// used to pass some compile-time attributes to the kernel.
template <int MinBlockPerCu = CK_TILE_MIN_BLOCK_PER_CU,
          typename Attr     = void,
          typename KernelImpl,
          typename... Args>
CK_TILE_HOST auto make_kernel(KernelImpl /*f*/,
                              dim3 grid_dim,
                              dim3 block_dim,
                              std::size_t lds_byte,
                              [[clang::lifetimebound]] Args... args)
{
    const auto kernel = []() {
        if constexpr(std::is_void_v<Attr>)
            return kentry<MinBlockPerCu, KernelImpl, Args...>;
        else
            return kentry<Attr, MinBlockPerCu, KernelImpl, Args...>;
    }();
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

// Measure the preprocess time during the cold iterations
template <typename TimerType, typename PreprocessFunc>
CK_TILE_HOST double
preprocess_profiling_impl(TimerType timer, const stream_config& s, PreprocessFunc preprocess)
{
    timer.start(s.stream_id_);
    for(int i = 0; i < s.nrepeat_; i++)
    {
        if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
        {
            preprocess();
        }
    }
    timer.stop(s.stream_id_);

    return timer.duration() / s.nrepeat_;
}

template <typename TimerType, typename CallablesFunc, typename PreprocessFunc = std::nullptr_t>
CK_TILE_HOST double timing_loop_flush_cache_impl(TimerType timer,
                                                 const stream_config& s,
                                                 CallablesFunc&& callables_func,
                                                 PreprocessFunc preprocess = nullptr)
{
    auto run_flush_cache = [&]() { ck_tile::flush_icache(); };
    // Warm up
    for(int i = 0; i < s.cold_niters_; i++)
    {
        if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
        {
            preprocess();
        }
        callables_func();
    }
    // Main timing loop
    int i = 0;
    timer.start(s.stream_id_);
    while(i < s.nrepeat_)
    {
        run_flush_cache();
        if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
        {
            preprocess();
        }

        callables_func();
        i++;
    }
    timer.stop(s.stream_id_);
    // Flush cache timing loop
    auto flush_cache_time = preprocess_profiling_impl(gpu_timer{}, s, run_flush_cache);
    if(i == 0)
    {
        return 0.;
    }
    // Exclude flush cache from result
    return (timer.duration() / s.nrepeat_) - flush_cache_time;
}

template <typename TimerType, typename CallablesFunc, typename PreprocessFunc = std::nullptr_t>
CK_TILE_HOST double timing_loop_impl(TimerType timer,
                                     const stream_config& s,
                                     CallablesFunc&& callables_func,
                                     PreprocessFunc preprocess = nullptr)
{
    for(int i = 0; i < s.cold_niters_; i++)
    {
        if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
        {
            preprocess();
        }
        callables_func();
    }

    int i = 0;
    timer.start(s.stream_id_);
    while(i < s.nrepeat_)
    {
        if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
        {
            preprocess();
        }

        callables_func();
        i++;
    }
    timer.stop(s.stream_id_);

    if(i == 0)
        return 0.;
    return timer.duration() / s.nrepeat_;
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
        return timing_loop_impl(gpu_timer{}, s, callables_func);
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
        return timing_loop_impl(gpu_timer{}, s, callables_func, preprocess);
    }
    else
    {
        return timing_loop_impl(cpu_timer{}, s, callables_func, preprocess);
    }
}

template <typename PreprocessFunc, typename... Callables>
CK_TILE_HOST float launch_kernel_time_mask_flush_cache(const stream_config& s,
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

    auto callables_func = [&]() { launch_and_check(s, std::forward<Callables>(callables)...); };

    if(s.is_gpu_timer_)
    {
        return timing_loop_flush_cache_impl(gpu_timer{}, s, callables_func, preprocess);
    }
    else
    {
        return timing_loop_flush_cache_impl(cpu_timer{}, s, callables_func, preprocess);
    }
}

} // namespace ck_tile

#pragma clang diagnostic pop
