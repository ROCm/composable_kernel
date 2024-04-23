// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <hip/hip_runtime.h>
#include <cstddef>

namespace ck_tile {
template <int MaxThreadPerBlock, int MinBlockPerCu, typename Kernel, typename... Args>
#if CK_TILE_USE_LAUNCH_BOUNDS
__launch_bounds__(MaxThreadPerBlock, MinBlockPerCu)
#endif
    __global__ void kentry(Kernel f, Args... args)
{
    f(args...);
}

template <typename... Args, typename F>
CK_TILE_HOST float launch_and_time_kernel(const stream_config& s,
                                          F kernel,
                                          dim3 grid_dim,
                                          dim3 block_dim,
                                          std::size_t lds_byte,
                                          Args... args)
{
#if CK_TILE_TIME_KERNEL
    if(s.time_kernel_)
    {
        // warm up
        for(int i = 0; i < s.cold_niters_; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, s.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        const int nrepeat = s.nrepeat_;
        hipEvent_t start, stop;

        HIP_CHECK_ERROR(hipEventCreate(&start));
        HIP_CHECK_ERROR(hipEventCreate(&stop));

        HIP_CHECK_ERROR(hipDeviceSynchronize());
        HIP_CHECK_ERROR(hipEventRecord(start, s.stream_id_));

        for(int i = 0; i < nrepeat; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, s.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        HIP_CHECK_ERROR(hipEventRecord(stop, s.stream_id_));
        HIP_CHECK_ERROR(hipEventSynchronize(stop));

        float total_time = 0;

        HIP_CHECK_ERROR(hipEventElapsedTime(&total_time, start, stop));

        return total_time / nrepeat;
    }
    else
    {
        kernel<<<grid_dim, block_dim, lds_byte, s.stream_id_>>>(args...);
        hip_check_error(hipGetLastError());
        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, s.stream_id_>>>(args...);
    hip_check_error(hipGetLastError());
    return 0;
#endif
}

template <typename... Args, typename F, typename PreProcessFunc>
CK_TILE_HOST float launch_and_time_kernel_with_preprocess(const stream_config& s,
                                                          PreProcessFunc preprocess,
                                                          F kernel,
                                                          dim3 grid_dim,
                                                          dim3 block_dim,
                                                          std::size_t lds_byte,
                                                          Args... args)
{
#if CK_TILE_TIME_KERNEL
    if(s.time_kernel_)
    {
#if CK_TILE_DEBUG_LOG
        printf("%s: grid_dim {%d, %d, %d}, block_dim {%d, %d, %d} \n",
               __func__,
               grid_dim.x,
               grid_dim.y,
               grid_dim.z,
               block_dim.x,
               block_dim.y,
               block_dim.z);

        printf("Warm up 1 time\n");
#endif
        // warm up
        preprocess();
        kernel<<<grid_dim, block_dim, lds_byte, s.stream_id_>>>(args...);
        hip_check_error(hipGetLastError());

        const int nrepeat = 10;
#if CK_TILE_DEBUG_LOG
        printf("Start running %d times...\n", nrepeat);
#endif
        hipEvent_t start, stop;

        HIP_CHECK_ERROR(hipEventCreate(&start));
        HIP_CHECK_ERROR(hipEventCreate(&stop));

        HIP_CHECK_ERROR(hipDeviceSynchronize());
        HIP_CHECK_ERROR(hipEventRecord(start, s.stream_id_));

        for(int i = 0; i < nrepeat; ++i)
        {
            preprocess();
            kernel<<<grid_dim, block_dim, lds_byte, s.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        HIP_CHECK_ERROR(hipEventRecord(stop, s.stream_id_));
        HIP_CHECK_ERROR(hipEventSynchronize(stop));

        float total_time = 0;

        HIP_CHECK_ERROR(hipEventElapsedTime(&total_time, start, stop));

        return total_time / nrepeat;
    }
    else
    {
        preprocess();
        kernel<<<grid_dim, block_dim, lds_byte, s.stream_id_>>>(args...);
        hip_check_error(hipGetLastError());

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, s.stream_id_>>>(args...);
    hip_check_error(hipGetLastError());

    return 0;
#endif
}

template <int MaxThreadPerBlock = CK_TILE_MAX_THREAD_PER_BLOCK,
          int MinBlockPerCu     = CK_TILE_MIN_BLOCK_PER_CU,
          typename KernelImpl,
          typename... Args>
CK_TILE_HOST float launch_kernel(const stream_config& s,
                                 KernelImpl kernel_impl,
                                 dim3 grid_dim,
                                 dim3 block_dim,
                                 std::size_t dynamic_smem_byte,
                                 Args... args)
{
    const auto kernel = kentry<MaxThreadPerBlock, MinBlockPerCu, KernelImpl, Args...>;

    return launch_and_time_kernel(
        s, kernel, grid_dim, block_dim, dynamic_smem_byte, kernel_impl, args...);
}
} // namespace ck_tile
