// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/host_utility/hip_check_error.hpp"

template <typename Args, typename F, typename PreProcessFunc>
float launch_and_time_kernel_flush_cache(const StreamConfig& stream_config,
                                         PreProcessFunc preprocess,
                                         F kernel,
                                         dim3 grid_dim,
                                         dim3 block_dim,
                                         std::size_t lds_byte,
                                         Args args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
#if DEBUG_LOG
        printf("%s: grid_dim {%d, %d, %d}, block_dim {%d, %d, %d} \n",
               __func__,
               grid_dim.x,
               grid_dim.y,
               grid_dim.z,
               block_dim.x,
               block_dim.y,
               block_dim.z);

        printf("Warm up %d times\n", stream_config.cold_niters_);
#endif
        using ADataType = decltype(args.p_a_grid);
        using BDataType = decltype(args.p_b_grid);
        using CDataType = decltype(args.p_c_grid);

        const char* p_a_grid = reinterpret_cast<const char*>(args.p_a_grid);
        const char* p_b_grid = reinterpret_cast<const char*>(args.p_b_grid);
        char* p_c_grid       = reinterpret_cast<char*>(args.p_c_grid);
        // warm up
        for(int i = 0; i < stream_config.cold_niters_; ++i)
        {
            if(stream_config.flush_icache)
                preprocess();

            if(stream_config.rotating_count > 1)
            {
                int idx       = i % stream_config.rotating_count;
                args.p_a_grid = reinterpret_cast<ADataType>(p_a_grid + idx * stream_config.size_a);
                args.p_b_grid = reinterpret_cast<BDataType>(p_b_grid + idx * stream_config.size_b);
                args.p_c_grid = reinterpret_cast<CDataType>(p_c_grid + idx * stream_config.size_c);
            }
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args);
            hip_check_error(hipGetLastError());
        }

        const int nrepeat = stream_config.nrepeat_;
#if DEBUG_LOG
        printf("Start running %d times...\n", nrepeat);
#endif

        float total_time = 0;
        for(int i = 0; i < nrepeat; ++i)
        {
            if(stream_config.flush_icache)
                preprocess();

            hipEvent_t start, stop;

            hip_check_error(hipEventCreate(&start));
            hip_check_error(hipEventCreate(&stop));

            hip_check_error(hipDeviceSynchronize());
            hip_check_error(hipEventRecord(start, stream_config.stream_id_));
            // run real kernel
            if(stream_config.rotating_count > 1)
            {
                int idx       = (stream_config.cold_niters_ + i) % stream_config.rotating_count;
                args.p_a_grid = reinterpret_cast<ADataType>(p_a_grid + idx * stream_config.size_a);
                args.p_b_grid = reinterpret_cast<BDataType>(p_b_grid + idx * stream_config.size_b);
                args.p_c_grid = reinterpret_cast<CDataType>(p_c_grid + idx * stream_config.size_c);
            }
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args);
            hip_check_error(hipGetLastError());
            // end real kernel

            hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
            hip_check_error(hipEventSynchronize(stop));
            float cur_time = 0;
            hip_check_error(hipEventElapsedTime(&cur_time, start, stop));
            total_time += cur_time;
        }

        return total_time / nrepeat;
    }
    else
    {
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args);
        hip_check_error(hipGetLastError());

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args);
    hip_check_error(hipGetLastError());

    return 0;
#endif
}
