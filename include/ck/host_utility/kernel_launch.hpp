// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#ifndef __HIPCC_RTC__
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>

#include <vector>
#include <cassert>

#include "ck/ck.hpp"
#include "ck/utility/env.hpp"
#include "ck/stream_config.hpp"
#include "ck/host_utility/hip_check_error.hpp"

#include "hip_fix_ext.h"

template <typename... Args, typename F>
float launch_and_time_kernel(const StreamConfig& stream_config,
                             F kernel,
                             dim3 grid_dim,
                             dim3 block_dim,
                             std::size_t lds_byte,
                             Args... args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("%s: kernel %p, grid_dim {%u, %u, %u}, block_dim {%u, %u, %u} \n",
                   __func__,
                   reinterpret_cast<void*>(kernel),
                   grid_dim.x,
                   grid_dim.y,
                   grid_dim.z,
                   block_dim.x,
                   block_dim.y,
                   block_dim.z);

            printf("Warm up %d times\n", stream_config.cold_niters_);
        }
        if(!ck::EnvIsEnabled(CK_ENV(CK_NEWTIMING)))
        {
            // old algorithm: warm up then time nrepeat iterations and take the mean
            // warm up
            for(int i = 0; i < stream_config.cold_niters_; ++i)
            {
                kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
                hip_check_error(hipGetLastError());
            }
        }

        const int nrepeat = stream_config.nrepeat_;
        assert(nrepeat > 0);

        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("Start running %d times...\n", nrepeat);
        }

        hipEvent_t start, stop;

        hip_check_error(hipEventCreate(&start));
        hip_check_error(hipEventCreate(&stop));

        hip_check_error(hipDeviceSynchronize());
        if(!ck::EnvIsEnabled(CK_ENV(CK_NEWTIMING)))
        {
            hip_check_error(hipEventRecord(start, stream_config.stream_id_));

            for(int i = 0; i < nrepeat; ++i)
            {
                kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
                hip_check_error(hipGetLastError());
            }

            hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
            hip_check_error(hipEventSynchronize(stop));

            float total_time = 0;

            hip_check_error(hipEventElapsedTime(&total_time, start, stop));

            hip_check_error(hipEventDestroy(start));
            hip_check_error(hipEventDestroy(stop));

            return total_time / nrepeat;
        }
        // New algorithm: adaptive, no warmup, return minimum
        // Need to print all the values for stats.
        constexpr float maxtime = 10;   // ms
        constexpr int minrep = 2;
        std::vector<float> times;
        float total_time = 0;
        hip_check_error(hipDeviceSynchronize());
        for (int i = 0; i < nrepeat; ++i)
        {
            myhipExtLaunchKernelGGL(kernel,
                                    grid_dim,
                                    block_dim,
                                    lds_byte,
                                    stream_config.stream_id_,
                                    start,
                                    stop,
                                    0,
                                    args...);

            hip_check_error(hipGetLastError());
            hip_check_error(hipEventSynchronize(stop));

            float cur_time = 0;
            hip_check_error(hipEventElapsedTime(&cur_time, start, stop));
            times.push_back(cur_time);
            total_time += cur_time;
            if(total_time >= maxtime && i + 1 >= minrep)
                break;
        }
        printf("%s: kernel %p, grid_dim {%u, %u, %u}, block_dim {%u, %u, %u}, ",
               __func__,
               reinterpret_cast<void *>(kernel),
               grid_dim.x,
               grid_dim.y,
               grid_dim.z,
               block_dim.x,
               block_dim.y,
               block_dim.z);
        printf("times {");
        for (float t: times) printf(" %.7f", t);
        printf("}\n");

        return total_time / times.size();
    }
    else
    {
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        hip_check_error(hipGetLastError());

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
    hip_check_error(hipGetLastError());

    return 0;
#endif
}

template <typename... Args, typename F, typename PreProcessFunc>
float launch_and_time_kernel_with_preprocess(const StreamConfig& stream_config,
                                             PreProcessFunc preprocess,
                                             F kernel,
                                             dim3 grid_dim,
                                             dim3 block_dim,
                                             std::size_t lds_byte,
                                             Args... args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("%s: kernel %p, grid_dim {%u, %u, %u}, block_dim {%u, %u, %u} \n",
                   __func__,
                   reinterpret_cast<void*>(kernel),
                   grid_dim.x,
                   grid_dim.y,
                   grid_dim.z,
                   block_dim.x,
                   block_dim.y,
                   block_dim.z);

            printf("Warm up %d times\n", stream_config.cold_niters_);
        }
        if(!ck::EnvIsEnabled(CK_ENV(CK_NEWTIMING)))
        {
            // old algorithm: warm up then time nrepeat iterations and take the mean
            // warm up
            preprocess();
            for(int i = 0; i < stream_config.cold_niters_; ++i)
            {
                kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
                hip_check_error(hipGetLastError());
            }
        }

        const int nrepeat = stream_config.nrepeat_;
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("Start running %d times...\n", nrepeat);
        }
        hipEvent_t start, stop;

        hip_check_error(hipEventCreate(&start));
        hip_check_error(hipEventCreate(&stop));

        hip_check_error(hipDeviceSynchronize());

        if(!ck::EnvIsEnabled(CK_ENV(CK_NEWTIMING)))
        {
            hip_check_error(hipEventRecord(start, stream_config.stream_id_));

            for(int i = 0; i < nrepeat; ++i)
            {
                preprocess();
                kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
                hip_check_error(hipGetLastError());
            }

            hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
            hip_check_error(hipEventSynchronize(stop));

            float total_time = 0;

            hip_check_error(hipEventElapsedTime(&total_time, start, stop));

            hip_check_error(hipEventDestroy(start));
            hip_check_error(hipEventDestroy(stop));

            return total_time / nrepeat;
        }
        // New algorithm: adaptive, no warmup, return minimum
        // Need to print all the values for stats.
        constexpr float maxtime = 10;   // ms
        constexpr int minrep = 2;
        std::vector<float> times;
        float total_time = 0;
        hip_check_error(hipDeviceSynchronize());
        for (int i = 0; i < nrepeat; ++i)
        {
            preprocess();

            hipExtLaunchKernelGGL(kernel,
                                  grid_dim,
                                  block_dim,
                                  lds_byte,
                                  stream_config.stream_id_,
                                  start,
                                  stop,
                                  0,
                                  args...);
            hip_check_error(hipGetLastError());
            hip_check_error(hipEventSynchronize(stop));

            float cur_time = 0;
            hip_check_error(hipEventElapsedTime(&cur_time, start, stop));
            times.push_back(cur_time);
            total_time += cur_time;
            if(total_time >= maxtime && i + 1 >= minrep)
                break;
        }
        printf("%s: kernel %p, grid_dim {%u, %u, %u}, block_dim {%u, %u, %u}, ",
               __func__,
               reinterpret_cast<void *>(kernel),
               grid_dim.x,
               grid_dim.y,
               grid_dim.z,
               block_dim.x,
               block_dim.y,
               block_dim.z);
        printf("times {");
        for (float t: times) printf(" %.7f", t);
        printf("}\n");

        return total_time / times.size();
    }
    else
    {
        preprocess();
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        hip_check_error(hipGetLastError());

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
    hip_check_error(hipGetLastError());

    return 0;
#endif
}
#endif
