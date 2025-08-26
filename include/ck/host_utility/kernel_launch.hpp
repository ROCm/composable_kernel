// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#ifndef __HIPCC_RTC__
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/utility/env.hpp"
#include "ck/stream_config.hpp"
#include "ck/host_utility/hip_check_error.hpp"

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
            printf("%s: grid_dim {%u, %u, %u}, block_dim {%u, %u, %u} \n",
                   __func__,
                   grid_dim.x,
                   grid_dim.y,
                   grid_dim.z,
                   block_dim.x,
                   block_dim.y,
                   block_dim.z);

            printf("Warm up %d times\n", stream_config.cold_niters_);
        }
        // warm up
        for(int i = 0; i < stream_config.cold_niters_; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            std::cout << "From CK 1:" << std::endl;
            hip_check_error(hipGetLastError());
            std::cout << "From CK 2:" << std::endl;
        }

        const int nrepeat = stream_config.nrepeat_;
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("Start running %d times...\n", nrepeat);
        }
        hipEvent_t start, stop;

        std::cout << "From CK 3:" << std::endl;
        hip_check_error(hipEventCreate(&start));
        std::cout << "From CK 4:" << std::endl;
        hip_check_error(hipEventCreate(&stop));
        std::cout << "From CK 5:" << std::endl;

        hip_check_error(hipDeviceSynchronize());
        std::cout << "From CK 6:" << std::endl;
        hip_check_error(hipEventRecord(start, stream_config.stream_id_));
        std::cout << "From CK 7:" << std::endl;

        for(int i = 0; i < nrepeat; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            std::cout << "From CK 8:" << std::endl;
            hip_check_error(hipGetLastError());
            std::cout << "From CK 9:" << std::endl;
        }

        std::cout << "From CK 10:" << std::endl;
        hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
        std::cout << "From CK 11:" << std::endl;
        hip_check_error(hipEventSynchronize(stop));
        std::cout << "From CK 12:" << std::endl;
        float total_time = 0;

        hip_check_error(hipEventElapsedTime(&total_time, start, stop));
        std::cout << "From CK 13:" << std::endl;

        hip_check_error(hipEventDestroy(start));
        std::cout << "From CK 14:" << std::endl;
        hip_check_error(hipEventDestroy(stop));
        std::cout << "From CK 15:" << std::endl;

        return total_time / nrepeat;
    }
    else
    {
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        std::cout << "From CK 16:" << std::endl;
        hip_check_error(hipGetLastError());

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
    std::cout << "From CK 17:" << std::endl;
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
    std::cout << "From CK 20:" << std::endl;
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("%s: grid_dim {%u, %u, %u}, block_dim {%u, %u, %u} \n",
                   __func__,
                   grid_dim.x,
                   grid_dim.y,
                   grid_dim.z,
                   block_dim.x,
                   block_dim.y,
                   block_dim.z);

            printf("Warm up %d times\n", stream_config.cold_niters_);
        }
        // warm up
        preprocess();
        for(int i = 0; i < stream_config.cold_niters_; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
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
std::cout << "From CK 21:" << std::endl;
}
#endif
