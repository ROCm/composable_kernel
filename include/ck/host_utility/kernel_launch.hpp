// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#ifndef __HIPCC_RTC__
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/utility/env.hpp"
#include "ck/stream_config.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/host_utility/flush_cache.hpp"

namespace ck {

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
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
        hip_check_error(hipEventSynchronize(stop));

        float total_time = 0;

        hip_check_error(hipEventElapsedTime(&total_time, start, stop));

        // hipEventElapsedTime can return a small negative value on Windows for a
        // very fast kernel. Clamp to zero, as negative elapsed time is never physical.
        if(total_time < 0)
            total_time = 0;

        hip_check_error(hipEventDestroy(start));
        hip_check_error(hipEventDestroy(stop));

        return total_time / nrepeat;
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

// Cluster launch overload: launches the kernel with hipLaunchKernelEx using
// hipLaunchAttributeClusterDimension. The cluster_dim parameter specifies
// how many WGPs form a cluster.
#if CK_ENABLE_CLUSTER_LAUNCH
template <typename... Args, typename F>
float launch_and_time_kernel(const StreamConfig& stream_config,
                             F kernel,
                             dim3 grid_dim,
                             dim3 cluster_dim,
                             dim3 block_dim,
                             std::size_t lds_byte,
                             Args... args)
{
    const auto launch_cluster_kernel = [&]() {
        hipLaunchConfig_t config{};
        config.gridDim          = grid_dim;
        config.blockDim         = block_dim;
        config.dynamicSmemBytes = lds_byte;
        config.stream           = stream_config.stream_id_;

        hipLaunchAttribute attrs[1] = {};
        attrs[0].id                 = hipLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x   = cluster_dim.x;
        attrs[0].val.clusterDim.y   = cluster_dim.y;
        attrs[0].val.clusterDim.z   = cluster_dim.z;
        config.attrs                = attrs;
        config.numAttrs             = 1;

        hip_check_error(hipLaunchKernelEx(&config, kernel, args...));
    };

#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("%s: cluster_dim {%u, %u, %u}, grid_dim {%u, %u, %u}, "
                   "block_dim {%u, %u, %u}\n",
                   __func__,
                   cluster_dim.x,
                   cluster_dim.y,
                   cluster_dim.z,
                   grid_dim.x,
                   grid_dim.y,
                   grid_dim.z,
                   block_dim.x,
                   block_dim.y,
                   block_dim.z);

            printf("Warm up %d times\n", stream_config.cold_niters_);
        }

        for(int i = 0; i < stream_config.cold_niters_; ++i)
        {
            launch_cluster_kernel();
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
            launch_cluster_kernel();
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
        launch_cluster_kernel();
        return 0;
    }
#else
    launch_cluster_kernel();
    return 0;
#endif
}
#endif // CK_ENABLE_CLUSTER_LAUNCH

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
}

template <typename... Args, typename F, typename PreProcessFunc>
float launch_and_time_kernel_with_preprocess_flush_cache(const StreamConfig& stream_config,
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
        auto run_flush_cache = [&]() { ck::utility::flush_icache(); };
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
        // Warm up
        preprocess();
        for(int i = 0; i < stream_config.cold_niters_; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }
        float total_time = 0, flush_cache_total_time = 0;
        const int nrepeat = stream_config.nrepeat_;
        // Main timing loop
        {
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
                run_flush_cache();
                preprocess();
                kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
                hip_check_error(hipGetLastError());
            }

            hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
            hip_check_error(hipEventSynchronize(stop));

            hip_check_error(hipEventElapsedTime(&total_time, start, stop));

            hip_check_error(hipEventDestroy(start));
            hip_check_error(hipEventDestroy(stop));
        }
        // Flush cache timing loop
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                printf("Profile flush cache %d times...\n", nrepeat);
            }
            hipEvent_t start, stop;

            hip_check_error(hipEventCreate(&start));
            hip_check_error(hipEventCreate(&stop));

            hip_check_error(hipDeviceSynchronize());
            hip_check_error(hipEventRecord(start, stream_config.stream_id_));

            for(int i = 0; i < nrepeat; ++i)
            {
                run_flush_cache();
            }

            hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
            hip_check_error(hipEventSynchronize(stop));

            hip_check_error(hipEventElapsedTime(&flush_cache_total_time, start, stop));

            hip_check_error(hipEventDestroy(start));
            hip_check_error(hipEventDestroy(stop));
        }
        // Exclude flush cache from result
        return (total_time - flush_cache_total_time) / nrepeat;
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

template <typename... Args, typename F>
float launch_and_time_kernel_flush_cache(const StreamConfig& stream_config,
                                         F kernel,
                                         dim3 grid_dim,
                                         dim3 block_dim,
                                         std::size_t lds_byte,
                                         Args... args)
{
    auto preprocess = [&]() {};
    return launch_and_time_kernel_with_preprocess_flush_cache(
        stream_config, preprocess, kernel, grid_dim, block_dim, lds_byte, args...);
}

} // namespace ck

#endif
