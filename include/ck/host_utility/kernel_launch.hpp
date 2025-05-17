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

template <typename Args>
float launch_and_time_kernel_from_module(const StreamConfig& stream_config,
                                         std::string hsa_dir,
                                         std::string kernel_name,
                                         dim3 grid_dim,
                                         dim3 block_dim,
                                         std::size_t lds_byte,
                                         Args args)
{
    hipModule_t module;
    hipFunction_t kernel_func;
    // printf("hsa_dir: %s, func: %s,\n", hsa_dir.c_str(), kernel_name.c_str());
    hip_check_error(hipModuleLoad(&module, hsa_dir.c_str()));
    hip_check_error(hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));
    auto arg_size  = sizeof(args);
    auto arg_ptr   = args;
    void* config[] = {reinterpret_cast<void*>(0x1),
                      reinterpret_cast<void*>(&arg_ptr),
                      reinterpret_cast<void*>(0x2),
                      &arg_size,
                      reinterpret_cast<void*>(0x3)};
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
            hip_check_error(hipModuleLaunchKernel(kernel_func,
                                                  grid_dim.x,
                                                  grid_dim.y,
                                                  grid_dim.z,
                                                  block_dim.x,
                                                  block_dim.y,
                                                  block_dim.z,
                                                  lds_byte,
                                                  stream_config.stream_id_,
                                                  nullptr,
                                                  reinterpret_cast<void**>(&config)));
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
            hip_check_error(hipModuleLaunchKernel(kernel_func,
                                                  grid_dim.x,
                                                  grid_dim.y,
                                                  grid_dim.z,
                                                  block_dim.x,
                                                  block_dim.y,
                                                  block_dim.z,
                                                  lds_byte,
                                                  stream_config.stream_id_,
                                                  nullptr,
                                                  reinterpret_cast<void**>(&config)));
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
        hip_check_error(hipModuleLaunchKernel(kernel_func,
                                              grid_dim.x,
                                              grid_dim.y,
                                              grid_dim.z,
                                              block_dim.x,
                                              block_dim.y,
                                              block_dim.z,
                                              lds_byte,
                                              stream_config.stream_id_,
                                              nullptr,
                                              reinterpret_cast<void**>(&config)));

        return 0;
    }
#else
    hip_check_error(hipModuleLaunchKernel(kernel_func,
                                          grid_dim.x,
                                          grid_dim.y,
                                          grid_dim.z,
                                          block_dim.x,
                                          block_dim.y,
                                          block_dim.z,
                                          lds_byte,
                                          stream_config.stream_id_,
                                          nullptr,
                                          reinterpret_cast<void**>(&config)));

    return 0;
#endif
}

template <typename Args, typename PreProcessFunc>
float launch_and_time_kernel_from_module_with_preprocess(const StreamConfig& stream_config,
                                                         PreProcessFunc preprocess,
                                                         std::string hsa_dir,
                                                         std::string kernel_name,
                                                         dim3 grid_dim,
                                                         dim3 block_dim,
                                                         std::size_t lds_byte,
                                                         Args args)
{
    hipModule_t module;
    hipFunction_t kernel_func;

    hip_check_error(hipModuleLoad(&module, hsa_dir.c_str()));
    hip_check_error(hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));
    auto arg_size  = sizeof(args);
    auto arg_ptr   = args;
    void* config[] = {reinterpret_cast<void*>(0x1),
                      reinterpret_cast<void*>(&arg_ptr),
                      reinterpret_cast<void*>(0x2),
                      &arg_size,
                      reinterpret_cast<void*>(0x3)};
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
            hip_check_error(hipModuleLaunchKernel(kernel_func,
                                                  grid_dim.x,
                                                  grid_dim.y,
                                                  grid_dim.z,
                                                  block_dim.x,
                                                  block_dim.y,
                                                  block_dim.z,
                                                  lds_byte,
                                                  stream_config.stream_id_,
                                                  nullptr,
                                                  reinterpret_cast<void**>(&config)));
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
            hip_check_error(hipModuleLaunchKernel(kernel_func,
                                                  grid_dim.x,
                                                  grid_dim.y,
                                                  grid_dim.z,
                                                  block_dim.x,
                                                  block_dim.y,
                                                  block_dim.z,
                                                  lds_byte,
                                                  stream_config.stream_id_,
                                                  nullptr,
                                                  reinterpret_cast<void**>(&config)));
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
        hip_check_error(hipModuleLaunchKernel(kernel_func,
                                              grid_dim.x,
                                              grid_dim.y,
                                              grid_dim.z,
                                              block_dim.x,
                                              block_dim.y,
                                              block_dim.z,
                                              lds_byte,
                                              stream_config.stream_id_,
                                              nullptr,
                                              reinterpret_cast<void**>(&config)));

        return 0;
    }
#else
    hip_check_error(hipModuleLaunchKernel(kernel_func,
                                          grid_dim.x,
                                          grid_dim.y,
                                          grid_dim.z,
                                          block_dim.x,
                                          block_dim.y,
                                          block_dim.z,
                                          lds_byte,
                                          stream_config.stream_id_,
                                          nullptr,
                                          reinterpret_cast<void**>(&config)));

    return 0;
#endif
}
#endif
