// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/flush_icache.hpp"
namespace ck {
namespace utility {

template <typename Argument>
struct RotatingMemWrapper
{
    using ADataType = decltype(Argument::p_a_grid);
    using BDataType = decltype(Argument::p_b_grid);
    using CDataType = decltype(Argument::p_c_grid);

    RotatingMemWrapper() = delete;
    RotatingMemWrapper(Argument& arg_,
                       std::size_t rotating_count_,
                       std::size_t size_a_,
                       std::size_t size_b_,
                       std::size_t size_c_)
        : arg(arg_),
          rotating_count(rotating_count_),
          size_a(size_a_),
          size_b(size_b_),
          size_c(size_c_)
    {
        p_a_grids.push_back(arg.p_a_grid);
        p_b_grids.push_back(arg.p_b_grid);
        p_c_grids.push_back(arg.p_c_grid);
        for(size_t i = 0; i < rotating_count - 1; i++)
        {
            {
                void* pADeviceBuf;
                hip_check_error(hipMalloc(static_cast<void**>(&pADeviceBuf), size_a_));
                hip_check_error(hipMemcpy(static_cast<void*>(pADeviceBuf),
                                          const_cast<void*>(p_a_grids[0]),
                                          size_a_,
                                          hipMemcpyDeviceToDevice));
                p_a_grids.push_back(pADeviceBuf);
            }

            {
                void* pBDeviceBuf;
                hip_check_error(hipMalloc(static_cast<void**>(&pBDeviceBuf), size_b_));
                hip_check_error(hipMemcpy(static_cast<void*>(pBDeviceBuf),
                                          const_cast<void*>(p_b_grids[0]),
                                          size_b_,
                                          hipMemcpyDeviceToDevice));
                p_b_grids.push_back(pBDeviceBuf);
            }

            {
                void* pCDeviceBuf;
                hip_check_error(hipMalloc(static_cast<void**>(&pCDeviceBuf), size_c_));
                p_c_grids.push_back(pCDeviceBuf);
            }
        }
    }

    void Next()
    {
        if(rotating_count > 1)
        {
            int idx      = iter++ % rotating_count;
            arg.p_a_grid = reinterpret_cast<ADataType>(p_a_grids[idx]);
            arg.p_b_grid = reinterpret_cast<BDataType>(p_b_grids[idx]);
            arg.p_c_grid = reinterpret_cast<CDataType>(p_c_grids[idx]);
        }
    }
    void Print()
    {
        std::cout << "RotatingMemWrapper{" << size_a << "," << size_b << "," << size_c
                  << "} rotating_count: " << rotating_count << std::endl;
    }
    ~RotatingMemWrapper()
    {
        if(rotating_count > 1)
        {
            // restore ptr
            arg.p_a_grid = reinterpret_cast<ADataType>(p_a_grids[0]);
            arg.p_b_grid = reinterpret_cast<BDataType>(p_b_grids[0]);
            arg.p_c_grid = reinterpret_cast<CDataType>(p_c_grids[0]);
#if 0
            // test rotating buffer gemm result
            hip_check_error(hipMemcpy(static_cast<void*>(p_c_grids[0]),
                                      const_cast<void*>(p_c_grids[--iter % rotating_count]),
                                      size_c,
                                      hipMemcpyDeviceToDevice));
#endif
            // free device mem
            for(size_t i = 1; i < rotating_count; i++)
            {
                hip_check_error(hipFree(const_cast<void*>(p_a_grids[i])));
                hip_check_error(hipFree(const_cast<void*>(p_b_grids[i])));
                hip_check_error(hipFree(p_c_grids[i]));
            }
        }
    }

    private:
    Argument& arg;
    std::size_t iter           = 0;
    std::size_t rotating_count = 1;
    std::size_t size_a         = 0;
    std::size_t size_b         = 0;
    std::size_t size_c         = 0;
    std::vector<const void*> p_a_grids;
    std::vector<const void*> p_b_grids;
    std::vector<void*> p_c_grids;
};

static void flush_icache()
{
    hipDeviceProp_t deviceProps;
    hip_check_error(hipGetDeviceProperties(&deviceProps, 0));
    int32_t gpu_block3 = deviceProps.multiProcessorCount * 60;

    int flush_iter = 200;

    for(int i = 0; i < flush_iter; i++)
    {
        ck::flush_icache<<<dim3(gpu_block3), dim3(64), 0, nullptr>>>();
    }
    hip_check_error(hipGetLastError());
}
// if TimePrePress == false, return time does not include preprocess's time
template <bool TimePreprocess, typename Args, typename F, typename PreProcessFunc>
float launch_and_time_kernel_with_preprocess(const StreamConfig& stream_config,
                                             PreProcessFunc preprocess,
                                             F kernel,
                                             dim3 grid_dim,
                                             dim3 block_dim,
                                             std::size_t lds_byte,
                                             Args& args)
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
        // warm up
        for(int i = 0; i < stream_config.cold_niters_; ++i)
        {
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
            if constexpr(!TimePreprocess)
            {
                preprocess();
            }

            hipEvent_t start, stop;

            hip_check_error(hipEventCreate(&start));
            hip_check_error(hipEventCreate(&stop));

            hip_check_error(hipDeviceSynchronize());
            hip_check_error(hipEventRecord(start, stream_config.stream_id_));
            // calculate preprocess time
            if constexpr(TimePreprocess)
            {
                preprocess();
            }
            // run real kernel
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args);
            hip_check_error(hipGetLastError());
            // end real kernel

            hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
            hip_check_error(hipEventSynchronize(stop));
            float cur_time = 0;
            hip_check_error(hipEventElapsedTime(&cur_time, start, stop));
            total_time += cur_time;
#if 0            
            std::cout << "i: " << i << " cur_time: " << cur_time << " total_time:" << total_time
                      << std::endl;

            printf("args.p_a_grid: %p, args.p_b_grid:%p, args.p_c_grid: %p\n",
                       static_cast<const void*>(args.p_a_grid),
                       static_cast<const void*>(args.p_b_grid),
                       static_cast<void*>(args.p_c_grid));
#endif
        }

        return total_time / nrepeat;
    }
    else
    {
        preprocess();
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

} // namespace utility
} // namespace ck
